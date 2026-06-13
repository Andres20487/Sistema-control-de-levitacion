# =============================================================================
# main.py — Levitador Neumático | ESP32 DEVKIT V1 | MicroPython | v7 MAESTRO
# -----------------------------------------------------------------------------
# FIRMWARE MULTIPLEXOR DINÁMICO: cuatro cerebros de IA embebidos, setpoint
# dinámico comandado por lenguaje natural desde el Dashboard.
#
#   PC -> ESP32 (TRAMA DE CONFIGURACIÓN):  "<SET:15.0,ALGO:1>\n"
#       SET  : nuevo setpoint en cm (saturado a [12.0, 28.0], envolvente
#              operativa con márgenes >= 7 cm a los límites de protección).
#       ALGO : 1 = Lógica Difusa TSK    (25 reglas, sintonización v3)
#              2 = Red Neuronal MLP     (clon del difuso, Etapa 2)
#              3 = Q-Learning Tabular   (política experta 7x7 O(1))
#              4 = Deep RL (DQN Goal-Conditioned, Etapa final)
#       La trama es CONFIGURACIÓN del piloto autónomo: NO conmuta a MODO PC
#       (eso lo hacen los deltas crudos del protocolo legado) y puede ser
#       parcial: "<SET:18.5>" o "<ALGO:4>".
#
#   PC -> ESP32 (PROTOCOLO LEGADO, Etapas 1-2): "<entero>\n" Delta PWM crudo
#       o "A<idx>\n". Mientras lleguen a >5 Hz, el firmware obedece a la PC
#       (MODO PC); si callan > 200 ms, vuelve al piloto autónomo.
#
#   ESP32 -> PC : "{:.2f}\n" posición (cm) cada Ts = 50 ms exactos, y
#                 "# ...\n" diagnósticos/ACK (el dashboard los muestra).
#
# DINÁMICA DEL SETPOINT (variable global, antes fija en 21.0):
#   - `setpoint_objetivo` salta al valor comandado; `setpoint_actual` se
#     desliza hacia él en rampa de 0.8 cm/s (sin escalones que exciten
#     dinámica espuria).
#   - CONTINUIDAD DE LA DERIVADA: el historial de errores se desplaza
#     algebraicamente en cada paso de rampa (e_hist -= d_setpoint), de modo
#     que la derivada refleja SOLO el movimiento físico de la bola y no el
#     movimiento de la referencia. Sin esto, cada cambio de setpoint
#     inyectaría un pico falso de velocidad al controlador activo.
#
# El error del lazo SIEMPRE es:  error = posicion - setpoint_actual.
#
# Paradigma intacto: NUNCA viaja PWM absoluto; pwm = PWM_BASE + delta,
# saturado a [0, 65535]. Protecciones físicas con prioridad absoluta.
# =============================================================================

import sys
import gc
import time
import uselect
from machine import Pin, PWM

from filtros import SensorPosicion

# --------------------------- Configuración de pines --------------------------
PIN_TRIG = 25            # HC-SR04 Trig
PIN_ECHO = 26            # HC-SR04 Echo
PIN_VENTILADOR = 27      # Gate del MOSFET de potencia

# ------------------------------ Parámetros PWM -------------------------------
PWM_FREQ_HZ = 1000       # 1 kHz: zona de conmutación segura del MOSFET
PWM_MIN = 0
PWM_MAX = 65535

# >>> CALIBRAR (REPL: ventilador.duty_u16(X) hasta que la bola flote sola)
# >>> y mantener idéntico en los scripts de PC que usen el protocolo legado.
PWM_BASE = 36850

# ------------------------------ Planta / Lazo --------------------------------
LONGITUD_TUBO_CM = 38.0
SETPOINT_DEFECTO_CM = 21.0
SETPOINT_MIN_CM = 12.0   # Envolvente operativa comandable
SETPOINT_MAX_CM = 28.0
RAMPA_SETPOINT_CM_CICLO = 0.04        # 0.04 cm/ciclo = 0.8 cm/s
SENSOR_EN_TAPA = True
TS_MS = 50               # Tiempo de muestreo exacto: 50 ms (20 Hz)
TS_S = 0.05

# --------------------------- Protecciones físicas ----------------------------
POS_TECHO_CM = 32.0
POS_SUELO_CM = 6.0
T_PROTECCION_TECHO_MS = 300
T_PROTECCION_SUELO_MS = 200
PWM_RESCATE = PWM_BASE + 5000 if PWM_BASE + 5000 <= PWM_MAX else PWM_MAX

PROT_NINGUNA = 0
PROT_TECHO = 1
PROT_SUELO = 2

# ------------------------- Enlace serial con la PC ---------------------------
TIMEOUT_PC_MS = 200                       # Deltas crudos mudos => AUTÓNOMO
ACCIONES = (-2500, -500, 0, 500, 2500)    # Protocolo legado "A<idx>"
MAX_LARGO_LINEA = 32                      # Cabe "<SET:28.0,ALGO:4>" holgado
DEPURAR_RX = False                        # ACK por delta crudo (diagnóstico)

# --------------------------- Pilotos autónomos (IA) --------------------------
DELTA_AUTONOMO_MAX = 4000                 # Blindaje de autoridad común
N_DERIVADA = 3                            # Derivada por ventana (3 muestras)
ALFA_DERIVADA = 0.6                       # EMA (idéntico a los sandboxes)
ALGO_DEFECTO = 3                          # Arranque: Q-Learning 7x7 (probado)
ALGO_NOMBRES = ("?", "Logica Difusa TSK", "Red Neuronal MLP",
                "Q-Learning 7x7", "Deep RL DQN")

# ----------------------- Amortiguador del actuador ---------------------------
ALFA_ACTUADOR = 0.5     # pwm += a*(objetivo-pwm); 1.0 = sin suavizado;
                         # <0.4 PROHIBIDO (anula el amortiguamiento derivativo)

# ------------------------------- Mantenimiento -------------------------------
GC_CADA_N_CICLOS = 100

# ----------------- VARIABLES GLOBALES DINÁMICAS (requisito) ------------------
setpoint_objetivo = SETPOINT_DEFECTO_CM   # Lo escribe la trama <SET:...>
setpoint_actual = SETPOINT_DEFECTO_CM     # Rampa suave hacia el objetivo
algo_activo = ALGO_DEFECTO                # Lo escribe la trama <ALGO:...>

# =============================================================================
#            ACTUADOR EN ÁMBITO GLOBAL (accesible desde el REPL)
# =============================================================================
ventilador = PWM(Pin(PIN_VENTILADOR), freq=PWM_FREQ_HZ)
ventilador.duty_u16(PWM_BASE)


# =============================================================================
# ALGO 1 — LÓGICA DIFUSA TSK EMBEBIDA (25 reglas, sintonización v3)
# -----------------------------------------------------------------------------
# Versión Takagi-Sugeno del Mamdani v3 de la PC: mismas funciones de
# membresía y misma matriz FAM, pero consecuentes SINGLETON en los picos de
# los conjuntos de salida y defuzzificación por promedio ponderado. Costo:
# O(25) por ciclo en lugar del centroide discreto de 321 puntos — la
# elección correcta para MicroPython a 20 Hz.
# =============================================================================

def _trapecio(x, a, b, c, d):
    """Membresía trapezoidal; b == c degenera en triángulo; a == b o c == d
    crean hombros saturados."""
    if x <= a:
        return 1.0 if a == b else 0.0
    if x >= d:
        return 1.0 if c == d else 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


# Conjuntos NG, NP, ZE, PP, PG (índices 0..4), sintonización v3 anti-osc.
_FZ_MF_E = (
    (-40.0, -40.0, -8.0, -3.0), (-6.0, -3.0, -3.0, 0.0),
    (-1.5, 0.0, 0.0, 1.5), (0.0, 3.0, 3.0, 6.0), (3.0, 8.0, 40.0, 40.0))
_FZ_MF_DE = (
    (-100.0, -100.0, -8.0, -4.0), (-5.0, -2.5, -2.5, 0.0),
    (-3.0, 0.0, 0.0, 3.0), (0.0, 2.5, 2.5, 5.0), (4.0, 8.0, 100.0, 100.0))
_FZ_SINGLETON = (-1200.0, -525.0, 0.0, 525.0, 1200.0)
# Matriz FAM (fila = conjunto del error, col = conjunto de la derivada):
_FZ_REGLAS = ((4, 4, 4, 3, 2),
              (4, 3, 3, 2, 1),
              (3, 3, 2, 1, 1),
              (3, 2, 1, 1, 0),
              (2, 1, 0, 0, 0))
# Buffers pre-asignados: cero allocations por inferencia.
_FZ_MU_E = [0.0] * 5
_FZ_MU_DE = [0.0] * 5


def calcular_pwm_difuso(error, derivada):
    """Inferencia TSK: w_r = min(mu_e, mu_de); salida = Σ w_r·s_r / Σ w_r.
    Retorna Delta PWM (int) en [-1200, +1200]."""
    for i in range(5):
        _FZ_MU_E[i] = _trapecio(error, _FZ_MF_E[i][0], _FZ_MF_E[i][1],
                                _FZ_MF_E[i][2], _FZ_MF_E[i][3])
        _FZ_MU_DE[i] = _trapecio(derivada, _FZ_MF_DE[i][0], _FZ_MF_DE[i][1],
                                 _FZ_MF_DE[i][2], _FZ_MF_DE[i][3])
    num = 0.0
    den = 0.0
    for i in range(5):
        ge = _FZ_MU_E[i]
        if ge <= 0.0:
            continue
        fila = _FZ_REGLAS[i]
        for j in range(5):
            gde = _FZ_MU_DE[j]
            if gde <= 0.0:
                continue
            w = ge if ge < gde else gde
            num += w * _FZ_SINGLETON[fila[j]]
            den += w
    if den <= 0.0:
        return 0
    return int(num / den)


# =============================================================================
# ALGO 2 — RED NEURONAL MLP (generado por entrenar_ia.py, Etapa 2)
# -----------------------------------------------------------------------------
# *** AVISO: entrenada con la dinámica de la Etapa 1. Re-cosechar y
# *** re-entrenar tras cambios físicos mayores; el recorte anti-extrapolación
# *** y la saturación la mantienen segura entretanto.
# =============================================================================
import math as _math_ia

_IA_MU_E = 1.331339388
_IA_SD_E = 2.578572022
_IA_MU_DE = 0.027938275
_IA_SD_DE = 2.811803994
_IA_MU_Y = -470.30475
_IA_SD_Y = 1135.410525
_IA_W1_E = (-0.1884030654, 1.895357136, -0.3892223925, 0.203627827, -0.4804844172, 0.5390129901)
_IA_W1_DE = (-3.534181164, 1.432918541, 0.01593976572, 0.10099036, -2.7868068, 2.825331143)
_IA_B1 = (7.08606627, 0.8955079723, -2.674295394, 1.196470273, -1.479641256, 1.774423414)
_IA_W2 = (0.3644208957, -0.6201393208, -2.683986602, -4.119077522, 2.663656403, 2.358563963)
_IA_B2 = 0.6196546213
_IA_E_MIN = -17.99
_IA_E_MAX = 7.7881
_IA_DE_MIN = -13.0221
_IA_DE_MAX = 13.2309
_IA_DELTA_MAX = 4000


def calcular_pwm_ia(error, delta_error):
    """Inferencia MLP 2-6-1 (tanh) en MicroPython puro. Delta PWM (int)."""
    if error < _IA_E_MIN:
        error = _IA_E_MIN
    elif error > _IA_E_MAX:
        error = _IA_E_MAX
    if delta_error < _IA_DE_MIN:
        delta_error = _IA_DE_MIN
    elif delta_error > _IA_DE_MAX:
        delta_error = _IA_DE_MAX
    xe = (error - _IA_MU_E) / _IA_SD_E
    xde = (delta_error - _IA_MU_DE) / _IA_SD_DE
    acumulado = _IA_B2
    for i in range(6):
        acumulado += _IA_W2[i] * _math_ia.tanh(
            _IA_W1_E[i] * xe + _IA_W1_DE[i] * xde + _IA_B1[i])
    delta = acumulado * _IA_SD_Y + _IA_MU_Y
    if delta > _IA_DELTA_MAX:
        delta = _IA_DELTA_MAX
    elif delta < -_IA_DELTA_MAX:
        delta = -_IA_DELTA_MAX
    return int(delta)


# =============================================================================
# ALGO 3 — Q-LEARNING TABULAR (política experta estática 7x7, Etapa 4)
# -----------------------------------------------------------------------------
# 49 estados con umbrales quirúrgicos no lineales; la política contiene
# ÍNDICES de acción (0..4) resueltos contra el catálogo asimétrico.
# =============================================================================
ACCIONES_PWM = (-1000, -150, 0, 180, 1200)

#                 derivada:  <-4.5  -4.5/-2 -2/-0.4 -.4/.4  .4/2   2/4.5  >4.5
_QL_POLITICA = (
    4,  4,  4,  4,  3,  3,  2,   # ie=0: e < -3.0     (muy abajo)
    4,  4,  4,  3,  3,  2,  1,   # ie=1: -3.0 .. -1.5
    4,  4,  3,  3,  2,  2,  1,   # ie=2: -1.5 .. -0.3
    4,  3,  3,  2,  1,  1,  0,   # ie=3: -0.3 .. +0.3 (setpoint)
    3,  2,  2,  1,  1,  0,  0,   # ie=4: +0.3 .. +1.5
    3,  2,  1,  1,  0,  0,  0,   # ie=5: +1.5 .. +3.0
    2,  1,  1,  0,  0,  0,  0,   # ie=6: e > +3.0     (muy arriba)
)


def calcular_pwm_qlearning(error, derivada):
    """Inferencia tabular O(1): discretiza (7x7) e indexa la política.
    Retorna Delta PWM (int) en [-1000, +1200]."""
    if error < -3.0:
        ie = 0
    elif error < -1.5:
        ie = 1
    elif error < -0.3:
        ie = 2
    elif error <= 0.3:
        ie = 3
    elif error <= 1.5:
        ie = 4
    elif error <= 3.0:
        ie = 5
    else:
        ie = 6
    if derivada < -4.5:
        ide = 0
    elif derivada < -2.0:
        ide = 1
    elif derivada < -0.4:
        ide = 2
    elif derivada <= 0.4:
        ide = 3
    elif derivada <= 2.0:
        ide = 4
    elif derivada <= 4.5:
        ide = 5
    else:
        ide = 6
    return ACCIONES_PWM[_QL_POLITICA[ie * 7 + ide]]


# =============================================================================
# ALGO 4 — DEEP RL (DQN Goal-Conditioned)
# ===== INICIO BLOQUE DQN (generado por entrenar_dqn.py) ==============
# DQN Goal-Conditioned 3-16-16-5 (ReLU). Entradas: error (cm),
# derivada (cm/s), setpoint (cm). Salida: Delta PWM del catalogo
# asimetrico ACCIONES_DQN = argmax_a Q(s, a). Sin numpy/torch.

ACCIONES_DQN = (-1000, -150, 0, 180, 1200)

_DQN_W1 = (
    (-3.7694485, -1.0075201, -0.18454511),
    (0.70095187, -2.183677, -0.40677524),
    (1.5166996, -1.345849, -0.14813454),
    (-4.1605482, 0.093741409, 0.080271445),
    (-0.0097037135, 0.0069567361, 0.07507354),
    (-1.734828, -1.4706036, 0.071174666),
    (8.6887865, 2.4069552, -0.04153499),
    (-2.2020938, 0.51705921, 0.66043466),
    (-2.2087817, -2.2434835, 0.26612684),
    (-3.0975282, -2.9053693, 0.031761326),
    (-7.0477691, -0.28830922, -0.0030303565),
    (-1.2661252, 3.0816545, -0.095927984),
    (-0.44217372, -1.0339649, 0.9458499),
    (-3.6442256, -1.3219861, 0.59417748),
    (6.9238076, -1.2478658, -0.029944919),
    (-1.7664728, -2.6901879, -0.464091),
)
_DQN_B1 = (-1.6904589, -1.3261179, 1.2150295, 1.4132787, -0.47051352, 1.7516963, -0.96835643, 1.2495496, 0.53315872, -1.0190883, -0.57340163, -0.18075666, 0.92921472, 0.47610798, -0.37512034, 0.56729543)
_DQN_W2 = (
    (-1.6804569, -0.50728798, 0.40958366, 0.39626944, -0.008191254, -0.68076473, 2.6595285, 0.33239436, 1.0495043, 0.90412557, 0.65356362, -2.2833602, -0.72451973, 0.31123382, -1.5407172, -0.2901974),
    (-1.6294063, -9.6519547, 0.79190743, 0.55096763, -0.027808458, 0.37664112, -23.336922, 0.11183931, -3.4711194, -17.181709, -2.1336071, -0.1673775, 0.15766262, 2.2237046, -1.4748507, 1.8276005),
    (-1.6457133, -0.31023911, 0.11418839, 0.064601012, -0.045241039, -0.33622396, 2.8253403, 0.41354308, 0.71232891, 0.71840185, 0.45621058, -1.8229053, -0.64708197, 0.70468122, -2.0534961, -0.012024985),
    (-1.6787535, -0.68439239, 0.31803048, 0.29100397, -0.19465035, -0.80084527, 2.5797794, 0.41108561, 1.0326688, 1.0145891, 0.75128096, -2.4883401, -0.23938455, 0.20671339, -1.8543783, -0.20613386),
    (-2.1063263, -13.653643, 1.6246473, 1.0120885, 0.2432327, 1.7490892, -22.254026, 0.51596254, -4.1668453, -17.424135, -26.975904, -4.4626665, 0.83767122, 0.75144553, -5.6057072, 1.7018224),
    (-10.812363, -0.28709379, 0.91308606, -0.28530118, 0.27510265, -0.11988372, 2.1217933, -0.36876765, 0.72190565, 1.4252925, 2.4240739, -1.8295484, -0.036621947, -0.82379329, 3.5737288, 0.39307269),
    (-0.59198248, -11.704309, 1.0021538, 1.3300201, 0.083375543, 1.606626, -3.2923472, 1.3638169, -4.6772213, -7.4086585, -13.973638, 1.2719163, 0.73380995, 1.2443674, 0.94753891, -0.22254474),
    (-1.5644153, -0.28620288, 0.58142066, -0.040667538, 0.16489078, -0.45010257, 2.6662004, 0.020634022, 0.65114677, 0.67004067, 0.8415693, -1.8662107, -0.44496226, 0.60422981, -2.0295603, -0.22432098),
    (-0.58732694, -2.7546859, -2.5576839, 2.7852345, 0.025726138, -4.8400478, 0.70091724, 1.6546328, 0.41748798, 2.7004266, 1.2986978, 1.5528172, -1.2280453, -0.57240534, 0.81139332, -1.8060664),
    (-1.4895844, -0.19006157, -0.069231272, 0.66136765, -0.19936411, -0.33176395, 2.5901258, 0.60701847, 0.59078467, 0.80636346, 1.2829072, -1.2899984, -0.56675273, 0.75640762, -1.8488541, -0.33457661),
    (-0.17134646, 0.08652854, -0.14295635, 0.025791943, 0.20734382, -0.20720223, -0.18191907, 0.15166119, 0.13535437, -0.17932436, 0.10083354, -0.16909286, -0.21012971, -0.10752088, 0.16732404, 0.022096008),
    (-0.97924697, 0.0079212477, 1.373457, 0.52176058, 0.10451201, 1.3622764, -10.425189, 0.34837854, 0.3638027, -0.054783832, 1.1443568, -5.6296258, -0.0071413675, 0.024151919, 1.7275076, 0.045025226),
    (-0.98374408, -1.19707, -1.6449933, -6.4824672, -0.055178195, 2.8769422, 2.2471459, -2.198642, 1.8494408, -1.601154, -5.0864344, 1.3702638, 2.4753647, 1.2903967, 3.2502515, 2.4826741),
    (0.23168185, -3.4829545, 0.83339947, 1.9740832, 0.051041216, 1.2574662, -5.2816916, 1.3923811, -1.2132661, -2.8105841, -2.2677374, 2.1139069, 1.3228716, -0.43505824, 0.38489902, 0.32106474),
    (-2.2291481, -12.337399, 1.8510126, 1.1087476, -0.17468065, 1.9134123, -17.649572, 0.69236702, -4.0932474, -16.238541, -27.618835, -4.0345364, 0.88820821, 0.3872779, -6.1134748, 1.5297787),
    (-1.4894145, -12.887887, 1.5540614, 1.2390462, 0.036665738, 1.4443256, -20.772366, 0.71398503, -4.8521953, -16.54401, -22.871994, -4.3895268, 0.8275106, 0.35077, -8.7693081, 1.1380013),
)
_DQN_B2 = (-0.070602752, -1.1879538, -0.31005499, -0.028487992, 1.2121913, 0.59231019, 1.2148215, -0.11206836, -0.42525801, -0.029688872, -0.14284936, 1.3221912, -0.91455925, 3.5952859, 1.6456035, 1.4074286)
_DQN_W3 = (
    (-1.2699662, 4.5907359, -1.855473, -1.5536737, 9.9324684, -2.6024995, 4.3870897, -1.6409754, -1.7348318, -1.2332203, 0.085972458, -0.91124189, 3.080795, 2.832649, 10.187439, 9.904273),
    (-1.6710734, 5.5525966, -1.6424695, -1.0904289, 10.099079, -2.4956474, 4.0324798, -1.7165101, -1.5827979, -1.5050472, -0.017681569, -0.81854796, 3.0204973, 2.8261833, 10.121129, 10.05169),
    (-1.7392955, 5.8104091, -1.4352742, -1.5077739, 10.209365, -2.4242299, 4.099246, -1.4390503, -1.5816762, -1.4202528, 0.21834192, -0.81188661, 2.9175479, 2.809489, 10.163686, 9.8204641),
    (-1.6076641, 5.7006726, -1.3771485, -1.511153, 10.151989, -2.4646373, 4.1110835, -1.3491181, -1.6010883, -1.4391196, 0.1744768, -0.8536101, 2.899971, 2.8421621, 10.180678, 9.8225355),
    (-1.3727219, 5.832304, -1.685426, -1.7562863, 10.780025, -2.4449093, 4.1384702, -1.4945664, -1.6393893, -1.2567947, -0.20404121, -0.79402727, 2.9379518, 2.8353262, 9.71138, 9.6802273),
)
_DQN_B3 = (2.8024054, 2.3712764, 2.0260098, 1.884227, 1.7533259)

# Buffers pre-asignados: cero allocations por inferencia (RAM ESP32)
_DQN_H1 = [0.0] * 16
_DQN_H2 = [0.0] * 16


def calcular_pwm_dqn(error, derivada, setpoint):
    """Inferencia DQN en MicroPython puro: ~384 MACs (~5 ms a 240 MHz).
    Retorna Delta PWM (int) del catalogo ACCIONES_DQN."""
    # Normalizacion (constantes horneadas del entrenamiento)
    x0 = error / 10
    x1 = derivada / 8
    x2 = (setpoint - 21) / 9
    # Capa 1: Linear(3->16) + ReLU
    for i in range(16):
        w = _DQN_W1[i]
        v = w[0] * x0 + w[1] * x1 + w[2] * x2 + _DQN_B1[i]
        _DQN_H1[i] = v if v > 0.0 else 0.0
    # Capa 2: Linear(16->16) + ReLU
    for i in range(16):
        w = _DQN_W2[i]
        v = _DQN_B2[i]
        for j in range(16):
            v += w[j] * _DQN_H1[j]
        _DQN_H2[i] = v if v > 0.0 else 0.0
    # Capa 3: Linear(16->5) + argmax
    mejor_a = 0
    mejor_q = -1e30
    for i in range(5):
        w = _DQN_W3[i]
        v = _DQN_B3[i]
        for j in range(16):
            v += w[j] * _DQN_H2[j]
        if v > mejor_q:
            mejor_q = v
            mejor_a = i
    return ACCIONES_DQN[mejor_a]
# ===== FIN BLOQUE DQN =================================================


def _clamp_pwm(valor):
    """Saturación ESTRICTA del duty al rango físico de 16 bits (entero)."""
    if valor < PWM_MIN:
        return PWM_MIN
    if valor > PWM_MAX:
        return PWM_MAX
    return int(valor)


def _clamp_delta_autonomo(valor):
    """Blindaje de autoridad común a los cuatro cerebros."""
    if valor > DELTA_AUTONOMO_MAX:
        return DELTA_AUTONOMO_MAX
    if valor < -DELTA_AUTONOMO_MAX:
        return -DELTA_AUTONOMO_MAX
    return valor


def _decodificar_delta(linea):
    """Protocolo legado: '<entero>' o 'A<idx>' -> Delta PWM (int) | None."""
    c0 = linea[0]
    if c0 == 'A' or c0 == 'a':
        try:
            idx = int(linea[1:].strip())
        except ValueError:
            return None
        if 0 <= idx < len(ACCIONES):
            return ACCIONES[idx]
        return None
    try:
        delta = int(linea)
    except ValueError:
        try:
            delta = int(float(linea))
        except ValueError:
            return None
    if delta > PWM_MAX:
        delta = PWM_MAX
    elif delta < -PWM_MAX:
        delta = -PWM_MAX
    return delta


def _decodificar_trama(linea):
    """Trama de configuración '<SET:X,ALGO:Y>' (admite campos parciales).

    Retorna (setpoint | None, algo | None), o None si la trama es inválida.
    Robustez: campos desconocidos, valores corruptos o fuera de rango se
    descartan campo a campo sin invalidar el resto de la trama.
    """
    if len(linea) < 3 or linea[0] != '<' or linea[-1] != '>':
        return None
    sp = None
    algo = None
    for campo in linea[1:-1].split(','):
        kv = campo.split(':')
        if len(kv) != 2:
            continue
        clave = kv[0].strip().upper()
        valor = kv[1].strip()
        if clave == 'SET':
            try:
                x = float(valor)
            except ValueError:
                continue
            # Saturación a la envolvente operativa comandable.
            if x < SETPOINT_MIN_CM:
                x = SETPOINT_MIN_CM
            elif x > SETPOINT_MAX_CM:
                x = SETPOINT_MAX_CM
            sp = x
        elif clave == 'ALGO':
            try:
                a = int(valor)
            except ValueError:
                continue
            if 1 <= a <= 4:
                algo = a
    if sp is None and algo is None:
        return None
    return (sp, algo)


def ejecutar_lazo():
    """Bucle de control principal a 20 Hz: multiplexor de 4 cerebros con
    setpoint dinámico. Usa el `ventilador` GLOBAL (sin re-instanciar PWM)."""
    global setpoint_objetivo, setpoint_actual, algo_activo

    ventilador.duty_u16(PWM_BASE)

    # ----- Sensor + cascada de filtros aligerada (Mediana 3 -> Prom. 2) -----
    sensor = SensorPosicion(
        pin_trig=PIN_TRIG,
        pin_echo=PIN_ECHO,
        longitud_tubo_cm=LONGITUD_TUBO_CM,
        sensor_en_tapa=SENSOR_EN_TAPA,
        valor_inicial_cm=setpoint_actual,
        ventana_mediana=3,
        ventana_promedio=2,
    )
    sensor.calibrar_inicial()

    # ----- Entrada serial NO bloqueante sobre stdin (USB/UART0) -----
    sondeo = uselect.poll()
    sondeo.register(sys.stdin, uselect.POLLIN)
    buffer_rx = ""

    # ----- Estado del lazo -----
    delta_pwm_pc = 0
    t_ultimo_rx = time.ticks_ms() - 10 * TIMEOUT_PC_MS   # Arranca autónomo
    modo_proteccion = PROT_NINGUNA
    t_fin_proteccion = time.ticks_ms()
    en_modo_autonomo = True
    ciclos = 0
    pwm_suavizado = float(PWM_BASE)

    # ----- Derivada por ventana (siempre "caliente") -----
    hist_error = [0.0] * (N_DERIVADA + 1)
    i_hist = 0
    muestras_hist = 0
    d_error_filtrada = 0.0

    print("# LEVITADOR LISTO v7-MAESTRO Ts=50ms PWM_BASE={} SET={} ALGO={} "
          "({})".format(PWM_BASE, setpoint_actual, algo_activo,
                        ALGO_NOMBRES[algo_activo]))
    print("# MODO AUTONOMO ({} al mando)".format(ALGO_NOMBRES[algo_activo]))

    gc.collect()
    t_deadline = time.ticks_add(time.ticks_ms(), TS_MS)

    while True:
        # ================= 1) ADQUISICIÓN Y FILTRADO =================
        posicion = sensor.leer_cm()

        # ---------- 1a) Rampa del setpoint dinámico ------------------
        # El objetivo salta por trama; el setpoint efectivo se desliza a
        # 0.8 cm/s. El historial de errores se desplaza por el mismo paso
        # para que la derivada NO vea el movimiento de la referencia.
        if setpoint_actual != setpoint_objetivo:
            paso_sp = setpoint_objetivo - setpoint_actual
            if paso_sp > RAMPA_SETPOINT_CM_CICLO:
                paso_sp = RAMPA_SETPOINT_CM_CICLO
            elif paso_sp < -RAMPA_SETPOINT_CM_CICLO:
                paso_sp = -RAMPA_SETPOINT_CM_CICLO
            setpoint_actual += paso_sp
            for k in range(N_DERIVADA + 1):
                hist_error[k] -= paso_sp

        # ---------- 1b) Error y derivada por ventana -----------------
        error = posicion - setpoint_actual
        error_antiguo = hist_error[i_hist]       # e[k - N_DERIVADA]
        hist_error[i_hist] = error
        i_hist += 1
        if i_hist > N_DERIVADA:
            i_hist = 0
        muestras_hist += 1
        if muestras_hist > N_DERIVADA:
            d_error_cruda = (error - error_antiguo) / (N_DERIVADA * TS_S)
        else:
            d_error_cruda = 0.0
        d_error_filtrada = (ALFA_DERIVADA * d_error_cruda
                            + (1.0 - ALFA_DERIVADA) * d_error_filtrada)

        # ================= 2) TELEMETRÍA HACIA LA PC =================
        print("{:.2f}".format(posicion))

        # ============ 3) RECEPCIÓN NO BLOQUEANTE DESDE LA PC =========
        while sondeo.poll(0):
            ch = sys.stdin.read(1)
            if not ch:
                break
            if ch == '\n' or ch == '\r':
                if buffer_rx:
                    linea = buffer_rx.strip().strip("\x00").strip()
                    buffer_rx = ""
                    if not linea:
                        pass
                    elif linea[0] == '<':
                        # ---- TRAMA DE CONFIGURACIÓN (no conmuta modo) ----
                        cfg = _decodificar_trama(linea)
                        if cfg is not None:
                            sp_nuevo, algo_nuevo = cfg
                            if sp_nuevo is not None:
                                setpoint_objetivo = sp_nuevo
                            if algo_nuevo is not None and \
                                    algo_nuevo != algo_activo:
                                algo_activo = algo_nuevo
                            print("# CFG SET={:.2f} ALGO={} ({})".format(
                                setpoint_objetivo, algo_activo,
                                ALGO_NOMBRES[algo_activo]))
                    else:
                        # ---- PROTOCOLO LEGADO: delta crudo (MODO PC) ----
                        nuevo_delta = _decodificar_delta(linea)
                        if nuevo_delta is not None:
                            delta_pwm_pc = nuevo_delta
                            t_ultimo_rx = time.ticks_ms()
                            if DEPURAR_RX:
                                print("# RX dPWM={} PWM={}".format(
                                    delta_pwm_pc,
                                    _clamp_pwm(PWM_BASE + delta_pwm_pc)))
            else:
                buffer_rx += ch
                if len(buffer_rx) > MAX_LARGO_LINEA:
                    buffer_rx = ""

        ahora = time.ticks_ms()

        # ================= 4) MÁQUINA DE PROTECCIONES ================
        if modo_proteccion != PROT_NINGUNA and \
                time.ticks_diff(ahora, t_fin_proteccion) >= 0:
            modo_proteccion = PROT_NINGUNA

        if modo_proteccion == PROT_NINGUNA:
            if posicion >= POS_TECHO_CM:
                modo_proteccion = PROT_TECHO
                t_fin_proteccion = time.ticks_add(ahora, T_PROTECCION_TECHO_MS)
            elif posicion <= POS_SUELO_CM:
                modo_proteccion = PROT_SUELO
                t_fin_proteccion = time.ticks_add(ahora, T_PROTECCION_SUELO_MS)

        # ============== 5) SELECCIÓN DE MODO: PC vs AUTÓNOMO =========
        pc_viva = time.ticks_diff(ahora, t_ultimo_rx) <= TIMEOUT_PC_MS
        if pc_viva and en_modo_autonomo:
            en_modo_autonomo = False
            print("# MODO PC (deltas crudos al mando)")
        elif not pc_viva and not en_modo_autonomo:
            en_modo_autonomo = True
            delta_pwm_pc = 0
            print("# MODO AUTONOMO ({} al mando)".format(
                ALGO_NOMBRES[algo_activo]))

        # ====== 6) SELECTOR DE ALGORITMO + APLICACIÓN + AMORTIGUADOR =
        if modo_proteccion == PROT_TECHO:
            pwm_suavizado = float(PWM_MIN)
        elif modo_proteccion == PROT_SUELO:
            pwm_suavizado = float(PWM_RESCATE)
        else:
            if en_modo_autonomo:
                # -------- MULTIPLEXOR DE CEREBROS DE IA --------
                if algo_activo == 1:
                    delta = calcular_pwm_difuso(error, d_error_filtrada)
                elif algo_activo == 2:
                    delta = calcular_pwm_ia(error, d_error_filtrada)
                elif algo_activo == 3:
                    delta = calcular_pwm_qlearning(error, d_error_filtrada)
                else:                              # algo_activo == 4
                    delta = calcular_pwm_dqn(error, d_error_filtrada,
                                             setpoint_actual)
                pwm_objetivo = PWM_BASE + _clamp_delta_autonomo(delta)
            else:
                pwm_objetivo = PWM_BASE + delta_pwm_pc
            pwm_suavizado += ALFA_ACTUADOR * (pwm_objetivo - pwm_suavizado)

        ventilador.duty_u16(_clamp_pwm(pwm_suavizado))

        # ============== 7) MANTENIMIENTO DELIBERADO DE RAM ===========
        ciclos += 1
        if ciclos >= GC_CADA_N_CICLOS:
            ciclos = 0
            gc.collect()

        # ================ 8) SCHEDULING EXACTO A 50 ms ===============
        restante = time.ticks_diff(t_deadline, time.ticks_ms())
        if restante > 0:
            time.sleep_ms(restante)
            t_deadline = time.ticks_add(t_deadline, TS_MS)
        elif restante > -TS_MS:
            t_deadline = time.ticks_add(t_deadline, TS_MS)
        else:
            t_deadline = time.ticks_add(time.ticks_ms(), TS_MS)


def main():
    """Supervisor anti-crash: cualquier excepción lleva el actuador a estado
    seguro sobre el `ventilador` global y relanza el lazo. Ctrl+C apaga el
    motor y sale al REPL con `ventilador` disponible."""
    while True:
        try:
            ejecutar_lazo()
        except KeyboardInterrupt:
            try:
                ventilador.duty_u16(PWM_MIN)
            except Exception:
                pass
            print("# DETENIDO (objeto 'ventilador' disponible en el REPL)")
            return
        except Exception as exc:
            try:
                ventilador.duty_u16(PWM_BASE)
            except Exception:
                pass
            try:
                print("# ERROR {}: reiniciando lazo".format(exc))
            except Exception:
                pass
            gc.collect()
            time.sleep_ms(250)


main()