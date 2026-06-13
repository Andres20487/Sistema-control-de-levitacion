# =============================================================================
# entrenar_dqn.py — Levitador Neumático | Etapa Final (PC)
# -----------------------------------------------------------------------------
# DEEP REINFORCEMENT LEARNING: DQN CONDICIONADO POR OBJETIVOS (Goal-Conditioned)
#
# Diferencia clave con la Q-Table 7x7: la red recibe el SETPOINT DESEADO como
# tercera entrada, así que UNA SOLA política sirve para llevar la bola a
# cualquier altura objetivo, no solo a 21 cm.
#
#   Estado s = [error_cm, derivada_cm_s, setpoint_deseado_cm]
#   Acción a ∈ catálogo asimétrico (-1000, -150, 0, 180, 1200)
#   Salida de la red: Q(s, a) para las 5 acciones; el Delta PWM aplicado es
#   ACCIONES_DQN[argmax_a Q(s, a)].
#
# Rigor matemático (la receta final es resultado de una investigación de
# ablación documentada; cada ingrediente corrige un modo de falla observado):
#
#   1. Objetivo TD con DOUBLE DQN (van Hasselt, 2016):
#        y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
#      El max ingenuo de Bellman SOBREESTIMA cerca de los castigos de -100 y
#      desestabiliza la política.
#   2. Pérdida Huber + Experience Replay (ring buffer) + Target Network.
#   3. Recompensa DENSA: -0.3*|error| fuera de la banda. Sin ella, el agente
#      colapsa al óptimo local "no hacer nada" (el shaping potencial por sí
#      solo da retorno ~0 a quedarse inmóvil lejos del objetivo).
#   4. Shaping basado en potencial Phi(s) = -0.6*|error| (Ng et al., 1999):
#      densifica el gradiente SIN alterar la política óptima.
#   5. TUTOR MEZCLADO: la política de comportamiento intercala acciones del
#      experto tabular 7x7 (probabilidad 0.5 -> 0.05). El Q-learning es
#      off-policy: aprender de un tutor es válido y llena el replay de
#      trayectorias buenas CON acciones diversas. (El pre-entrenamiento
#      offline puro falla por extrapolación: el max bootstrapea acciones
#      que el experto nunca tomó.)
#   6. CURRÍCULUM: 50% de los episodios sortean objetivos ALTOS (22-30 cm),
#      donde el catálogo asimétrico (+1200 de empuje vs -1000 de freno)
#      produce sobreimpulso contra el techo si no se sobre-entrena esa zona.
#
# Fidelidad sim-to-real: la observación pasa por la MISMA cadena del firmware
# (cuantización 0.3 cm, retardos de sensor/actuación, derivada por ventana de
# 3 muestras + EMA 0.8) y cada episodio sortea un desbalance de calibración
# de PWM_BASE de ±300 cuentas.
#
# Al finalizar: evalúa la política greedy (rango completo 12-30 cm y
# envolvente operativa 14-28 cm), IMPRIME LOS PESOS del modelo, transpila la
# red a MicroPython puro (calcular_pwm_dqn), VERIFICA la equivalencia contra
# PyTorch y guarda el bloque en 'dqn_inference_esp32.py'.
#
# Uso:
#   python entrenar_dqn.py
#   python entrenar_dqn.py --episodios 2000 --semilla 7
#
# Dependencias: pip install torch
# =============================================================================

import argparse
import math
import random
import time

import torch
import torch.nn as nn

# ------------------------- Planta / lazo (como el firmware) ------------------
TS_S = 0.050
LIMITE_INFERIOR_CM = 5.0
LIMITE_SUPERIOR_CM = 35.0
N_DERIVADA = 3
ALFA_DERIVADA = 0.8

# ----------------------------- Física simulada -------------------------------
K_PLANTA = 0.012                      # cm/s^2 por cuenta de PWM
C_ARRASTRE = 1.2                      # 1/s (fricción del aire)
TAU_VENTILADOR_S = 0.15               # Inercia de primer orden de la turbina
RETARDO_ACTUACION = 3                 # Muestras de retardo de transporte
RETARDO_SENSOR = 2                    # Muestras de retardo del filtro
CUANTIZACION_CM = 0.3
RUIDO_TURBULENCIA = 2.5               # cm/s^2
DESBALANCE_EQ_MAX = 300               # cuentas

# ------------------------- Objetivos (Goal-Conditioned) ----------------------
SETPOINT_MIN_CM = 12.0
SETPOINT_MAX_CM = 30.0
ENVOLVENTE_MIN_CM = 14.0              # Envolvente operativa recomendada
ENVOLVENTE_MAX_CM = 28.0              # (márgenes >= 7 cm a los límites duros)
CURRICULUM_SP_MIN = 22.0              # Zona de sobremuestreo (objetivos altos)
CURRICULUM_PROB = 0.5

# ------------------------- Acciones (catálogo asimétrico) --------------------
ACCIONES_DQN = (-1000, -150, 0, 180, 1200)
N_ACCIONES = len(ACCIONES_DQN)

# ------------------------ Normalización de entradas --------------------------
NORM_ERROR = 10.0                     # error/10        -> ~[-2, 2]
NORM_DERIVADA = 8.0                   # derivada/8      -> ~[-2, 2]
NORM_SP_CENTRO = 21.0                 # (sp-21)/9       -> [-1, 1]
NORM_SP_ESCALA = 9.0

# ----------------------------- Hiperparámetros DQN ---------------------------
NEURONAS_OCULTAS = 16                 # Red 3-16-16-5: ~400 MACs transpilados
EPISODIOS_DEFECTO = 1200
PASOS_MAX_EPISODIO = 200              # 10 s simulados
GAMMA = 0.97
LR = 1e-3
BATCH = 128
REPLAY_CAPACIDAD = 60000
WARMUP_TRANSICIONES = 2000
APRENDER_CADA = 2
TARGET_UPDATE_CADA = 1000
EPSILON_INI, EPSILON_FIN = 0.20, 0.05
TUTOR_INI, TUTOR_FIN = 0.50, 0.05     # Probabilidad de acción del tutor 7x7
RECOMPENSA_BANDA = 10.0
ZONA_MUERTA_CM = 1.0
CASTIGO_LIMITE = -100.0
PESO_POTENCIAL = 0.6
PESO_PENA_DENSA = 0.3
PESO_LLEGADA_SUAVE = 0.10

EPISODIOS_EVAL = 200
N_CORRIDAS = 2                        # Selección de modelo: la varianza entre
                                      # corridas de DQN es alta (62% vs 81% de
                                      # éxito observados con la misma receta);
                                      # se entrena N veces y se elige la mejor
EPISODIOS_EVAL_SELECCION = 150
ARCHIVO_SALIDA = "dqn_inference_esp32.py"


# =============================================================================
#        TUTOR EXPERTO (política tabular 7x7 de la Etapa 4, embebida)
# =============================================================================

_TUTOR_POLITICA = (
    4, 4, 4, 4, 3, 3, 2,
    4, 4, 4, 3, 3, 2, 1,
    4, 4, 3, 3, 2, 2, 1,
    4, 3, 3, 2, 1, 1, 0,
    3, 2, 2, 1, 1, 0, 0,
    3, 2, 1, 1, 0, 0, 0,
    2, 1, 1, 0, 0, 0, 0,
)


def tutor_experto(error, derivada):
    """Acción del experto tabular 7x7 (índice 0..4 del catálogo)."""
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
    return _TUTOR_POLITICA[ie * 7 + ide]


# =============================================================================
#                ENTORNO SIMULADO CONDICIONADO POR OBJETIVOS
# =============================================================================

class LevitadorEnv:
    """Física básica del tubo (aceleración ∝ PWM, amortiguamiento viscoso,
    límites duros con castigo -100) observada A TRAVÉS de la cadena de
    medición del firmware. Cada reset() sortea un setpoint objetivo."""

    def __init__(self, rng):
        self._rng = rng
        self.setpoint = 21.0
        self.reiniciar()

    def reiniciar(self, setpoint=None):
        rng = self._rng
        self.setpoint = (rng.uniform(SETPOINT_MIN_CM, SETPOINT_MAX_CM)
                         if setpoint is None else setpoint)
        self._z = rng.uniform(8.0, 32.0)
        self._v = rng.uniform(-2.0, 2.0)
        self._desbalance = rng.uniform(-DESBALANCE_EQ_MAX, DESBALANCE_EQ_MAX)
        self._u_ventilador = 0.0
        self._cola_actuacion = [0.0] * RETARDO_ACTUACION
        self._cola_sensor = [self._z] * RETARDO_SENSOR
        self._hist_error = [0.0] * (N_DERIVADA + 1)
        self._i_hist = 0
        self._muestras = 0
        self._de_filtrada = 0.0
        return self._observar()

    def _observar(self):
        self._cola_sensor.append(self._z)
        z_ret = self._cola_sensor.pop(0)
        z_med = round(z_ret / CUANTIZACION_CM) * CUANTIZACION_CM
        error = z_med - self.setpoint
        e_ant = self._hist_error[self._i_hist]
        self._hist_error[self._i_hist] = error
        self._i_hist = (self._i_hist + 1) % (N_DERIVADA + 1)
        self._muestras += 1
        de_cruda = ((error - e_ant) / (N_DERIVADA * TS_S)
                    if self._muestras > N_DERIVADA else 0.0)
        self._de_filtrada = (ALFA_DERIVADA * de_cruda
                             + (1.0 - ALFA_DERIVADA) * self._de_filtrada)
        return error, self._de_filtrada

    def paso(self, delta_pwm):
        """Retorna (error, derivada, recompensa_base, fin)."""
        self._cola_actuacion.append(float(delta_pwm))
        delta_ap = self._cola_actuacion.pop(0)
        self._u_ventilador += (TS_S / TAU_VENTILADOR_S) * (
            delta_ap + self._desbalance - self._u_ventilador)
        acel = (K_PLANTA * self._u_ventilador - C_ARRASTRE * self._v
                + self._rng.gauss(0.0, RUIDO_TURBULENCIA))
        self._v += acel * TS_S
        self._z += self._v * TS_S
        if self._z <= LIMITE_INFERIOR_CM or self._z >= LIMITE_SUPERIOR_CM:
            self._z = max(LIMITE_INFERIOR_CM,
                          min(LIMITE_SUPERIOR_CM, self._z))
            e, de = self._observar()
            return e, de, CASTIGO_LIMITE, True
        e, de = self._observar()
        r = RECOMPENSA_BANDA if abs(e) <= ZONA_MUERTA_CM else 0.0
        return e, de, r, False


def recompensa_total(r_base, e2, de2, e_prev, fin):
    """Recompensa base + pena densa + shaping potencial + llegada suave."""
    r = r_base
    if not fin:
        # Shaping basado en potencial (no altera la política óptima):
        r += GAMMA * (-PESO_POTENCIAL * abs(e2)) \
            - (-PESO_POTENCIAL * abs(e_prev))
        # Pena densa: sin ella el agente colapsa a "no hacer nada":
        if abs(e2) > ZONA_MUERTA_CM:
            r -= PESO_PENA_DENSA * abs(e2)
    if abs(e2) <= 3.0:
        r -= PESO_LLEGADA_SUAVE * abs(de2)
    return r


def a_tensor_estado(error, derivada, setpoint):
    """Vector de estado normalizado [e/10, de/8, (sp-21)/9]."""
    return torch.tensor(
        [error / NORM_ERROR,
         derivada / NORM_DERIVADA,
         (setpoint - NORM_SP_CENTRO) / NORM_SP_ESCALA],
        dtype=torch.float32)


# =============================================================================
#                              RED Q PROFUNDA
# =============================================================================

class QNet(nn.Module):
    """MLP 3 -> 16 -> 16 -> 5 con ReLU: Q(s, ·) para las 5 acciones."""

    def __init__(self):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Linear(3, NEURONAS_OCULTAS), nn.ReLU(),
            nn.Linear(NEURONAS_OCULTAS, NEURONAS_OCULTAS), nn.ReLU(),
            nn.Linear(NEURONAS_OCULTAS, N_ACCIONES),
        )

    def forward(self, x):
        return self.capas(x)


# =============================================================================
#                               ENTRENAMIENTO
# =============================================================================

def entrenar(episodios, semilla):
    rng = random.Random(semilla)
    torch.manual_seed(semilla)

    env = LevitadorEnv(rng)
    online = QNet()
    target = QNet()
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizador = torch.optim.Adam(online.parameters(), lr=LR)
    perdida_fn = nn.SmoothL1Loss()                # Huber

    # Replay como ring buffer pre-dimensionado (muestreo O(1)).
    replay = []
    idx_replay = 0

    def guardar(transicion):
        nonlocal idx_replay
        if len(replay) < REPLAY_CAPACIDAD:
            replay.append(transicion)
        else:
            replay[idx_replay] = transicion
            idx_replay = (idx_replay + 1) % REPLAY_CAPACIDAD

    def paso_gradiente():
        lote = [replay[rng.randrange(len(replay))] for _ in range(BATCH)]
        s_b = torch.stack([t[0] for t in lote])
        a_b = torch.tensor([t[1] for t in lote], dtype=torch.long)
        r_b = torch.tensor([t[2] for t in lote], dtype=torch.float32)
        s2_b = torch.stack([t[3] for t in lote])
        f_b = torch.tensor([float(t[4]) for t in lote])
        q_sa = online(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: argmax con la red online, valor con la target.
            a_star = online(s2_b).argmax(dim=1, keepdim=True)
            q_eval = target(s2_b).gather(1, a_star).squeeze(1)
            y = r_b + GAMMA * (1.0 - f_b) * q_eval
        perdida = perdida_fn(q_sa, y)
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

    dec_eps = (EPSILON_FIN / EPSILON_INI) ** (1.0 / episodios)
    dec_tutor = (TUTOR_FIN / TUTOR_INI) ** (1.0 / episodios)
    epsilon = EPSILON_INI
    p_tutor = TUTOR_INI
    pasos_globales = 0
    actualizaciones = 0

    t0 = time.perf_counter()
    for _ep in range(episodios):                  # Silencioso
        # Currículum: la mitad de los episodios atacan la zona de fallas
        # (objetivos altos, donde el sobreimpulso choca contra el techo).
        if rng.random() < CURRICULUM_PROB:
            e, de = env.reiniciar(
                rng.uniform(CURRICULUM_SP_MIN, SETPOINT_MAX_CM))
        else:
            e, de = env.reiniciar()
        sp = env.setpoint
        e_prev = e

        for _ in range(PASOS_MAX_EPISODIO):
            estado = a_tensor_estado(e, de, sp)
            # ---- Política de comportamiento: epsilon / tutor / greedy ----
            u = rng.random()
            if u < epsilon:
                a = rng.randrange(N_ACCIONES)
            elif u < epsilon + p_tutor:
                a = tutor_experto(e, de)
            else:
                with torch.no_grad():
                    a = int(online(estado).argmax().item())

            e2, de2, r_base, fin = env.paso(ACCIONES_DQN[a])
            r = recompensa_total(r_base, e2, de2, e_prev, fin)
            guardar((estado, a, r, a_tensor_estado(e2, de2, sp), fin))
            e, de, e_prev = e2, de2, e2
            pasos_globales += 1

            if (len(replay) >= WARMUP_TRANSICIONES
                    and pasos_globales % APRENDER_CADA == 0):
                paso_gradiente()
                actualizaciones += 1
                if actualizaciones % TARGET_UPDATE_CADA == 0:
                    target.load_state_dict(online.state_dict())

            if fin:
                break
        epsilon *= dec_eps
        p_tutor *= dec_tutor

    t_total = time.perf_counter() - t0
    return online, t_total, pasos_globales, actualizaciones


# =============================================================================
#                       EVALUACIÓN GREEDY MULTI-OBJETIVO
# =============================================================================

def evaluar(modelo, episodios_eval, semilla, sp_min=SETPOINT_MIN_CM,
            sp_max=SETPOINT_MAX_CM):
    rng = random.Random(semilla + 5000)
    env = LevitadorEnv(rng)
    exitos = 0
    en_banda = 0
    pasos = 0
    suma_e2 = 0.0
    modelo.eval()
    with torch.no_grad():
        for _ in range(episodios_eval):
            e, de = env.reiniciar(rng.uniform(sp_min, sp_max))
            sp = env.setpoint
            fin = False
            for _ in range(PASOS_MAX_EPISODIO):
                a = int(modelo(a_tensor_estado(e, de, sp)).argmax().item())
                e, de, _r, fin = env.paso(ACCIONES_DQN[a])
                pasos += 1
                suma_e2 += e * e
                if abs(e) <= ZONA_MUERTA_CM:
                    en_banda += 1
                if fin:
                    break
            if not fin:
                exitos += 1
    return (100.0 * exitos / episodios_eval,
            100.0 * en_banda / pasos,
            math.sqrt(suma_e2 / pasos))


# =============================================================================
#                  TRANSPILACIÓN A MICROPYTHON (sin torch/numpy)
# =============================================================================

def _fmt(v):
    return "{:.8g}".format(float(v))


def _fmt_matriz(nombre, tensor, lineas):
    lineas.append("{} = (".format(nombre))
    for fila in tensor.tolist():
        lineas.append("    ({}),".format(", ".join(_fmt(v) for v in fila)))
    lineas.append(")")


def generar_micropython(modelo):
    sd = modelo.state_dict()
    W1, B1 = sd["capas.0.weight"], sd["capas.0.bias"]    # (16,3), (16,)
    W2, B2 = sd["capas.2.weight"], sd["capas.2.bias"]    # (16,16), (16,)
    W3, B3 = sd["capas.4.weight"], sd["capas.4.bias"]    # (5,16), (5,)
    H = NEURONAS_OCULTAS

    ls = []
    a = ls.append
    a("# ===== INICIO BLOQUE DQN (generado por entrenar_dqn.py) ==============")
    a("# DQN Goal-Conditioned 3-{}-{}-5 (ReLU). Entradas: error (cm),".format(
        H, H))
    a("# derivada (cm/s), setpoint (cm). Salida: Delta PWM del catalogo")
    a("# asimetrico ACCIONES_DQN = argmax_a Q(s, a). Sin numpy/torch.")
    a("")
    a("ACCIONES_DQN = {}".format(ACCIONES_DQN))
    a("")
    _fmt_matriz("_DQN_W1", W1, ls)
    a("_DQN_B1 = ({})".format(", ".join(_fmt(v) for v in B1.tolist())))
    _fmt_matriz("_DQN_W2", W2, ls)
    a("_DQN_B2 = ({})".format(", ".join(_fmt(v) for v in B2.tolist())))
    _fmt_matriz("_DQN_W3", W3, ls)
    a("_DQN_B3 = ({})".format(", ".join(_fmt(v) for v in B3.tolist())))
    a("")
    a("# Buffers pre-asignados: cero allocations por inferencia (RAM ESP32)")
    a("_DQN_H1 = [0.0] * {}".format(H))
    a("_DQN_H2 = [0.0] * {}".format(H))
    a("")
    a("")
    a("def calcular_pwm_dqn(error, derivada, setpoint):")
    a('    """Inferencia DQN en MicroPython puro: ~{} MACs (~5 ms a 240 MHz).'
      .format(3 * H + H * H + H * N_ACCIONES))
    a('    Retorna Delta PWM (int) del catalogo ACCIONES_DQN."""')
    a("    # Normalizacion (constantes horneadas del entrenamiento)")
    a("    x0 = error / {}".format(_fmt(NORM_ERROR)))
    a("    x1 = derivada / {}".format(_fmt(NORM_DERIVADA)))
    a("    x2 = (setpoint - {}) / {}".format(
        _fmt(NORM_SP_CENTRO), _fmt(NORM_SP_ESCALA)))
    a("    # Capa 1: Linear(3->{}) + ReLU".format(H))
    a("    for i in range({}):".format(H))
    a("        w = _DQN_W1[i]")
    a("        v = w[0] * x0 + w[1] * x1 + w[2] * x2 + _DQN_B1[i]")
    a("        _DQN_H1[i] = v if v > 0.0 else 0.0")
    a("    # Capa 2: Linear({}->{}) + ReLU".format(H, H))
    a("    for i in range({}):".format(H))
    a("        w = _DQN_W2[i]")
    a("        v = _DQN_B2[i]")
    a("        for j in range({}):".format(H))
    a("            v += w[j] * _DQN_H1[j]")
    a("        _DQN_H2[i] = v if v > 0.0 else 0.0")
    a("    # Capa 3: Linear({}->5) + argmax".format(H))
    a("    mejor_a = 0")
    a("    mejor_q = -1e30")
    a("    for i in range(5):")
    a("        w = _DQN_W3[i]")
    a("        v = _DQN_B3[i]")
    a("        for j in range({}):".format(H))
    a("            v += w[j] * _DQN_H2[j]")
    a("        if v > mejor_q:")
    a("            mejor_q = v")
    a("            mejor_a = i")
    a("    return ACCIONES_DQN[mejor_a]")
    a("# ===== FIN BLOQUE DQN =================================================")
    return "\n".join(ls)


def verificar_transpilacion(modelo, codigo, rng, n_muestras=5000):
    """La función MicroPython debe elegir la MISMA acción que torch.argmax."""
    ns = {}
    exec(codigo, ns)
    fn = ns["calcular_pwm_dqn"]
    coincidencias = 0
    modelo.eval()
    with torch.no_grad():
        for _ in range(n_muestras):
            e = rng.uniform(-18.0, 18.0)
            de = rng.uniform(-12.0, 12.0)
            sp = rng.uniform(SETPOINT_MIN_CM, SETPOINT_MAX_CM)
            a_torch = int(modelo(a_tensor_estado(e, de, sp)).argmax().item())
            if fn(e, de, sp) == ACCIONES_DQN[a_torch]:
                coincidencias += 1
    return 100.0 * coincidencias / n_muestras


# =============================================================================
#                                    MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DQN Goal-Conditioned + transpilación a MicroPython")
    parser.add_argument("--episodios", type=int, default=EPISODIOS_DEFECTO)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    print("[DQN] Entrenando {} corridas x {} episodios (Goal-Conditioned, "
          "objetivos {}-{} cm,\n[DQN] tutor 7x7 mezclado {}->{}, currículum "
          "de objetivos altos y selección de modelo)...".format(
              N_CORRIDAS, args.episodios, SETPOINT_MIN_CM, SETPOINT_MAX_CM,
              TUTOR_INI, TUTOR_FIN))

    candidatas = []
    t_acumulado = 0.0
    for i in range(N_CORRIDAS):
        semilla_i = args.semilla + i
        modelo_i, t_i, pasos, updates = entrenar(args.episodios, semilla_i)
        t_acumulado += t_i
        ex_c, bd_c, rms_c = evaluar(modelo_i, EPISODIOS_EVAL_SELECCION,
                                    semilla_i)
        ex_e, bd_e, rms_e = evaluar(modelo_i, EPISODIOS_EVAL_SELECCION,
                                    semilla_i + 1, ENVOLVENTE_MIN_CM,
                                    ENVOLVENTE_MAX_CM)
        puntaje = 2.0 * (ex_c + ex_e) + bd_c + bd_e - 5.0 * (rms_c + rms_e)
        candidatas.append((puntaje, i, modelo_i))
        print("[DQN]   corrida {}/{} (sem {}): completo {:.0f}%/{:.0f}%/"
              "{:.2f} | envolvente {:.0f}%/{:.0f}%/{:.2f} | {:.0f} s, "
              "{} updates".format(i + 1, N_CORRIDAS, semilla_i, ex_c, bd_c,
                                  rms_c, ex_e, bd_e, rms_e, t_i, updates))
    print("[DQN] Tiempo total de entrenamiento: {:.1f} s".format(t_acumulado))

    modelo = max(candidatas)[2]
    print("[DQN] Corrida ganadora: {} (selección por evaluación dual)".format(
        max(candidatas)[1] + 1))

    ex1, bd1, rms1 = evaluar(modelo, EPISODIOS_EVAL, args.semilla + 7777)
    print("[EVALUACIÓN] Rango completo {}-{} cm ({} episodios greedy):"
          .format(SETPOINT_MIN_CM, SETPOINT_MAX_CM, EPISODIOS_EVAL))
    print("[EVALUACIÓN]   sin tocar límites: {:.1f}% | en ±1 cm: {:.1f}% | "
          "RMS: {:.2f} cm".format(ex1, bd1, rms1))
    ex2, bd2, rms2 = evaluar(modelo, EPISODIOS_EVAL, args.semilla + 7778,
                             ENVOLVENTE_MIN_CM, ENVOLVENTE_MAX_CM)
    print("[EVALUACIÓN] Envolvente operativa {}-{} cm:".format(
        ENVOLVENTE_MIN_CM, ENVOLVENTE_MAX_CM))
    print("[EVALUACIÓN]   sin tocar límites: {:.1f}% | en ±1 cm: {:.1f}% | "
          "RMS: {:.2f} cm".format(ex2, bd2, rms2))

    # ---- Pesos del modelo (requisito de la rúbrica) ----
    print("\n[PESOS] state_dict del modelo entrenado:")
    for nombre, tensor in modelo.state_dict().items():
        print("  {} {} = {}".format(
            nombre, tuple(tensor.shape),
            [round(v, 6) for v in tensor.flatten().tolist()]))

    # ---- Transpilación + verificación ----
    codigo = generar_micropython(modelo)
    rng = random.Random(args.semilla + 99)
    acuerdo = verificar_transpilacion(modelo, codigo, rng)
    print("\n[VERIFICACIÓN] Acuerdo de acción MicroPython vs PyTorch sobre "
          "5000 estados: {:.2f}%".format(acuerdo))
    assert acuerdo >= 99.9, "la transpilación no es equivalente"

    with open(ARCHIVO_SALIDA, "w") as f:
        f.write(codigo + "\n")
    print("[SALIDA] Bloque guardado en '{}'. Pégalo en esp32/main.py "
          "reemplazando el stub\n[SALIDA] entre 'INICIO BLOQUE DQN' y "
          "'FIN BLOQUE DQN':\n".format(ARCHIVO_SALIDA))
    print("=" * 79)
    print(codigo)
    print("=" * 79)


if __name__ == "__main__":
    main()