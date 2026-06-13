# =============================================================================
# etapa4_qlearning.py — Levitador Neumático | Etapa 4 (PC)
# -----------------------------------------------------------------------------
# Q-LEARNING TABULAR EN SANDBOX SIMULADO + TRANSPILACIÓN A MICROPYTHON O(1).
#
# El entrenamiento NO toca el hardware: la exploración epsilon-greedy inicial
# (acciones aleatorias) castigaría el MOSFET/turbina con conmutaciones
# violentas. Se entrena contra una planta simulada y solo la POLÍTICA GREEDY
# final (argmax de la Q-Table por estado: 25 enteros) viaja al ESP32.
#
# FIDELIDAD SIM-TO-REAL — el estado del agente replica EXACTAMENTE lo que
# computa el firmware (esp32/main.py v6):
#   - Derivada por ventana de 3 muestras + EMA alfa 0.8 (idéntica).
#   - Cuantización del sensor (~0.3 cm) y retardo de 2 muestras del filtro.
#   - Retardo de transporte de la actuación + inercia de la turbina (la
#     planta real tiene ~400 ms de fase total).
#   - Desbalance aleatorio del equilibrio por episodio (±300 cuentas): la
#     política aprende a tolerar una calibración imperfecta de PWM_BASE.
#
# TÉCNICAS DE CONVERGENCIA (validadas empíricamente; con Bellman "a secas"
# los valores Q de acciones vecinas quedan casi empatados por el retardo de
# fase y el argmax resulta incoherente en estados poco visitados):
#   - Inicialización informada: prior heurístico anti-diagonal PEQUEÑO
#     (máx 1.0 vs recompensa de 10.0) que rompe empates hacia lo físicamente
#     sensato; los datos lo sobreescriben sin esfuerzo.
#   - Shaping basado en potencial Phi(s) = -0.6*|error|: densifica el
#     gradiente de recompensa SIN alterar la política óptima (Ng et al.).
#   - Penalización de velocidad cerca del setpoint: fomenta llegada suave.
#   - Tasa de aprendizaje decreciente por par (estado, acción).
#   - Selección de modelo: 4 corridas independientes; se elige la mejor
#     evaluación entre las que pasan la sanidad física de esquinas.
#
# Espacios (hiper-reducidos):
#   ESTADOS  : 5 bins de error x 5 bins de derivada = 25 estados.
#   ACCIONES : 5 deltas de PWM = (-1500, -500, 0, 500, 1500).
#   Q-TABLE  : 25 x 5 = 125 valores -> política aplanada de 25 enteros.
#
# Recompensas:
#   +10  si |error| <= 1.0 cm (zona muerta del setpoint 21.0 cm)
#   -100 y fin de episodio al tocar los límites duros (5.0 / 35.0 cm)
#   shaping potencial + llegada suave en el resto
#
# Uso:
#   python etapa4_qlearning.py                       (4 x 5000 episodios)
#   python etapa4_qlearning.py --episodios 10000     (4 x 10000)
#
# Dependencias: ninguna (Python estándar).
# =============================================================================

import argparse
import random
import sys
import time

# ------------------------- Planta / lazo (como el firmware) ------------------
SETPOINT_CM = 21.0
TS_S = 0.050                          # 50 ms (20 Hz)
LIMITE_INFERIOR_CM = 5.0              # Castigo severo al tocarlo
LIMITE_SUPERIOR_CM = 35.0
N_DERIVADA = 3                        # Ventana de derivada (idéntica al FW)
ALFA_DERIVADA = 0.8                   # EMA de la derivada (idéntica al FW)

# ----------------------------- Física simulada -------------------------------
K_PLANTA = 0.012                      # cm/s^2 por cuenta de PWM
C_ARRASTRE = 1.2                      # 1/s (fricción del aire)
TAU_VENTILADOR_S = 0.15               # Inercia de primer orden de la turbina
RETARDO_ACTUACION = 3                 # Muestras de retardo de transporte
RETARDO_SENSOR = 2                    # Muestras de retardo del filtro
CUANTIZACION_CM = 0.3                 # Escalón efectivo del HC-SR04 filtrado
RUIDO_TURBULENCIA = 2.5               # Desv. estándar (cm/s^2) de las ráfagas
DESBALANCE_EQ_MAX = 300               # Error simulado de calibración de
                                      # PWM_BASE por episodio (cuentas)

# --------------------------- Espacios discretos ------------------------------
# 5 bins de error (cm), por cadena de comparaciones O(1):
#   0: e < -6 | 1: [-6,-1.5) | 2: [-1.5,1.5] | 3: (1.5,6] | 4: e > 6
UMBRAL_E_1, UMBRAL_E_2 = -6.0, -1.5
UMBRAL_E_3, UMBRAL_E_4 = 1.5, 6.0
# 5 bins de derivada (cm/s):
UMBRAL_DE_1, UMBRAL_DE_2 = -5.0, -1.5
UMBRAL_DE_3, UMBRAL_DE_4 = 1.5, 5.0

N_BINS_ERROR = 5
N_BINS_DERIVADA = 5
N_ESTADOS = N_BINS_ERROR * N_BINS_DERIVADA          # 25
ACCIONES = (-1500, -500, 0, 500, 1500)
N_ACCIONES = len(ACCIONES)

# ----------------------- Hiperparámetros de Q-Learning -----------------------
EPISODIOS_POR_CORRIDA = 5000          # Mínimo exigido por corrida
N_CORRIDAS = 4                        # Selección de modelo entre corridas
PASOS_MAX_EPISODIO = 200              # 200 * 50 ms = 10 s simulados
ALPHA_INICIAL = 0.5                   # Decrece por visitas de (s, a)
ALPHA_MINIMO = 0.05
ALPHA_DECAIMIENTO_K = 0.004
GAMMA = 0.95
EPSILON_INICIAL = 1.0
EPSILON_FINAL = 0.05
RECOMPENSA_BANDA = 10.0
ZONA_MUERTA_CM = 1.0
CASTIGO_LIMITE = -100.0
PESO_POTENCIAL = 0.6                  # Phi(s) = -PESO_POTENCIAL * |error|
PESO_LLEGADA_SUAVE = 0.10             # Penaliza |derivada| con |error| <= 3

EPISODIOS_EVALUACION = 100            # Por corrida (selección de modelo)
EPISODIOS_EVAL_FINAL = 300            # De la política elegida

ARCHIVO_SALIDA = "qlearning_inference_esp32.py"


def discretizar_error(e):
    """Bin del error en O(1). Idéntico a la función transpilada."""
    if e < UMBRAL_E_1:
        return 0
    if e < UMBRAL_E_2:
        return 1
    if e <= UMBRAL_E_3:
        return 2
    if e <= UMBRAL_E_4:
        return 3
    return 4


def discretizar_derivada(de):
    """Bin de la derivada en O(1). Idéntico a la función transpilada."""
    if de < UMBRAL_DE_1:
        return 0
    if de < UMBRAL_DE_2:
        return 1
    if de <= UMBRAL_DE_3:
        return 2
    if de <= UMBRAL_DE_4:
        return 3
    return 4


# =============================================================================
#                        ENTORNO FÍSICO SIMULADO (SANDBOX)
# =============================================================================

class SimuladorLevitador:
    """Planta simulada del tubo, vista A TRAVÉS de la misma cadena de
    medición y estimación del firmware: el agente nunca ve el estado ideal,
    ve lo que vería el ESP32."""

    def __init__(self, rng):
        self._rng = rng
        self.reiniciar()

    def reiniciar(self):
        rng = self._rng
        # Estado físico verdadero (oculto para el agente).
        self._z = rng.uniform(9.0, 31.0)           # posición real (cm)
        self._v = rng.uniform(-2.0, 2.0)           # velocidad real (cm/s)
        # Desbalance de calibración del PWM_BASE simulado por episodio.
        self._desbalance = rng.uniform(-DESBALANCE_EQ_MAX, DESBALANCE_EQ_MAX)
        self._u_ventilador = 0.0                   # delta efectivo en el fan
        # Colas de retardo (transporte de actuación y filtro del sensor).
        self._cola_actuacion = [0.0] * RETARDO_ACTUACION
        self._cola_sensor = [self._z] * RETARDO_SENSOR
        # Estimador de derivada idéntico al firmware.
        self._hist_error = [0.0] * (N_DERIVADA + 1)
        self._i_hist = 0
        self._muestras = 0
        self._de_filtrada = 0.0
        return self._observar()

    def _observar(self):
        """Sensor cuantizado y retardado -> (error, derivada) como en el FW."""
        self._cola_sensor.append(self._z)
        z_retrasada = self._cola_sensor.pop(0)
        z_medida = round(z_retrasada / CUANTIZACION_CM) * CUANTIZACION_CM

        error = z_medida - SETPOINT_CM
        e_antiguo = self._hist_error[self._i_hist]
        self._hist_error[self._i_hist] = error
        self._i_hist += 1
        if self._i_hist > N_DERIVADA:
            self._i_hist = 0
        self._muestras += 1
        if self._muestras > N_DERIVADA:
            de_cruda = (error - e_antiguo) / (N_DERIVADA * TS_S)
        else:
            de_cruda = 0.0
        self._de_filtrada = (ALFA_DERIVADA * de_cruda
                             + (1.0 - ALFA_DERIVADA) * self._de_filtrada)
        return error, self._de_filtrada

    def paso(self, delta_pwm):
        """Aplica una acción. Retorna (error, derivada, recompensa, fin)."""
        # Retardo de transporte de la actuación (aplicación + lazo).
        self._cola_actuacion.append(float(delta_pwm))
        delta_aplicado = self._cola_actuacion.pop(0)

        # Inercia de la turbina (primer orden) sobre el delta efectivo.
        objetivo = delta_aplicado + self._desbalance
        self._u_ventilador += (TS_S / TAU_VENTILADOR_S) * (
            objetivo - self._u_ventilador)

        # Dinámica: aceleración proporcional al PWM - amortiguamiento + ruido.
        acel = (K_PLANTA * self._u_ventilador
                - C_ARRASTRE * self._v
                + self._rng.gauss(0.0, RUIDO_TURBULENCIA))
        self._v += acel * TS_S
        self._z += self._v * TS_S

        # Límites duros del tubo: castigo severo y fin de episodio.
        if self._z <= LIMITE_INFERIOR_CM or self._z >= LIMITE_SUPERIOR_CM:
            self._z = max(LIMITE_INFERIOR_CM,
                          min(LIMITE_SUPERIOR_CM, self._z))
            error, derivada = self._observar()
            return error, derivada, CASTIGO_LIMITE, True

        error, derivada = self._observar()
        if abs(error) <= ZONA_MUERTA_CM:
            recompensa = RECOMPENSA_BANDA
        else:
            recompensa = 0.0
        return error, derivada, recompensa, False


# =============================================================================
#                       ENTRENAMIENTO Q-LEARNING (BELLMAN)
# =============================================================================

def _q_inicial_heuristica():
    """Prior anti-diagonal PEQUEÑO (máx 1.0 frente a recompensas de ±10/100):
    rompe los empates de estados poco informados hacia la física correcta
    (abajo->empujar, arriba->recortar) sin sesgar el aprendizaje."""
    Q = [[0.0] * N_ACCIONES for _ in range(N_ESTADOS)]
    for ie in range(N_BINS_ERROR):
        for ide in range(N_BINS_DERIVADA):
            tendencia = (ie - 2) + (ide - 2)        # >0: alto/subiendo
            accion_h = 2 - max(-2, min(2, tendencia))
            for a in range(N_ACCIONES):
                Q[ie * N_BINS_DERIVADA + ide][a] = \
                    1.0 - 0.4 * abs(a - accion_h)
    return Q


def entrenar_corrida(episodios, semilla):
    """Una corrida completa de Q-Learning. Silenciosa. Retorna la Q-Table."""
    rng = random.Random(semilla)
    entorno = SimuladorLevitador(rng)
    Q = _q_inicial_heuristica()
    visitas = [[0] * N_ACCIONES for _ in range(N_ESTADOS)]

    decaimiento_eps = (EPSILON_FINAL / EPSILON_INICIAL) ** (1.0 / episodios)
    epsilon = EPSILON_INICIAL

    for _ep in range(episodios):
        error, derivada = entorno.reiniciar()
        estado = discretizar_error(error) * N_BINS_DERIVADA \
            + discretizar_derivada(derivada)
        error_previo = error

        for _paso in range(PASOS_MAX_EPISODIO):
            # ---- Política epsilon-greedy ----
            if rng.random() < epsilon:
                accion = rng.randrange(N_ACCIONES)
            else:
                fila = Q[estado]
                accion = 0
                mejor = fila[0]
                for a in range(1, N_ACCIONES):
                    if fila[a] > mejor:
                        mejor = fila[a]
                        accion = a

            error, derivada, recompensa, fin = entorno.paso(ACCIONES[accion])

            # ---- Shaping basado en potencial (no altera la política óptima)
            if not fin:
                recompensa += (GAMMA * (-PESO_POTENCIAL * abs(error))
                               - (-PESO_POTENCIAL * abs(error_previo)))
            # ---- Llegada suave: penaliza velocidad cerca del setpoint ----
            if abs(error) <= 3.0:
                recompensa -= PESO_LLEGADA_SUAVE * abs(derivada)

            estado_sig = discretizar_error(error) * N_BINS_DERIVADA \
                + discretizar_derivada(derivada)

            # ---- Ecuación de Bellman con alfa decreciente por (s, a) ----
            #   Q(s,a) <- Q(s,a) + alpha*(r + gamma*max_a' Q(s',a') - Q(s,a))
            visitas[estado][accion] += 1
            alfa = ALPHA_INICIAL / (1.0
                                    + ALPHA_DECAIMIENTO_K
                                    * visitas[estado][accion])
            if alfa < ALPHA_MINIMO:
                alfa = ALPHA_MINIMO
            fila_sig = Q[estado_sig]
            max_sig = fila_sig[0]
            for a in range(1, N_ACCIONES):
                if fila_sig[a] > max_sig:
                    max_sig = fila_sig[a]
            objetivo_td = recompensa + (0.0 if fin else GAMMA * max_sig)
            Q[estado][accion] += alfa * (objetivo_td - Q[estado][accion])

            estado = estado_sig
            error_previo = error
            if fin:
                break

        epsilon *= decaimiento_eps
    return Q


def extraer_politica(Q):
    """Política greedy: argmax por estado -> 25 enteros (deltas de PWM)."""
    politica = []
    for s in range(N_ESTADOS):
        fila = Q[s]
        mejor_a = 0
        mejor_q = fila[0]
        for a in range(1, N_ACCIONES):
            if fila[a] > mejor_q:
                mejor_q = fila[a]
                mejor_a = a
        politica.append(ACCIONES[mejor_a])
    return politica


def politica_es_sana(politica):
    """Sanidad física en las DOS esquinas inequívocas del espacio de estados
    (las demás combinaciones admiten control anticipatorio legítimo):
      - muy abajo Y cayendo rápido  -> empuje máximo (+1500)
      - muy arriba Y subiendo rápido-> recorte máximo (-1500)"""
    return politica[0] == 1500 and politica[N_ESTADOS - 1] == -1500


# =============================================================================
#                  EVALUACIÓN DE LA POLÍTICA GREEDY (sin explorar)
# =============================================================================

def evaluar(politica, episodios_eval, semilla):
    rng = random.Random(semilla + 1000)
    entorno = SimuladorLevitador(rng)
    exitosos = 0
    pasos_en_banda = 0
    pasos_totales = 0
    suma_e2 = 0.0

    for _ in range(episodios_eval):
        error, derivada = entorno.reiniciar()
        fin = False
        for _paso in range(PASOS_MAX_EPISODIO):
            idx = discretizar_error(error) * N_BINS_DERIVADA \
                + discretizar_derivada(derivada)
            error, derivada, _r, fin = entorno.paso(politica[idx])
            pasos_totales += 1
            suma_e2 += error * error
            if abs(error) <= ZONA_MUERTA_CM:
                pasos_en_banda += 1
            if fin:
                break
        if not fin:
            exitosos += 1

    rms = (suma_e2 / pasos_totales) ** 0.5
    return (100.0 * exitosos / episodios_eval,
            100.0 * pasos_en_banda / pasos_totales,
            rms)


# =============================================================================
#                       TRANSPILACIÓN A MICROPYTHON O(1)
# =============================================================================

def generar_micropython(politica):
    lineas = []
    a = lineas.append
    a("# ===== INICIO BLOQUE QLEARNING (generado por etapa4_qlearning.py) =====")
    a("# Politica greedy de la Q-Table (25 estados = 5 bins error x 5 bins")
    a("# derivada). Inferencia por indexacion directa: complejidad O(1).")
    a("# Entradas: error (cm), derivada (cm/s). Salida: delta_pwm (int).")
    a("")
    a("# Q-Table aplanada (argmax por estado, en cuentas de Delta PWM):")
    a("_QL_POLITICA = (")
    for f in range(N_BINS_ERROR):
        fila = politica[f * N_BINS_DERIVADA:(f + 1) * N_BINS_DERIVADA]
        a("    {},  # error bin {}".format(
            ", ".join("{:6d}".format(v) for v in fila), f))
    a(")")
    a("")
    a("")
    a("def calcular_pwm_qlearning(error, derivada):")
    a('    """Inferencia Q-Learning O(1): discretiza y consulta la politica.')
    a('    Retorna Delta PWM (int) dentro de [{}, {}]."""'.format(
        min(ACCIONES), max(ACCIONES)))
    a("    # Bin del error (umbrales en cm)")
    a("    if error < {}:".format(UMBRAL_E_1))
    a("        ie = 0")
    a("    elif error < {}:".format(UMBRAL_E_2))
    a("        ie = 1")
    a("    elif error <= {}:".format(UMBRAL_E_3))
    a("        ie = 2")
    a("    elif error <= {}:".format(UMBRAL_E_4))
    a("        ie = 3")
    a("    else:")
    a("        ie = 4")
    a("    # Bin de la derivada (umbrales en cm/s)")
    a("    if derivada < {}:".format(UMBRAL_DE_1))
    a("        ide = 0")
    a("    elif derivada < {}:".format(UMBRAL_DE_2))
    a("        ide = 1")
    a("    elif derivada <= {}:".format(UMBRAL_DE_3))
    a("        ide = 2")
    a("    elif derivada <= {}:".format(UMBRAL_DE_4))
    a("        ide = 3")
    a("    else:")
    a("        ide = 4")
    a("    return _QL_POLITICA[ie * 5 + ide]")
    a("# ====== FIN BLOQUE QLEARNING ======")
    return "\n".join(lineas)


# =============================================================================
#                                    MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Etapa 4: Q-Learning tabular en sandbox + transpilación")
    parser.add_argument("--episodios", type=int,
                        default=EPISODIOS_POR_CORRIDA,
                        help="Episodios POR CORRIDA (mínimo 5000; "
                             "total = 4 corridas x episodios)")
    parser.add_argument("--semilla", type=int, default=42,
                        help="Semilla base de reproducibilidad")
    args = parser.parse_args()

    if args.episodios < 5000:
        print("ERROR: se exigen al menos 5000 episodios por corrida.")
        sys.exit(1)

    total = N_CORRIDAS * args.episodios
    print("[Q-LEARNING] Entrenando {} corridas x {} episodios = {} episodios "
          "totales (silencioso)...".format(N_CORRIDAS, args.episodios, total))

    t_inicio = time.perf_counter()
    candidatas = []
    for i in range(N_CORRIDAS):
        semilla_i = args.semilla + i
        Q = entrenar_corrida(args.episodios, semilla_i)
        politica = extraer_politica(Q)
        exito, banda, rms = evaluar(politica, EPISODIOS_EVALUACION, semilla_i)
        sana = politica_es_sana(politica)
        puntaje = exito * 2.0 + banda - rms * 10.0
        candidatas.append((sana, puntaje, exito, banda, rms,
                           politica, semilla_i))
        print("[Q-LEARNING]   corrida {}/{}: sana={} exito={:.0f}% "
              "banda={:.0f}% rms={:.2f}".format(
                  i + 1, N_CORRIDAS, "SI" if sana else "NO",
                  exito, banda, rms))
    t_total = time.perf_counter() - t_inicio
    print("[Q-LEARNING] Tiempo total de entrenamiento: {:.2f} s "
          "({:.0f} episodios/s)".format(t_total, total / t_total))

    # ---- Selección de modelo: mejor puntaje ENTRE las sanas ----
    sanas = [c for c in candidatas if c[0]]
    if sanas:
        elegida = max(sanas, key=lambda c: c[1])
    else:
        # Plan B documentado: ninguna corrida pasó la sanidad. Se toma la de
        # mejor puntaje y se reparan SOLO las dos esquinas inequívocas (cuyo
        # valor óptimo es demostrable por física, no por aprendizaje).
        elegida = max(candidatas, key=lambda c: c[1])
        print("[Q-LEARNING] AVISO: ninguna corrida pasó la sanidad; se "
              "reparan las 2 esquinas inequívocas de la mejor política.")
        elegida[5][0] = 1500
        elegida[5][N_ESTADOS - 1] = -1500
    politica = elegida[5]
    print("[Q-LEARNING] Política elegida: semilla {} (sana={}).".format(
        elegida[6], "SI" if elegida[0] else "REPARADA"))

    assert politica_es_sana(politica)

    # ---- Evaluación final extendida con semilla virgen ----
    exito, banda, rms = evaluar(politica, EPISODIOS_EVAL_FINAL,
                                args.semilla + 7777)
    print("[EVALUACIÓN FINAL] {} episodios greedy (10 s c/u, turbulencia y "
          "desbalance ±{} cuentas):".format(
              EPISODIOS_EVAL_FINAL, DESBALANCE_EQ_MAX))
    print("[EVALUACIÓN FINAL]   Episodios sin tocar límites: {:.1f}%".format(
        exito))
    print("[EVALUACIÓN FINAL]   Tiempo dentro de ±{:.0f} cm:    {:.1f}%"
          .format(ZONA_MUERTA_CM, banda))
    print("[EVALUACIÓN FINAL]   Error RMS:                   {:.2f} cm"
          .format(rms))

    # ---- Transpilación + verificación de equivalencia ----
    codigo = generar_micropython(politica)
    entorno_exec = {}
    exec(codigo, entorno_exec)
    fn = entorno_exec["calcular_pwm_qlearning"]
    for e, de in ((-10, -6), (-10, 6), (-3, 0), (0, 0), (0, 2),
                  (3, 0), (10, -6), (10, 6)):
        ie = discretizar_error(e)
        ide = discretizar_derivada(de)
        assert fn(e, de) == politica[ie * 5 + ide]
    print("[VERIFICACIÓN] Función transpilada equivalente a la política: OK")

    with open(ARCHIVO_SALIDA, "w") as f:
        f.write(codigo + "\n")
    print("\n[SALIDA] Bloque guardado en '{}'.".format(ARCHIVO_SALIDA))
    print("[SALIDA] Pega TODO el bloque siguiente en esp32/main.py, "
          "reemplazando el stub\n[SALIDA] marcado entre 'INICIO BLOQUE "
          "QLEARNING' y 'FIN BLOQUE QLEARNING':\n")
    print("=" * 79)
    print(codigo)
    print("=" * 79)


if __name__ == "__main__":
    main()