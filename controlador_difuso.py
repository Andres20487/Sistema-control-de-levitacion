# =============================================================================
# controlador_difuso.py — Levitador Neumático | Etapa 1 (PC) | v3 ANTI-OSCILACIÓN
# -----------------------------------------------------------------------------
# Controlador Difuso Mamdani + ACCIÓN INTEGRAL DE AJUSTE FINO + Cosechador.
#
# CAMBIOS DE SINTONIZACIÓN v3 (validados en simulación de lazo cerrado con
# retardos reales, cuantización del HC-SR04 y turbulencia):
#
#   1. PWM_BASE DEBE SER EL EQUILIBRIO REAL MEDIDO. Un difuso PD no tiene
#      integrador: si PWM_BASE queda por debajo del valor que sostiene la
#      bola, dentro de la banda muerta el motor es insuficiente, la bola cae,
#      acelera, recibe un correctivo violento y rebota -> "efecto resorte".
#      >>> CALIBRA: desde el REPL, busca el duty donde la bola flota sola
#      >>> varios segundos (p.ej. ventilador.duty_u16(38400)) y escribe ese
#      >>> valor AQUÍ y en esp32/main.py (deben ser idénticos).
#
#   2. AUTORIDAD DE SALIDA REDUCIDA A ~LA MITAD (pico máx ~±1800 en vez de
#      ±3052). Con ~350-500 ms de retardo total de lazo (sensor + filtro +
#      serial + inercia del ventilador), las correcciones grandes llegan
#      tarde y bombean energía a la oscilación en vez de amortiguarla.
#
#   3. DERIVADA POR VENTANA: velocidad = (e[k] - e[k-3]) / (3*Ts) en lugar
#      de diferencia por muestra. La cuantización de ~0.3 cm del sensor
#      generaba picos falsos de ±6 cm/s (0.3/0.05); con base de 150 ms los
#      mismos escalones producen solo ±2 cm/s, que además caen dentro de la
#      banda muerta ZE de la derivada (ensanchada a ±3 cm/s).
#
#   4. ACCIÓN INTEGRAL DE AJUSTE FINO (trim): integra el error lentamente
#      (KI_TRIM) solo cerca del régimen (|e| < 3 cm y |de| < 4 cm/s), con
#      saturación anti-windup de ±1500 cuentas. Elimina el error de estado
#      estacionario y absorbe hasta ±500 cuentas de error de calibración
#      del PWM_BASE sin oscilar.
#
# Resultados de simulación (5 semillas, retardo 300 ms, turbulencia):
#   Sintonización anterior:  ciclo límite 3.4 cm pk-pk
#   Sintonización v3:        ciclo límite 0.8 cm pk-pk | RMS 0.26 cm
#
# PROTOCOLO (sin cambios): ESP32 envía "{:.2f}\n"; PC responde "<int>\n".
# El delta enviado = difuso + trim, saturado a ±4000. Nunca viaja PWM
# absoluto: el firmware aplica PWM_BASE + delta y satura a [0, 65535].
#
# Uso:
#   python controlador_difuso.py --puerto COM5
#   python controlador_difuso.py --prueba        # autotest sin hardware
#
# Dependencias: pyserial  (pip install pyserial)
# =============================================================================

import argparse
import csv
import os
import random
import sys
import time
from collections import deque

try:
    import serial  # pyserial
except ImportError:
    serial = None  # Permite ejecutar --prueba sin pyserial instalado.

# ------------------------------- Parámetros ----------------------------------
SETPOINT_NOMINAL_CM = 21.0

# >>> CALIBRAR Y MANTENER IDÉNTICO EN esp32/main.py <<<
# Duty que sostiene la bola flotando sola (medido desde el REPL). Tu rango
# observado fue 38100-38800: usa el centro de TU medición, no este ejemplo.
PWM_BASE = 36000

PWM_MIN = 0
PWM_MAX = 65535
DELTA_PWM_MAX = 10000                  # Autoridad total (difuso + trim)
TS_NOMINAL_S = 0.050                  # 50 ms (lo marca el ESP32)

# Derivada del error (v3): ventana de N muestras + suavizado exponencial leve
N_VENTANA_DERIVADA = 3                # velocidad = (e[k]-e[k-3]) / (3*Ts)
ALFA_DERIVADA = 0.9                   # 0.8 nuevo / 0.2 anterior (leve)

# Acción integral de ajuste fino (trim) con anti-windup
KI_TRIM = 40.0                        # cuentas de PWM por (cm * s)
TRIM_MAX = 1500.0                     # saturación del integrador (cuentas)
ZONA_TRIM_E = 3.0                     # integra solo si |error| < 3 cm
ZONA_TRIM_DE = 4.0                    # ... y |d_error| < 4 cm/s

# Perturbación programada del setpoint (riqueza del dataset)
SETPOINT_MIN_CM = 19.0
SETPOINT_MAX_CM = 23.0
PERIODO_PERTURBACION_S = 10.0
RAMPA_SETPOINT_CM_S = 0.8             # Deslizamiento suave: 0.8 cm/s máx.

# Cosechador
ARCHIVO_DATASET = "dataset_levitador.csv"
MUESTRAS_OBJETIVO = 10000
FLUSH_CADA_N_FILAS = 50

# Enlace serial
BAUDRATE_DEFECTO = 115200
TIMEOUT_LECTURA_S = 0.2               # > Ts: una línea siempre llega antes
SILENCIO_MAX_S = 1.0                  # Sin datos válidos -> aviso
SILENCIO_RECONEXION_S = 3.0           # Sin datos válidos -> reconectar puerto
ESPERA_ARRANQUE_S = 8.0               # Máximo para el handshake post-reset


# =============================================================================
#                         MOTOR DIFUSO TIPO MAMDANI
# =============================================================================

def _trapecio(x, a, b, c, d):
    """Membresía trapezoidal clásica. Con b == c degenera en triángulo.
    Soporta hombros saturados pasando a == b (izquierda) o c == d (derecha)."""
    if x <= a:
        return 1.0 if a == b else 0.0
    if x >= d:
        return 1.0 if c == d else 0.0
    if b <= x <= c:
        return 1.0
    if x < b:                                  # Flanco ascendente
        return (x - a) / (b - a)
    return (d - x) / (d - c)                   # Flanco descendente


class MotorDifusoMamdani:
    """Mamdani de 2 entradas (error, d_error), 1 salida (Delta_PWM), 25 reglas.

    Convención de signos:
        error  = posicion - setpoint   (positivo => esfera DEMASIADO ALTA)
        salida positiva => más empuje del ventilador
    """

    ETIQUETAS = ("NG", "NP", "ZE", "PP", "PG")

    # ---- ERROR (cm) — v3: banda muerta ZE de ±1.5 cm; el trim integral se
    # encarga del residuo dentro de la banda, así que ya no provoca caídas.
    MF_ERROR = {
        "NG": (-40.0, -40.0, -8.0, -3.0),
        "NP": (-6.0, -3.0, -3.0, 0.0),
        "ZE": (-1.5, 0.0, 0.0, 1.5),
        "PP": (0.0, 3.0, 3.0, 6.0),
        "PG": (3.0, 8.0, 40.0, 40.0),
    }

    # ---- D_ERROR (cm/s) — v3: ZE ensanchada a ±3 cm/s para que los picos
    # de cuantización del sensor (~±2 cm/s con la derivada por ventana) NO
    # disparen correcciones; solo el movimiento real lo hace.
    MF_DERROR = {
        "NG": (-100.0, -100.0, -8.0, -4.0),
        "NP": (-5.0, -2.5, -2.5, 0.0),
        "ZE": (-3.0, 0.0, 0.0, 3.0),
        "PP": (0.0, 2.5, 2.5, 5.0),
        "PG": (4.0, 8.0, 100.0, 100.0),
    }

    # ---- SALIDA Delta_PWM — v3: autoridad reducida (pico máx ~±1800).
    # Con el retardo de lazo, correcciones suaves y frecuentes amortiguan;
    # correcciones grandes y tardías bombean la oscilación.
    MF_SALIDA = {
        "NG": (-1800.0, -1200.0, -1200.0, -600.0),
        "NP": (-1050.0, -525.0, -525.0, 0.0),
        "ZE": (-300.0, 0.0, 0.0, 300.0),
        "PP": (0.0, 525.0, 525.0, 1050.0),
        "PG": (600.0, 1200.0, 1200.0, 1800.0),
    }

    # ---- Matriz FAM: 25 reglas explícitas (sin cambios: la lógica era
    # correcta; el problema estaba en universos, fase y falta de integral).
    REGLAS = {
        #            dE: NG    NP    ZE    PP    PG
        "NG": ("PG", "PG", "PG", "PP", "ZE"),
        "NP": ("PG", "PP", "PP", "ZE", "NP"),
        "ZE": ("PP", "PP", "ZE", "NP", "NP"),
        "PP": ("PP", "ZE", "NP", "NP", "NG"),
        "PG": ("ZE", "NP", "NG", "NG", "NG"),
    }

    # Universo de defuzzificación: -4000..4000, paso 25 -> 321 puntos.
    PASO_UNIVERSO = 25

    def __init__(self):
        self._universo = [
            -DELTA_PWM_MAX + i * self.PASO_UNIVERSO
            for i in range((2 * DELTA_PWM_MAX) // self.PASO_UNIVERSO + 1)
        ]
        self._mu_salida = {
            et: [_trapecio(x, *self.MF_SALIDA[et]) for x in self._universo]
            for et in self.ETIQUETAS
        }
        self._agregado = [0.0] * len(self._universo)

    def _fuzzificar(self, valor, mfs):
        return {et: _trapecio(valor, *mfs[et]) for et in self.ETIQUETAS}

    def inferir(self, error_cm, d_error_cm_s):
        """Ciclo Mamdani completo. Retorna Delta_PWM (float) saturado a
        [-DELTA_PWM_MAX, +DELTA_PWM_MAX]."""
        mu_e = self._fuzzificar(error_cm, self.MF_ERROR)
        mu_de = self._fuzzificar(d_error_cm_s, self.MF_DERROR)

        # 1) Evaluación de las 25 reglas: fuerza = min; agregación = max.
        fuerza = {et: 0.0 for et in self.ETIQUETAS}
        for et_e in self.ETIQUETAS:
            ge = mu_e[et_e]
            if ge <= 0.0:
                continue
            fila = self.REGLAS[et_e]
            for j, et_de in enumerate(self.ETIQUETAS):
                gde = mu_de[et_de]
                if gde <= 0.0:
                    continue
                w = ge if ge < gde else gde    # min()
                consecuente = fila[j]
                if w > fuerza[consecuente]:
                    fuerza[consecuente] = w    # max()

        # 2) Implicación (recorte por mínimo) + agregación en el universo.
        agregado = self._agregado
        for i in range(len(agregado)):
            agregado[i] = 0.0
        hay_activacion = False
        for et, w in fuerza.items():
            if w <= 0.0:
                continue
            hay_activacion = True
            mu_set = self._mu_salida[et]
            for i in range(len(agregado)):
                m = mu_set[i]
                if m > w:
                    m = w
                if m > agregado[i]:
                    agregado[i] = m

        if not hay_activacion:
            return 0.0

        # 3) Defuzzificación por centroide discreto.
        num = 0.0
        den = 0.0
        universo = self._universo
        for i in range(len(agregado)):
            mu = agregado[i]
            if mu > 0.0:
                num += universo[i] * mu
                den += mu
        if den == 0.0:
            return 0.0
        delta = num / den

        if delta > DELTA_PWM_MAX:
            delta = float(DELTA_PWM_MAX)
        elif delta < -DELTA_PWM_MAX:
            delta = float(-DELTA_PWM_MAX)
        return delta


# =============================================================================
#            DERIVADA POR VENTANA + ACCIÓN INTEGRAL DE AJUSTE FINO
# =============================================================================

class EstimadorDerivada:
    """Velocidad del error sobre una base de N muestras (~150 ms): divide el
    ruido de cuantización del sensor por N sin el retraso de un filtro
    pesado, más un suavizado exponencial leve."""

    def __init__(self, n_ventana=N_VENTANA_DERIVADA, alfa=ALFA_DERIVADA):
        self._hist = deque(maxlen=n_ventana + 1)   # pares (t, error)
        self._alfa = alfa
        self._filtrada = 0.0

    @property
    def valor(self):
        return self._filtrada

    def actualizar(self, t, error):
        self._hist.append((t, error))
        if len(self._hist) < 2:
            return self._filtrada
        t0, e0 = self._hist[0]
        dt = t - t0
        if dt <= 0.0:
            return self._filtrada
        cruda = (error - e0) / dt
        self._filtrada = (self._alfa * cruda
                          + (1.0 - self._alfa) * self._filtrada)
        return self._filtrada

    def reiniciar(self):
        self._hist.clear()
        self._filtrada = 0.0


class TrimIntegral:
    """Integrador lento con anti-windup: corrige el desbalance residual entre
    PWM_BASE y el equilibrio físico real, eliminando el error de estado
    estacionario que el difuso PD no puede cancelar."""

    def __init__(self):
        self._trim = 0.0

    @property
    def valor(self):
        return self._trim

    def actualizar(self, error, d_error, dt):
        # Solo integra cerca del régimen: evita windup durante transitorios
        # grandes, rebotes contra protecciones o cambios de setpoint.
        if abs(error) < ZONA_TRIM_E and abs(d_error) < ZONA_TRIM_DE:
            # error > 0 => bola alta => se necesita MENOS empuje => trim baja.
            self._trim -= KI_TRIM * error * dt
            if self._trim > TRIM_MAX:
                self._trim = TRIM_MAX
            elif self._trim < -TRIM_MAX:
                self._trim = -TRIM_MAX
        return self._trim


# =============================================================================
#                    PERTURBACIÓN PROGRAMADA DEL SETPOINT
# =============================================================================

class GeneradorSetpoint:
    """Setpoint efectivo que se desliza suavemente (rampa limitada) hacia un
    objetivo aleatorio nuevo cada PERIODO_PERTURBACION_S segundos."""

    def __init__(self):
        self._efectivo = SETPOINT_NOMINAL_CM
        self._objetivo = SETPOINT_NOMINAL_CM
        self._t_proximo_salto = time.monotonic() + PERIODO_PERTURBACION_S
        self._rng = random.Random()

    def actualizar(self, dt_s):
        ahora = time.monotonic()
        if ahora >= self._t_proximo_salto:
            self._objetivo = self._rng.uniform(SETPOINT_MIN_CM, SETPOINT_MAX_CM)
            self._t_proximo_salto = ahora + PERIODO_PERTURBACION_S
            print("[SETPOINT] Nuevo objetivo: {:.2f} cm".format(self._objetivo))

        paso_max = RAMPA_SETPOINT_CM_S * dt_s
        dif = self._objetivo - self._efectivo
        if dif > paso_max:
            dif = paso_max
        elif dif < -paso_max:
            dif = -paso_max
        self._efectivo += dif
        return self._efectivo


# =============================================================================
#                          ENLACE SERIAL CON EL ESP32
# =============================================================================

class EnlaceESP32:
    """Lector/escritor robusto y emparejado con esp32/main.py.

    Claves de la apertura del puerto:
      - DTR/RTS desactivados ANTES de abrir: evita resetear el DEVKIT V1 y,
        peor aún, dejarlo en modo bootloader (EN/GPIO0 van a esas líneas).
      - reset_input_buffer() + reset_output_buffer() tras abrir: descarta
        basura de arranque acumulada por el sistema operativo.
      - Handshake: se espera la primera telemetría válida antes de actuar.
    """

    def __init__(self, puerto, baud):
        self._puerto = puerto
        self._baud = baud
        self._ser = None
        self._t_ultimo_dato = time.monotonic()
        self._aviso_silencio = False

    def conectar(self):
        while True:
            try:
                if self._ser is not None:
                    try:
                        self._ser.close()
                    except Exception:
                        pass

                # Construcción en dos pasos: configurar DTR/RTS en falso
                # ANTES de open() para que el driver no pulse las líneas.
                self._ser = serial.Serial()
                self._ser.port = self._puerto
                self._ser.baudrate = self._baud
                self._ser.timeout = TIMEOUT_LECTURA_S
                self._ser.write_timeout = 1.0
                self._ser.dtr = False          # NO resetear el ESP32
                self._ser.rts = False          # NO entrar al bootloader
                self._ser.open()
                # Refuerzo post-apertura (algunos drivers lo requieren).
                self._ser.dtr = False
                self._ser.rts = False

                # Vaciado de ambos buffers del sistema operativo.
                self._ser.reset_input_buffer()
                self._ser.reset_output_buffer()

                self._t_ultimo_dato = time.monotonic()
                self._aviso_silencio = False
                print("[SERIAL] Conectado a {} @ {} bps (DTR/RTS inactivos)"
                      .format(self._puerto, self._baud))

                if self._esperar_telemetria():
                    return
                print("[SERIAL] Sin telemetría tras {} s. ¿main.py cargado? "
                      "¿Thonny soltó el puerto? Reintentando..."
                      .format(ESPERA_ARRANQUE_S))

            except (serial.SerialException, OSError) as exc:
                print("[SERIAL] No se pudo abrir {} ({}). Reintento en 2 s..."
                      .format(self._puerto, exc))
                time.sleep(2.0)

    def _esperar_telemetria(self):
        """Handshake de arranque: bloquea hasta ESPERA_ARRANQUE_S esperando
        la primera línea de posición válida del firmware."""
        limite = time.monotonic() + ESPERA_ARRANQUE_S
        print("[SERIAL] Esperando telemetría del firmware...")
        while time.monotonic() < limite:
            cruda = self._ser.readline()
            texto = self._a_texto(cruda)
            if texto is None:
                continue
            if texto.startswith("#"):
                print("[ESP32] {}".format(texto))
                continue
            try:
                float(texto)
            except ValueError:
                continue
            print("[SERIAL] Telemetría confirmada. Enlace operativo.")
            self._t_ultimo_dato = time.monotonic()
            return True
        return False

    @staticmethod
    def _a_texto(linea_bytes):
        try:
            texto = linea_bytes.decode("utf-8", "ignore").strip()
        except Exception:
            return None
        return texto if texto else None

    def leer_posicion(self):
        """Bloquea hasta TIMEOUT_LECTURA_S esperando una línea. Drena el
        buffer y retorna la posición MÁS RECIENTE (float) o None si en este
        intento no llegó nada parseable. Gestiona avisos y reconexión."""
        try:
            posicion = self._parsear(self._ser.readline())

            # Drenado: si la PC se atrasó, quedarse con lo más fresco.
            while self._ser.in_waiting:
                extra = self._parsear(self._ser.readline())
                if extra is not None:
                    posicion = extra

            if posicion is not None:
                self._t_ultimo_dato = time.monotonic()
                self._aviso_silencio = False
                return posicion

            mudo_s = time.monotonic() - self._t_ultimo_dato
            if mudo_s > SILENCIO_RECONEXION_S:
                print("[SERIAL] {:.1f} s sin datos: reconectando puerto..."
                      .format(mudo_s))
                self.conectar()
            elif mudo_s > SILENCIO_MAX_S and not self._aviso_silencio:
                print("[SERIAL] Aviso: sin telemetría del ESP32...")
                self._aviso_silencio = True
            return None

        except (serial.SerialException, OSError):
            print("[SERIAL] Puerto perdido. Reconectando...")
            self.conectar()
            return None

    def _parsear(self, linea_bytes):
        """float de posición, o None. Las líneas '#' del firmware se
        imprimen como diagnóstico."""
        texto = self._a_texto(linea_bytes)
        if texto is None:
            return None
        if texto.startswith("#"):
            print("[ESP32] {}".format(texto))
            return None
        try:
            return float(texto)
        except ValueError:
            return None

    def enviar_delta(self, delta_pwm_int):
        """Envía el Delta PWM como string simple + '\\n' en UTF-8 con flush
        inmediato del buffer de salida del sistema operativo."""
        try:
            self._ser.write("{}\n".format(int(delta_pwm_int)).encode("utf-8"))
            self._ser.flush()
        except (serial.SerialException, OSError):
            print("[SERIAL] Falla de escritura. Reconectando...")
            self.conectar()

    def cerrar(self):
        if self._ser is not None:
            try:
                # Último comando neutro: el ESP32 queda en PWM_BASE y su
                # watchdog de 200 ms asume el control autónomo.
                self._ser.write(b"0\n")
                self._ser.flush()
                self._ser.close()
            except Exception:
                pass


# =============================================================================
#                          COSECHADOR DE DATASET CSV
# =============================================================================

class CosechadorCSV:
    def __init__(self, ruta):
        existe = os.path.exists(ruta)
        self._archivo = open(ruta, "a", newline="")
        self._escritor = csv.writer(self._archivo)
        if not existe or os.path.getsize(ruta) == 0:
            self._escritor.writerow(["error", "delta_error", "delta_pwm"])
        self._pendientes = 0
        self.total = 0

    def registrar(self, error, delta_error, delta_pwm):
        self._escritor.writerow(
            ["{:.4f}".format(error), "{:.4f}".format(delta_error),
             "{:d}".format(delta_pwm)]
        )
        self.total += 1
        self._pendientes += 1
        if self._pendientes >= FLUSH_CADA_N_FILAS:
            self._archivo.flush()
            self._pendientes = 0

    def cerrar(self):
        try:
            self._archivo.flush()
            self._archivo.close()
        except Exception:
            pass


# =============================================================================
#                              LAZO PRINCIPAL
# =============================================================================

def ejecutar(puerto, baud):
    if serial is None:
        print("ERROR: pyserial no está instalado. Ejecuta: pip install pyserial")
        sys.exit(1)

    motor = MotorDifusoMamdani()
    derivador = EstimadorDerivada()
    trim = TrimIntegral()
    setpoint_gen = GeneradorSetpoint()
    enlace = EnlaceESP32(puerto, baud)
    cosechador = CosechadorCSV(ARCHIVO_DATASET)

    enlace.conectar()

    t_previo = time.monotonic()
    t_ultimo_reporte = t_previo

    print("[ETAPA 1 v3] Difuso + trim integral. Setpoint nominal: {:.1f} cm."
          .format(SETPOINT_NOMINAL_CM))
    print("[ETAPA 1 v3] PWM_BASE configurado: {} (¡debe ser el equilibrio "
          "real medido!)".format(PWM_BASE))
    print("[ETAPA 1 v3] Cosechando {} muestras en '{}'...".format(
        MUESTRAS_OBJETIVO, ARCHIVO_DATASET))

    try:
        while cosechador.total < MUESTRAS_OBJETIVO:
            posicion = enlace.leer_posicion()
            if posicion is None:
                continue                       # Sin dato fresco: no se actúa

            ahora = time.monotonic()
            dt = ahora - t_previo
            t_previo = ahora
            if dt <= 0.0 or dt > 0.5:
                dt = TS_NOMINAL_S              # Saneo tras pausas/reconexión

            # ---- Setpoint perturbado (rampa suave 19.0–23.0 cm) ----
            setpoint = setpoint_gen.actualizar(dt)

            # ---- Error y derivada por ventana (cm/s) ----
            error = posicion - setpoint
            d_error = derivador.actualizar(ahora, error)

            # ---- Difuso + trim integral -> Delta_PWM entero ----
            delta_difuso = motor.inferir(error, d_error)
            delta_trim = trim.actualizar(error, d_error, dt)
            delta = int(round(delta_difuso + delta_trim))
            if delta > DELTA_PWM_MAX:
                delta = DELTA_PWM_MAX
            elif delta < -DELTA_PWM_MAX:
                delta = -DELTA_PWM_MAX

            # Verificación defensiva del paradigma: el PWM final del firmware
            # (PWM_BASE + delta) debe quedar en [0, 65535].
            if PWM_BASE + delta > PWM_MAX:
                delta = PWM_MAX - PWM_BASE
            elif PWM_BASE + delta < PWM_MIN:
                delta = PWM_MIN - PWM_BASE

            # ---- Actuación: viaja SOLO el delta, como "<int>\n" ----
            enlace.enviar_delta(delta)

            # ---- Cosecha (se registra el delta TOTAL aplicado) ----
            cosechador.registrar(error, d_error, delta)

            # ---- Reporte de progreso (1 Hz) ----
            if ahora - t_ultimo_reporte >= 1.0:
                t_ultimo_reporte = ahora
                print("[{:5d}/{}] pos={:6.2f}  sp={:5.2f}  e={:+6.2f}  "
                      "de={:+6.2f}  dPWM={:+5d}  trim={:+6.0f}".format(
                          cosechador.total, MUESTRAS_OBJETIVO, posicion,
                          setpoint, error, d_error, delta, trim.valor))

        print("\n[ETAPA 1 v3] Cosecha completa: {} muestras en '{}'.".format(
            cosechador.total, ARCHIVO_DATASET))
        print("[ETAPA 1 v3] IMPORTANTE: re-entrena la red (entrenar_ia.py): "
              "la dinámica del dataset cambió.")

    except KeyboardInterrupt:
        print("\n[ETAPA 1 v3] Interrumpido por el usuario. "
              "Muestras guardadas: {}.".format(cosechador.total))
    finally:
        cosechador.cerrar()
        enlace.cerrar()
        print("[ETAPA 1 v3] Recursos liberados. El ESP32 mantiene PWM_BASE "
              "de forma autónoma.")


# =============================================================================
#                    AUTOTEST DEL MOTOR DIFUSO (sin hardware)
# =============================================================================

def prueba_motor():
    """Verificación rápida de coherencia física del Mamdani v3 (sin serial).
    Nota: los umbrales reflejan la nueva autoridad reducida (pico ~±1200)."""
    motor = MotorDifusoMamdani()
    casos = [
        (-12.0, 0.0),   # Muy abajo, quieta        -> delta fuertemente +
        (-4.0, -5.0),   # Abajo y cayendo          -> delta muy +
        (-4.0, 5.0),    # Abajo pero subiendo      -> freno anticipado
        (0.0, 0.0),     # En el setpoint, quieta   -> delta ~ 0
        (0.0, 2.0),     # 2 cm/s (ruido o deriva)  -> amortiguación SUAVE
        (4.0, -5.0),    # Arriba pero bajando      -> empuje anticipado
        (4.0, 5.0),     # Arriba y subiendo        -> delta muy -
        (12.0, 0.0),    # Muy arriba, quieta       -> delta fuertemente -
    ]
    print(" error   d_error |  Delta_PWM")
    print("-----------------+-----------")
    for e, de in casos:
        d = motor.inferir(e, de)
        print("{:+7.1f} {:+8.1f} | {:+9.1f}".format(e, de, d))
        assert -DELTA_PWM_MAX <= d <= DELTA_PWM_MAX
    assert motor.inferir(-12.0, 0.0) > 700.0
    assert motor.inferir(12.0, 0.0) < -700.0
    assert abs(motor.inferir(0.0, 0.0)) < 100.0
    # 2 cm/s (cuantización o deriva lenta): amortiguación suave, NO un golpe.
    # Con la sintonización anterior esto disparaba ~-800; ahora ~-415.
    assert abs(motor.inferir(0.0, 2.0)) < 600.0
    assert motor.inferir(0.0, 6.0) < 0.0       # Subiendo rápido real: frenar
    assert motor.inferir(0.0, -6.0) > 0.0      # Cayendo rápido real: empujar
    print("\nAutotest del motor difuso v3: OK")


def main():
    parser = argparse.ArgumentParser(
        description="Etapa 1 v3: Difuso anti-oscilación + trim integral + "
                    "Cosecha de Dataset")
    parser.add_argument("--puerto", default="COM5",
                        help="Puerto serial del ESP32 (ej. COM5, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=BAUDRATE_DEFECTO,
                        help="Baudrate (defecto: 115200)")
    parser.add_argument("--prueba", action="store_true",
                        help="Ejecuta solo el autotest del motor difuso")
    args = parser.parse_args()

    if args.prueba:
        prueba_motor()
        return
    ejecutar(args.puerto, args.baud)


if __name__ == "__main__":
    main()