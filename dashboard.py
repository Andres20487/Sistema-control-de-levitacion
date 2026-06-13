# =============================================================================
# etapa3_dashboard.py — Levitador Neumático | Dashboard + AGENTE IA MAESTRO
# -----------------------------------------------------------------------------
# Telemetría en tiempo real + interfaz de chat en lenguaje natural que
# comanda el setpoint y el algoritmo de control del ESP32.
#
# Arquitectura:
#   [Hilo lector serial] --(deque 150)--> [FuncAnimation @ 50 ms]
#                ^                                |
#                |  escribir() thread-safe        v
#   [TextBox + Button] --regex NLP--> trama "<SET:15.0,ALGO:1>"
#
#   - El hilo lector aísla los bloqueos del puerto: la animación nunca se
#     congela; el hilo reconecta solo ante desconexiones.
#   - El chat NO interrumpe la gráfica: la escritura serial es un write()
#     puntual protegido por candado, en el hilo de la GUI.
#   - DTR/RTS desactivados al abrir (DEVKIT V1: esas líneas resetean la
#     placa vía EN/GPIO0).
#
# NLP por expresiones regulares — ejemplos que entiende:
#   "Lleva la bola a 15cm usando lógica difusa"        -> <SET:15.0,ALGO:1>
#   "sube a 24.5 con la red neuronal"                  -> <SET:24.5,ALGO:2>
#   "usa q-learning"                                   -> <ALGO:3>
#   "deep rl a 18"                                     -> <SET:18.0,ALGO:4>
#   "setpoint 21" / "algoritmo 2"                      -> parciales
#
# Algoritmos: 1=Lógica Difusa | 2=Red Neuronal | 3=Q-Learning Tabular |
#             4=Deep RL (DQN). Setpoint válido: 12.0 a 28.0 cm (envolvente
#             operativa del firmware; fuera de rango se rechaza con aviso).
#
# Protocolo de entrada (esp32/main.py v7):
#   "{:.2f}\n" -> telemetría (posición cm)   | "# ...\n" -> diagnóstico/ACK
#
# Uso:  python etapa3_dashboard.py [COM7]
# Dependencias: pip install pyserial matplotlib
# =============================================================================

import re
import sys
import threading
import time
from collections import deque

import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

# ----------------------------- CONFIGURACIÓN ---------------------------------
PUERTO_SERIAL = "COM5"
BAUDIOS = 115200
TIMEOUT_LECTURA_S = 0.2

SETPOINT_CM = 21.0                # Setpoint inicial (el del firmware)
SETPOINT_MIN_CM = 12.0            # Envolvente comandable (igual al firmware)
SETPOINT_MAX_CM = 28.0
LONGITUD_TUBO_CM = 38.0

VENTANA_MUESTRAS = 150            # Buffer circular: 150 muestras = 7.5 s
TS_S = 0.050
INTERVALO_ANIMACION_MS = 50

REINTENTO_CONEXION_S = 2.0

ANSI_CIAN = "\033[96m"
ANSI_AMARILLO = "\033[93m"
ANSI_VERDE = "\033[92m"
ANSI_RESET = "\033[0m"

NOMBRES_ALGO = {1: "Lógica Difusa", 2: "Red Neuronal",
                3: "Q-Learning Tabular", 4: "Deep RL (DQN)"}


# =============================================================================
#                 NLP POR EXPRESIONES REGULARES (AGENTE MAESTRO)
# =============================================================================

# Algoritmo por nombre (el orden importa: lo específico antes que lo general;
# "red neuronal profunda" debe resolver a Deep RL, no a la MLP).
_PATRONES_ALGO = (
    (re.compile(r"deep|dqn|profund|drl|refuerzo\s+profundo", re.I), 4),
    (re.compile(r"q[\s_\-]?learn|tabular|tabla|q[\s_\-]?tabla", re.I), 3),
    (re.compile(r"difus|fuzzy|mamdani|borros", re.I), 1),
    (re.compile(r"neuro|mlp|perceptr|\bred\b", re.I), 2),
)
# Algoritmo por número explícito: "algoritmo 2", "algo: 3", "usa el 4"... solo
# con la palabra clave para no confundirlo con el setpoint.
_PATRON_ALGO_NUM = re.compile(r"algo(?:ritmo)?\s*[:=]?\s*([1-4])\b", re.I)

# Setpoint: prioridad 1) número pegado a 'cm'; 2) número tras verbo/preposición
# de destino; 3) cualquier número restante.
_PATRON_SP_CM = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*cm", re.I)
_PATRON_SP_DESTINO = re.compile(
    r"(?:\ba\b|\ben\b|hasta|hacia|sube\w*|baja\w*|lleva\w*|pon\w*|"
    r"setpoint|objetivo|altura|target)\s*(?:la\s+bola\s+a\s*)?[:=]?\s*"
    r"(-?\d+(?:[.,]\d+)?)", re.I)
_PATRON_SP_LIBRE = re.compile(r"(-?\d+(?:[.,]\d+)?)")


def interpretar_comando(texto):
    """Extrae (setpoint | None, algo | None, mensaje) del lenguaje natural.

    Nunca lanza: cualquier texto produce una interpretación o un mensaje de
    rechazo explicando qué faltó o qué quedó fuera de rango.
    """
    if not texto or not texto.strip():
        return None, None, "Comando vacío."
    texto = texto.strip()

    # ---- Algoritmo ----
    algo = None
    texto_sin_algo = texto
    m = _PATRON_ALGO_NUM.search(texto)
    if m:
        algo = int(m.group(1))
        # Quitar el número del algoritmo para no confundirlo con el setpoint.
        texto_sin_algo = texto[:m.start()] + " " + texto[m.end():]
    else:
        for patron, codigo in _PATRONES_ALGO:
            if patron.search(texto):
                algo = codigo
                break

    # ---- Setpoint ----
    setpoint = None
    m = (_PATRON_SP_CM.search(texto_sin_algo)
         or _PATRON_SP_DESTINO.search(texto_sin_algo)
         or _PATRON_SP_LIBRE.search(texto_sin_algo))
    if m:
        try:
            setpoint = float(m.group(1).replace(",", "."))
        except ValueError:
            setpoint = None

    if setpoint is None and algo is None:
        return None, None, ("No entendí. Ej.: 'lleva la bola a 15cm con "
                            "lógica difusa'.")
    if setpoint is not None and not (
            SETPOINT_MIN_CM <= setpoint <= SETPOINT_MAX_CM):
        return None, None, ("Setpoint {:.1f} fuera de la envolvente segura "
                            "[{:.0f}, {:.0f}] cm.".format(
                                setpoint, SETPOINT_MIN_CM, SETPOINT_MAX_CM))

    partes = []
    if setpoint is not None:
        partes.append("objetivo {:.1f} cm".format(setpoint))
    if algo is not None:
        partes.append(NOMBRES_ALGO[algo])
    return setpoint, algo, "Entendido: " + " + ".join(partes) + "."


def construir_trama(setpoint, algo):
    """Trama estricta del protocolo. Admite campos parciales."""
    campos = []
    if setpoint is not None:
        campos.append("SET:{:.1f}".format(setpoint))
    if algo is not None:
        campos.append("ALGO:{:d}".format(algo))
    return "<" + ",".join(campos) + ">"


# =============================================================================
#                  HILO LECTOR/ESCRITOR SERIAL (PRODUCTOR)
# =============================================================================

class LectorSerial(threading.Thread):
    """Lee el puerto en segundo plano (telemetría -> deque; '#' -> consola)
    y expone escribir() thread-safe para inyectar tramas sin interrumpir la
    animación. Inmune a tramas corruptas y desconexiones: reconecta solo."""

    def __init__(self, puerto, baudios, buffer_posiciones):
        super().__init__(daemon=True)
        self._puerto = puerto
        self._baudios = baudios
        self._buffer = buffer_posiciones
        self._ser = None
        self._candado_tx = threading.Lock()
        self._detener = threading.Event()
        self.conectado = False

    def detener(self):
        self._detener.set()

    # ------------------------- Gestión del puerto ---------------------------

    def _abrir_puerto(self):
        try:
            self._cerrar_puerto()
            self._ser = serial.Serial()
            self._ser.port = self._puerto
            self._ser.baudrate = self._baudios
            self._ser.timeout = TIMEOUT_LECTURA_S
            self._ser.write_timeout = 1.0
            self._ser.dtr = False             # NO resetear el ESP32
            self._ser.rts = False             # NO entrar al bootloader
            self._ser.open()
            self._ser.dtr = False
            self._ser.rts = False
            self._ser.reset_input_buffer()
            self.conectado = True
            print("{}[DASHBOARD] Conectado a {} @ {} bps{}".format(
                ANSI_AMARILLO, self._puerto, self._baudios, ANSI_RESET))
            return True
        except (serial.SerialException, OSError) as exc:
            self.conectado = False
            print("{}[DASHBOARD] Sin acceso a {} ({}). Reintento en {} s...{}"
                  .format(ANSI_AMARILLO, self._puerto, exc,
                          REINTENTO_CONEXION_S, ANSI_RESET))
            return False

    def _cerrar_puerto(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        self.conectado = False

    # ------------------- Escritura thread-safe (chat IA) --------------------

    def escribir(self, linea):
        """Inyecta una trama por serial desde el hilo de la GUI. Retorna
        True si salió al cable. Nunca lanza ni bloquea la animación."""
        with self._candado_tx:
            if self._ser is None or not self.conectado:
                return False
            try:
                self._ser.write((linea + "\n").encode("utf-8"))
                self._ser.flush()
                return True
            except (serial.SerialException, OSError):
                self._cerrar_puerto()         # El hilo lector reconectará
                return False

    # --------------------------- Bucle del hilo -----------------------------

    def run(self):
        while not self._detener.is_set():
            if not self.conectado:
                if not self._abrir_puerto():
                    self._detener.wait(REINTENTO_CONEXION_S)
                    continue
            try:
                linea = self._ser.readline()
            except (serial.SerialException, OSError):
                print("{}[DASHBOARD] Puerto perdido. Reconectando...{}"
                      .format(ANSI_AMARILLO, ANSI_RESET))
                self._cerrar_puerto()
                continue
            self._procesar(linea)
        self._cerrar_puerto()

    def _procesar(self, linea_bytes):
        """Discrimina telemetría / diagnóstico / basura. Nunca lanza."""
        try:
            texto = linea_bytes.decode("utf-8", "ignore").strip()
        except Exception:
            return
        if not texto:
            return
        if texto.startswith("#"):
            print("{}[ESP32] {}{}".format(ANSI_CIAN, texto, ANSI_RESET))
            return
        try:
            posicion = float(texto)
        except ValueError:
            return
        if 0.0 <= posicion <= LONGITUD_TUBO_CM:
            self._buffer.append(posicion)


# =============================================================================
#                  DASHBOARD GRÁFICO + CHAT DEL AGENTE MAESTRO
# =============================================================================

def construir_dashboard(buffer_posiciones, lector):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.canvas.manager.set_window_title(
        "Levitador Neumático — Agente IA Maestro")
    # Reservar la franja inferior para la interfaz de chat.
    fig.subplots_adjust(bottom=0.24)

    estado = {"setpoint": SETPOINT_CM, "algo": None}

    ax.set_title("Levitador Neumático — Agente IA Maestro "
                 "(Difuso / MLP / Q-Learning / Deep RL)")
    ax.set_xlabel("Tiempo (s)  [ventana deslizante de {:.1f} s]".format(
        VENTANA_MUESTRAS * TS_S))
    ax.set_ylabel("Posición de la bola (cm)")
    ax.set_xlim(-VENTANA_MUESTRAS * TS_S, 0.0)
    ax.set_ylim(0.0, LONGITUD_TUBO_CM)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Setpoint DINÁMICO: línea punteada roja (se mueve con cada comando).
    linea_setpoint = ax.axhline(
        estado["setpoint"], color="red", linestyle="--", linewidth=1.5,
        label="Setpoint")
    (linea_bola,) = ax.plot([], [], color="blue", linewidth=1.8,
                            solid_joinstyle="round", label="Posición")
    ax.legend(loc="upper left", framealpha=0.9)

    cuadro_metricas = ax.text(
        0.985, 0.97, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5f5dc",
                  edgecolor="#888888", alpha=0.92))

    # ----------------------- Interfaz de chat (widgets) ---------------------
    texto_estado = fig.text(0.10, 0.145, "IA Maestro lista. Ej.: 'lleva la "
                            "bola a 15cm usando lógica difusa'",
                            fontsize=9, color="#444444")
    eje_caja = fig.add_axes([0.10, 0.05, 0.60, 0.07])
    caja = TextBox(eje_caja, "Comando ", textalignment="left")
    eje_boton = fig.add_axes([0.72, 0.05, 0.12, 0.07])
    boton = Button(eje_boton, "Enviar")

    def enviar_comando(texto):
        """Handler común de TextBox(Enter) y Button(click). Nunca lanza."""
        try:
            setpoint, algo, mensaje = interpretar_comando(texto)
            if setpoint is None and algo is None:
                texto_estado.set_text("✗ " + mensaje)
                texto_estado.set_color("#aa3300")
            else:
                trama = construir_trama(setpoint, algo)
                if lector.escribir(trama):
                    texto_estado.set_text(
                        "✓ {} -> {}".format(mensaje, trama))
                    texto_estado.set_color("#006622")
                    print("{}[IA MAESTRO] {} -> {}{}".format(
                        ANSI_VERDE, mensaje, trama, ANSI_RESET))
                    if setpoint is not None:
                        estado["setpoint"] = setpoint
                        linea_setpoint.set_ydata([setpoint, setpoint])
                    if algo is not None:
                        estado["algo"] = algo
                    caja.set_val("")
                else:
                    texto_estado.set_text(
                        "✗ Sin conexión serial: comando no enviado.")
                    texto_estado.set_color("#aa3300")
            fig.canvas.draw_idle()
        except Exception as exc:                   # Blindaje total de la GUI
            texto_estado.set_text("✗ Error interno: {}".format(exc))
            fig.canvas.draw_idle()

    caja.on_submit(enviar_comando)
    boton.on_clicked(lambda _evento: enviar_comando(caja.text))

    # --------------------------- Animación ----------------------------------
    def actualizar(_frame):
        datos = list(buffer_posiciones)            # Instantánea atómica
        if not datos:
            cuadro_metricas.set_text(
                "ESPERANDO TELEMETRÍA..." if lector.conectado
                else "SIN CONEXIÓN SERIAL")
            linea_bola.set_data([], [])
            return linea_bola, cuadro_metricas, linea_setpoint

        n = len(datos)
        eje_t = [(-(n - 1 - i)) * TS_S for i in range(n)]
        linea_bola.set_data(eje_t, datos)

        sp = estado["setpoint"]
        altura = datos[-1]
        algo_txt = (NOMBRES_ALGO[estado["algo"]]
                    if estado["algo"] else "firmware (defecto)")
        cuadro_metricas.set_text(
            "Altura : {:6.2f} cm\n"
            "Error  : {:+6.2f} cm\n"
            "Banda  : {:6.2f} cm\n"
            "IA     : {}".format(
                altura, altura - sp, max(datos) - min(datos), algo_txt))
        return linea_bola, cuadro_metricas, linea_setpoint

    animacion = FuncAnimation(
        fig, actualizar, interval=INTERVALO_ANIMACION_MS,
        blit=True, cache_frame_data=False)
    return fig, animacion


# =============================================================================
#                                    MAIN
# =============================================================================

def main():
    puerto = sys.argv[1] if len(sys.argv) > 1 else PUERTO_SERIAL

    buffer_posiciones = deque(maxlen=VENTANA_MUESTRAS)
    lector = LectorSerial(puerto, BAUDIOS, buffer_posiciones)
    lector.start()

    fig, _animacion = construir_dashboard(buffer_posiciones, lector)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        lector.detener()
        lector.join(timeout=2.0)
        print("{}[DASHBOARD] Cerrado. El ESP32 continúa en vuelo autónomo "
              "con su última configuración.{}".format(
                  ANSI_AMARILLO, ANSI_RESET))


if __name__ == "__main__":
    main()