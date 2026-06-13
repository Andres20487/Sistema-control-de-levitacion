# =============================================================================
# etapa3_dashboard.py — Pneumatic Levitator | Dashboard + MASTER AI AGENT
# -----------------------------------------------------------------------------
# Real-time telemetry + natural language chat interface that commands the
# setpoint and control algorithm of the ESP32.
#
# Usage:  python etapa3_dashboard.py [COM7]
# Dependencies: pip install pyserial matplotlib
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

# ----------------------------- CONFIGURATION ---------------------------------
PUERTO_SERIAL = "COM5"
BAUDIOS = 115200
TIMEOUT_LECTURA_S = 0.2

SETPOINT_CM = 21.0                # Initial Setpoint (firmware default)
SETPOINT_MIN_CM = 12.0            # Commandable envelope (firmware limits)
SETPOINT_MAX_CM = 28.0
LONGITUD_TUBO_CM = 38.0

VENTANA_MUESTRAS = 150            # Circular buffer: 150 samples = 7.5 s
TS_S = 0.050
INTERVALO_ANIMACION_MS = 50

REINTENTO_CONEXION_S = 2.0

ANSI_CIAN = "\033[96m"
ANSI_AMARILLO = "\033[93m"
ANSI_VERDE = "\033[92m"
ANSI_RESET = "\033[0m"

NOMBRES_ALGO = {1: "Fuzzy Logic", 2: "Neural Network",
                3: "Tabular Q-Learning", 4: "Deep RL (DQN)"}


# =============================================================================
#                 NLP VIA REGULAR EXPRESSIONS (MASTER AGENT)
# =============================================================================

# Algorithm by name (order matters: specific before general)
_PATRONES_ALGO = (
    (re.compile(r"deep|dqn|drl|deep\s+rl", re.I), 4),
    (re.compile(r"q[\s_\-]?learn|tabular|table|q[\s_\-]?table", re.I), 3),
    (re.compile(r"fuzzy|mamdani", re.I), 1),
    (re.compile(r"neuro|neural|mlp|perceptron|\bnet\b|\bnetwork\b", re.I), 2),
)
# Algorithm by explicit number: "algorithm 2", "algo: 3", "use 4"...
_PATRON_ALGO_NUM = re.compile(r"algo(?:rithm)?\s*[:=]?\s*([1-4])\b", re.I)

# Setpoint: 1) number next to 'cm'; 2) number after destination verbs/prepositions; 3) any remaining number.
_PATRON_SP_CM = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*cm", re.I)
_PATRON_SP_DESTINO = re.compile(
    r"(?:\bto\b|\bat\b|up|down|take\w*|move\w*|put\w*|set\w*|"
    r"setpoint|target|height)\s*(?:the\s+ball\s+to\s*)?[:=]?\s*"
    r"(-?\d+(?:[.,]\d+)?)", re.I)
_PATRON_SP_LIBRE = re.compile(r"(-?\d+(?:[.,]\d+)?)")


def interpretar_comando(texto):
    """Extracts (setpoint | None, algo | None, message) from natural language."""
    if not texto or not texto.strip():
        return None, None, "Empty command."
    texto = texto.strip()

    # ---- Algorithm ----
    algo = None
    texto_sin_algo = texto
    m = _PATRON_ALGO_NUM.search(texto)
    if m:
        algo = int(m.group(1))
        # Remove the algorithm number so it's not confused with the setpoint
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
        return None, None, ("I didn't understand. E.g.: 'take the ball to 15cm with "
                            "fuzzy logic'.")
    if setpoint is not None and not (
            SETPOINT_MIN_CM <= setpoint <= SETPOINT_MAX_CM):
        return None, None, ("Setpoint {:.1f} out of safe envelope "
                            "[{:.0f}, {:.0f}] cm.".format(
                                setpoint, SETPOINT_MIN_CM, SETPOINT_MAX_CM))

    partes = []
    if setpoint is not None:
        partes.append("target {:.1f} cm".format(setpoint))
    if algo is not None:
        partes.append(NOMBRES_ALGO[algo])
    return setpoint, algo, "Understood: " + " + ".join(partes) + "."


def construir_trama(setpoint, algo):
    """Strict protocol frame. Allows partial fields."""
    campos = []
    if setpoint is not None:
        campos.append("SET:{:.1f}".format(setpoint))
    if algo is not None:
        campos.append("ALGO:{:d}".format(algo))
    return "<" + ",".join(campos) + ">"


# =============================================================================
#                  SERIAL READER/WRITER THREAD (PRODUCER)
# =============================================================================

class LectorSerial(threading.Thread):
    """Reads the port in the background and exposes thread-safe escribir()."""

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

    # ------------------------- Port Management ---------------------------

    def _abrir_puerto(self):
        try:
            self._cerrar_puerto()
            self._ser = serial.Serial()
            self._ser.port = self._puerto
            self._ser.baudrate = self._baudios
            self._ser.timeout = TIMEOUT_LECTURA_S
            self._ser.write_timeout = 1.0
            self._ser.dtr = False             # DO NOT reset ESP32
            self._ser.rts = False             # DO NOT enter bootloader
            self._ser.open()
            self._ser.dtr = False
            self._ser.rts = False
            self._ser.reset_input_buffer()
            self.conectado = True
            print("{}[DASHBOARD] Connected to {} @ {} bps{}".format(
                ANSI_AMARILLO, self._puerto, self._baudios, ANSI_RESET))
            return True
        except (serial.SerialException, OSError) as exc:
            self.conectado = False
            print("{}[DASHBOARD] No access to {} ({}). Retrying in {} s...{}"
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

    # ------------------- Thread-safe Writing (AI Chat) --------------------

    def escribir(self, linea):
        """Injects a serial frame from the GUI thread."""
        with self._candado_tx:
            if self._ser is None or not self.conectado:
                return False
            try:
                self._ser.write((linea + "\n").encode("utf-8"))
                self._ser.flush()
                return True
            except (serial.SerialException, OSError):
                self._cerrar_puerto()
                return False

    # --------------------------- Thread Loop -----------------------------

    def run(self):
        while not self._detener.is_set():
            if not self.conectado:
                if not self._abrir_puerto():
                    self._detener.wait(REINTENTO_CONEXION_S)
                    continue
            try:
                linea = self._ser.readline()
            except (serial.SerialException, OSError):
                print("{}[DASHBOARD] Port lost. Reconnecting...{}"
                      .format(ANSI_AMARILLO, ANSI_RESET))
                self._cerrar_puerto()
                continue
            self._procesar(linea)
        self._cerrar_puerto()

    def _procesar(self, linea_bytes):
        """Discriminates telemetry / diagnostics / garbage."""
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
#                  GRAPHICAL DASHBOARD + MASTER AI CHAT
# =============================================================================

def construir_dashboard(buffer_posiciones, lector):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.canvas.manager.set_window_title(
        "Pneumatic Levitator — Master AI Agent")
    # Reserve bottom area for chat interface
    fig.subplots_adjust(bottom=0.24)

    estado = {"setpoint": SETPOINT_CM, "algo": None}

    ax.set_title("Pneumatic Levitator — Master AI Agent\n"
                 "(Fuzzy / MLP / Q-Learning / Deep RL)")
    ax.set_xlabel("Time (s)  [sliding window of {:.1f} s]".format(
        VENTANA_MUESTRAS * TS_S))
    ax.set_ylabel("Ball Position (cm)")
    ax.set_xlim(-VENTANA_MUESTRAS * TS_S, 0.0)
    ax.set_ylim(0.0, LONGITUD_TUBO_CM)
    ax.grid(True, linestyle=":", alpha=0.5)

    # DYNAMIC Setpoint: red dashed line
    linea_setpoint = ax.axhline(
        estado["setpoint"], color="red", linestyle="--", linewidth=1.5,
        label="Setpoint")
    (linea_bola,) = ax.plot([], [], color="blue", linewidth=1.8,
                            solid_joinstyle="round", label="Position")
    ax.legend(loc="upper left", framealpha=0.9)

    cuadro_metricas = ax.text(
        0.985, 0.97, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5f5dc",
                  edgecolor="#888888", alpha=0.92))

    # ----------------------- Chat Interface (Widgets) ---------------------
    texto_estado = fig.text(0.10, 0.145, "Master AI ready. E.g.: 'take the "
                            "ball to 15cm using fuzzy logic'",
                            fontsize=9, color="#444444")
    eje_caja = fig.add_axes([0.10, 0.05, 0.60, 0.07])
    caja = TextBox(eje_caja, "Command ", textalignment="left")
    eje_boton = fig.add_axes([0.72, 0.05, 0.12, 0.07])
    boton = Button(eje_boton, "Send")

    def enviar_comando(texto):
        """TextBox(Enter) and Button(click) handler. Never throws."""
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
                    print("{}[MASTER AI] {} -> {}{}".format(
                        ANSI_VERDE, mensaje, trama, ANSI_RESET))
                    if setpoint is not None:
                        estado["setpoint"] = setpoint
                        linea_setpoint.set_ydata([setpoint, setpoint])
                    if algo is not None:
                        estado["algo"] = algo
                    caja.set_val("")
                else:
                    texto_estado.set_text(
                        "✗ No serial connection: command not sent.")
                    texto_estado.set_color("#aa3300")
            fig.canvas.draw_idle()
        except Exception as exc:                   # Full GUI shielding
            texto_estado.set_text("✗ Internal error: {}".format(exc))
            fig.canvas.draw_idle()

    caja.on_submit(enviar_comando)
    boton.on_clicked(lambda _evento: enviar_comando(caja.text))

    # --------------------------- Animation ----------------------------------
    def actualizar(_frame):
        datos = list(buffer_posiciones)            # Atomic snapshot
        if not datos:
            cuadro_metricas.set_text(
                "WAITING FOR TELEMETRY..." if lector.conectado
                else "NO SERIAL CONNECTION")
            linea_bola.set_data([], [])
            return linea_bola, cuadro_metricas, linea_setpoint

        n = len(datos)
        eje_t = [(-(n - 1 - i)) * TS_S for i in range(n)]
        linea_bola.set_data(eje_t, datos)

        sp = estado["setpoint"]
        altura = datos[-1]
        algo_txt = (NOMBRES_ALGO[estado["algo"]]
                    if estado["algo"] else "firmware (default)")
        cuadro_metricas.set_text(
            "Height : {:6.2f} cm\n"
            "Error  : {:+6.2f} cm\n"
            "Band   : {:6.2f} cm\n"
            "AI     : {}".format(
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
        print("{}[DASHBOARD] Closed. The ESP32 continues in autonomous flight "
              "with its last configuration.{}".format(
                  ANSI_AMARILLO, ANSI_RESET))


if __name__ == "__main__":
    main()