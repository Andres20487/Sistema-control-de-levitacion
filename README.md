# 🚀 SISTEMA DE LEVITACIÓN NEUMÁTICA AUTOMÁTICA CON AGENTE IA MAESTRO

> **Sistema ciberfísico cerrado de control de altura en tiempo real.** Una esfera levita dentro de un tubo acrílico de 38 cm impulsada por un flujo de aire modulado por PWM, gobernada por un lazo determinista a **20 Hz** sobre un **ESP32**. El sistema **multiplexa en caliente 4 algoritmos** de inteligencia artificial y control clásico —Lógica Difusa, Red Neuronal, Q-Learning y Deep RL—, supervisados por un **dashboard concurrente multi-hilo** inmune a congelamientos y un **Agente IA Maestro** que interpreta lenguaje natural y dispone de un **tutor teórico** acoplado vía LLM.

Este repositorio contiene el firmware embebido (MicroPython), la aplicación de supervisión (Python/Matplotlib), los scripts de entrenamiento de los algoritmos de IA (scikit-learn y PyTorch) y la documentación técnica completa.

---

## 📊 Características Principales

* ⚡ **Control en Tiempo Real (20 Hz):** Lazo cerrado determinista de período estricto **50 ms** ejecutado en MicroPython sobre ESP32, con planificador de *deadline absoluto* (sin deriva de jitter) y recolección de basura controlada.
* 🧠 **4 Cerebros de Control Seleccionables:** **Lógica Difusa TSK** (25 reglas), **Red Neuronal MLP** (2-6-1), **Q-Learning Tabular** (política experta 7×7 de inferencia O(1)) y **Deep RL — DQN Goal-Conditioned** (3-16-16-5), todos intercambiables en caliente por comando serial.
* 🧵 **Dashboard Multi-hilo (Cero Bloqueos):** Arquitectura concurrente nativa en Python con **aislamiento estricto de hilos** (Serial, GUI con Matplotlib y HTTP asíncrono). La gráfica jamás se congela, ni con la red caída ni con la API lenta.
* 🤖 **Agente IA Maestro (NLP Local):** Enrutador inteligente por expresiones regulares con **3 vías de procesamiento** (órdenes válidas → puerto serial, envolvente de seguridad física → rechazo local, y consultas teóricas → **Tutor Teórico vía API de OpenRouter con Llama 3.3**), más una vía extendida de **consulta técnica al estado vivo** del firmware.
* 🛡️ **Doble Envolvente de Seguridad:** Protecciones físicas embebidas en el firmware (techo/suelo + watchdog serial) **y** validación de consigna en el dashboard antes de transmitir.
* 🔁 **Paradigma ΔPWM Incremental:** Ningún cerebro emite PWM absoluto; todos calculan un incremento sobre un punto de equilibrio calibrado, linealizando la planta y cancelando la gravedad de forma implícita.

---

## 🧠 Los 4 Cerebros de Control

El firmware aloja cuatro controladores intercambiables. Todos reciben el **error** y la **derivada del error**, y devuelven un **ΔPWM** incremental sobre `PWM_BASE`.

| # | Algoritmo | Arquitectura | Entrenamiento | Inferencia | Selección |
|:-:|-----------|--------------|---------------|------------|:---------:|
| **1** | 🌫️ **Lógica Difusa TSK** | 25 reglas, 5×5 conjuntos, singletons | Sintonización manual anti-oscilación | O(25), promedio ponderado | `ALGO:1` |
| **2** | 🕸️ **Red Neuronal MLP** | 2-6-1, activación `tanh` | Clonación de comportamiento (R²=0.97) | O(1) forward-pass | `ALGO:2` |
| **3** | 📊 **Q-Learning Tabular** | Política experta 7×7 (49 estados) | RL tabular + prior PD experto | **O(1)** lookup puro | `ALGO:3` |
| **4** | 🧬 **Deep RL — DQN** | 3-16-16-5, ReLU, Goal-Conditioned | Double DQN en simulación (PyTorch) | ~384 MACs forward-pass | `ALGO:4` |

> [!NOTE]
> El **DQN (ALGO 4)** es el único cerebro *condicionado por objetivos*: recibe el setpoint como tercera entrada, por lo que una sola red sirve para cualquier altura objetivo. Los pesos entrenados se **transpilan a MicroPython puro** (sin numpy/torch) y se verifican al 100% de equivalencia antes de embeber.

---

## 📂 Estructura del Repositorio

```text
levitador-neumatico-ia/
│
├── 📁 firmware_esp32/                      # Código que corre DENTRO del ESP32 (MicroPython)
│   ├── main.py                             # Lazo 20 Hz + multiplexor de 4 cerebros + parser serial + <GET>
│   └── filtros.py                          # Driver HC-SR04 + filtro en cascada (mediana 3 → promedio 2)
│
├── 📁 dashboard_pc/                        # Aplicación de supervisión (corre en la PC)
│   ├── etapa3_dashboard.py                 # Dashboard concurrente + Agente IA Maestro + Tutor LLM
│   └── requirements.txt                    # Dependencias Python del dashboard
│
├── 📁 entrenamiento/                       # Scripts de entrenamiento (offline, en la PC)
│   ├── 📁 etapa1_fuzzy/
│   │   └── controlador_difuso.py           # Mamdani 25 reglas + cosechador de dataset CSV
│   ├── 📁 etapa2_nn/
│   │   ├── entrenar_ia.py                  # Entrena la MLP 2-6-1 (clona el difuso) + transpila
│   │   └── nn_inference_esp32.py           # Bloque MLP transpilado a MicroPython puro
│   ├── 📁 etapa4_qlearning/
│   │   ├── etapa4_qlearning.py             # Entrena Q-Learning tabular en simulación
│   │   └── qlearning_politica_experta_7x7.py  # Política PD experta 7×7 (versión de producción)
│   └── 📁 etapa5_dqn/
│       ├── entrenar_dqn.py                 # DQN Goal-Conditioned (PyTorch) + transpila + verifica
│       ├── dqn_inference_esp32.py          # Bloque DQN transpilado a MicroPython puro
│       └── _modelo_final.pt                # Checkpoint del modelo ganador (selección entre corridas)
│
├── 📁 docs/                                # Informes técnicos completos (LaTeX en Markdown)
│   ├── Informe_Tecnico_ES.md
│   └── Technical_Report_EN.md
│
└── README.md                               # Este archivo
```

---

## 🛠️ Requisitos e Instalación

### 1. Hardware Requerido

| Componente | Modelo / Especificación | Pin ESP32 | Notas |
|------------|-------------------------|:---------:|-------|
| 🧩 Microcontrolador | **ESP32 DevKit V1** (MicroPython) | — | SoC dual-core Xtensa LX6 @ 240 MHz |
| 📏 Sensor de distancia | **HC-SR04** ultrasónico (40 kHz) | `Trig=25`, `Echo=26` | Montado en la tapa superior, mirando hacia abajo |
| 🌀 Actuador | **Ventilador ductado (blower)** 12 V | — | Genera el flujo de aire sustentador |
| ⚡ Etapa de potencia | **MOSFET de canal N** (+ diodo *flyback*) | `Gate=27` | Conmuta la carga inductiva del motor |
| 🟠 Planta física | **Tubo acrílico 38 cm** + esfera ligera | — | Esfera de baja densidad (icopor/ping-pong) |
| 🔌 Alimentación | Fuente 12 V para el ventilador + USB para el ESP32 | — | Masas (GND) comunes obligatorias |

> [!WARNING]
> El periférico PWM se configura a **1 kHz**, no a frecuencias ultrasónicas. Con muchos MOSFET de propósito general, frecuencias altas (~25 kHz) **no conmutan limpiamente** la carga inductiva del ventilador, provocando caídas de tensión y acoplamiento ineficiente. Mantén `PWM_FREQ_HZ = 1000`.

> [!CAUTION]
> **No alimentes el ventilador desde el pin de 5V/3V3 del ESP32.** Usa una fuente externa de 12 V con el MOSFET; de lo contrario, la corriente de arranque del motor puede dañar el regulador de la placa.

---

### 2. Configuración del Entorno (PC)

Clona el repositorio, crea un entorno virtual de Python e instala las dependencias del dashboard:

```bash
# Clonar el repositorio
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO/dashboard_pc

# Crear entorno virtual
python -m venv venv
source venv/bin/activate          # En Windows usar: venv\Scripts\activate

# Instalar dependencias desde el archivo de requerimientos
pip install -r requirements.txt
```

> [!TIP]
> Para reentrenar los modelos de IA necesitas dependencias adicionales que **no** son requeridas para operar el dashboard: `pip install scikit-learn` (etapa 2) y `pip install torch` (etapa 5). El dashboard de supervisión solo necesita los tres paquetes del `requirements.txt`.

---

### 3. Carga del Firmware en el ESP32

El firmware corre en MicroPython. Primero **flashea el intérprete** de MicroPython en la placa (con `esptool`), luego sube los archivos del directorio `firmware_esp32/`.

```bash
# (Una sola vez) Instalar las herramientas de carga
pip install esptool mpremote

# (Una sola vez) Flashear el firmware de MicroPython — ajusta el puerto
esptool.py --chip esp32 --port COM3 erase_flash
esptool.py --chip esp32 --port COM3 write_flash -z 0x1000 ESP32_GENERIC.bin

# Subir los archivos del proyecto a la raíz del sistema de archivos del ESP32
mpremote connect COM3 fs cp firmware_esp32/filtros.py :filtros.py
mpremote connect COM3 fs cp firmware_esp32/main.py    :main.py
```

> [!IMPORTANT]
> **Calibra `PWM_BASE` antes de volar.** Es el ciclo de trabajo donde la esfera flota sola (sustentación = peso). Conéctate al REPL y ajústalo manualmente:
> ```python
> # En el REPL de MicroPython, tras detener el lazo con Ctrl+C:
> ventilador.duty_u16(38400)   # sube/baja hasta que la esfera quede suspendida
> ```
> Anota el valor donde la esfera levita estable y escríbelo en la constante `PWM_BASE` de `main.py`. **Un `PWM_BASE` mal calibrado es la causa #1 de oscilaciones** (la esfera cae dentro de la banda muerta del controlador).

---

### 4. Configuración del Tutor IA (OpenRouter)

El Tutor Teórico usa la API de **OpenRouter**. Crea una clave gratuita y pégala en el dashboard.

1. Regístrate en [https://openrouter.ai](https://openrouter.ai) y genera una API key en [https://openrouter.ai/keys](https://openrouter.ai/keys).
2. Abre `dashboard_pc/etapa3_dashboard.py` y reemplaza el *placeholder*:

```python
OPENROUTER_API_KEY = "PEGA_AQUI_TU_API_KEY"   # <-- tu clave sk-or-v1-...
```

> [!WARNING]
> **Error HTTP 404 = modelo no disponible para tu cuenta.** Los IDs de modelos gratuitos cambian con el tiempo y algunos requieren permisos. El dashboard ya implementa un **fallback en cascada** sobre varios modelos `:free`, pero si todos fallan, consulta el catálogo actualizado en [https://openrouter.ai/models](https://openrouter.ai/models) (filtro *free*) y actualiza la tupla `MODELOS_LLM` con IDs vigentes.

---

## ▶️ Ejecución y Uso

### Arranque del sistema

1. **Energiza** el ventilador (12 V) y conecta el ESP32 por USB.
2. **Cierra** cualquier IDE que ocupe el puerto serial (Thonny, monitor serie, etc.) — el dashboard necesita el puerto en exclusiva.
3. **Lanza** el dashboard indicando tu puerto:

```bash
cd dashboard_pc
python etapa3_dashboard.py COM3        # Linux/macOS: python etapa3_dashboard.py /dev/ttyUSB0
```

El firmware arranca por defecto en **modo autónomo** con el cerebro **Q-Learning (ALGO 3)**, el más robusto y probado. La gráfica empezará a mostrar la trayectoria de la esfera en vivo.

### Comandos del chat (las 3 vías del Agente Maestro)

Escribe en lenguaje natural en la caja inferior del dashboard. El enrutador NLP clasifica tu mensaje automáticamente:

| Vía | Tipo de entrada | Ejemplo | Acción |
|:---:|-----------------|---------|--------|
| 🟢 **1** | Orden válida | `Lleva la bola a 18.5 cm con q-learning` | Envía `<SET:18.5,ALGO:3>` al ESP32 |
| 🔴 **2** | Orden insegura | `Sube el setpoint a 35 cm` | **Rechazo local** (fuera de [12, 28] cm) |
| 🔵 **3** | Pregunta teórica | `¿Por qué oscila la esfera con más retardo?` | Consulta al **Tutor LLM** (panel derecho) |
| 🟣 **+** | Consulta técnica | `¿Cuál es el PWM actual?` | Envía `<GET>` y muestra el estado vivo |

> [!TIP]
> Ejemplos de órdenes que el NLP entiende: `"sube a 24"`, `"usa la red neuronal"`, `"deep rl a 15cm"`, `"algoritmo 1"`, `"pon la altura en 20 cm y cambia a difuso"`. El número del algoritmo se excluye automáticamente de la búsqueda del setpoint.

---

## 🔌 Protocolo de Comunicación Serial

Comunicación a **115200 baudios**. El firmware emite telemetría continua; la PC inyecta tramas de configuración o consulta.

| Dirección | Trama | Significado |
|:---------:|-------|-------------|
| `ESP32 → PC` | `21.34\n` | **Telemetría:** posición de la esfera (cm), cada 50 ms |
| `ESP32 → PC` | `# CFG SET=18.50 ALGO=3 ...` | **ACK** de una configuración aplicada |
| `ESP32 → PC` | `# STATUS POS=.. SET=.. PWM=.. ALGO=.. MODO=.. PROT=..` | **Respuesta** a una consulta `<GET>` |
| `PC → ESP32` | `<SET:18.5,ALGO:3>` | **Configuración:** setpoint y/o algoritmo (campos parciales válidos) |
| `PC → ESP32` | `<GET>` | **Consulta técnica:** solicita una línea `# STATUS` con el estado vivo |
| `PC → ESP32` | `1500` ó `A4` | **Legado:** ΔPWM crudo / índice de acción (activa MODO PC temporal) |

> [!NOTE]
> Las tramas `<...>` **no conmutan** el modo de control: configuran el piloto autónomo. Solo los ΔPWM crudos del protocolo legado ponen el firmware en *MODO PC*. Si la PC calla más de **200 ms**, el watchdog devuelve el control al piloto autónomo.


## 🛡️ Envolvente de Seguridad y Calibración

El sistema implementa **dos capas de protección** independientes:

1. **Validación en el dashboard (preventiva):** Una orden con setpoint fuera de **[12.0, 28.0] cm** se rechaza localmente; no se transmite nada al ESP32 ni se consumen tokens del LLM.
2. **Protecciones físicas en el firmware (reactivas), con prioridad absoluta:**

| Condición | Disparo | Acción de protección | Duración |
|-----------|:-------:|----------------------|:--------:|
| 🔺 Techo | `pos ≥ 32 cm` | `PWM = 0` (corte total) | 300 ms |
| 🔻 Suelo | `pos ≤ 6 cm` | `PWM = PWM_BASE + 5000` (rescate) | 200 ms |
| 📡 Watchdog PC | silencio > 200 ms | Retorno a modo autónomo | inmediato |

**Parámetros clave de calibración** (en `firmware_esp32/main.py`):

| Constante | Valor por defecto | Descripción |
|-----------|:-----------------:|-------------|
| `PWM_BASE` | `38400` | ⚠️ **Calibrar.** Equilibrio donde la esfera flota sola |
| `SETPOINT_MIN/MAX_CM` | `12.0 / 28.0` | Envolvente operativa comandable |
| `ALFA_ACTUADOR` | `0.7` | Suavizado del actuador (< 0.4 prohibido) |
| `ALGO_DEFECTO` | `3` | Cerebro de arranque (Q-Learning) |
| `TS_MS` | `50` | Período del lazo (20 Hz) |

---

## 🧩 Lógica del Enrutador NLP (Orden de Evaluación)

```text
        ┌──────────────────────────┐
        │   Mensaje del usuario     │
        └────────────┬─────────────┘
                     ▼
        ¿Es pregunta explícita?  ──── Sí ──▶  🔵 TUTOR LLM (OpenRouter)
        (¿? · qué/cómo/por qué ·
         explica/define/compara)
                     │ No
                     ▼
        ¿Pide estado técnico vivo? ─ Sí ──▶  🟣 ESP32 <GET> (PWM, modo, etc.)
        (pwm/estado/actual/modo)
                     │ No
                     ▼
        ¿Es orden de movimiento?  ──── No ──▶  🔵 TUTOR LLM (texto no reconocido)
                     │ Sí
                     ▼
        ¿Setpoint en [12, 28] cm? ─ No ──▶  🔴 RECHAZO LOCAL (envolvente)
                     │ Sí
                     ▼
              🟢 TRAMA <SET:..,ALGO:..> ──▶ ESP32
```

---

## 🐛 Solución de Problemas (Troubleshooting)

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| El ESP32 entra en bootloader al abrir | DTR/RTS resetean la placa | Ya mitigado: el dashboard abre con `dtr=False, rts=False`. Verifica que no haya otro programa tocando el puerto |
| La esfera oscila sin estabilizar | `PWM_BASE` mal calibrado | Recalibra `PWM_BASE` desde el REPL (causa #1 de oscilación) |
| `could not open port COMx` | Puerto ocupado o incorrecto | Cierra Thonny/monitor serie; verifica el nombre del puerto |
| El Tutor responde error 404 | Modelo LLM sin permiso/retirado | Actualiza `MODELOS_LLM` con IDs vigentes de openrouter.ai/models |
| El Tutor dice "Falta la API key" | Placeholder sin reemplazar | Pega tu clave en `OPENROUTER_API_KEY` |
| La gráfica no muestra datos | Sin telemetría | Revisa cableado, masas comunes y que el firmware esté corriendo |
| `ALGO 4` no hace nada | Stub DQN sin entrenar | Pega el bloque de `dqn_inference_esp32.py` en `main.py` |

---


* **Firmware embebido:** MicroPython sobre ESP32 DevKit V1.
* **Algoritmos de IA:** scikit-learn (MLP) y PyTorch (DQN).
* **Tutor conversacional:** API de OpenRouter (Meta Llama 3.3).

> 📚 La documentación técnica matemática completa (desarrollo de los 4 algoritmos, demostración de no-bloqueo, formulaciones en LaTeX) se encuentra en el directorio [`docs/`](docs/), en español e inglés.

---

<div align="center">

⭐ **Si este proyecto te resultó útil, considera darle una estrella en GitHub** ⭐

</div>
