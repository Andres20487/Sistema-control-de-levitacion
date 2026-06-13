# =============================================================================
# entrenar_ia.py — Levitador Neumático | Etapa 2 (PC)
# -----------------------------------------------------------------------------
# Entrena una Red Neuronal ultra-ligera (MLPRegressor de scikit-learn) que
# CLONA al controlador difuso a partir del dataset cosechado en la Etapa 1, y
# TRANSPILA el modelo a una función de MicroPython puro:
#
#     calcular_pwm_ia(error, delta_error) -> delta_pwm (int)
#
# con pesos, sesgos y constantes de normalización HARDCODEADOS: cero
# dependencias en el ESP32 (solo `math`, que es nativo de MicroPython).
#
# Pipeline:
#   1. Carga dataset_levitador.csv  (error, delta_error, delta_pwm).
#   2. Estandariza entradas y salida (StandardScaler: media 0, desviación 1).
#   3. Entrena MLP de 1 capa oculta (defecto: 6 neuronas, tanh, solver LBFGS).
#   4. Reporta MAE y R² sobre un conjunto de prueba (20%).
#   5. Genera el bloque MicroPython y VERIFICA numéricamente que la función
#      generada reproduce a sklearn (diferencia máxima reportada).
#   6. Imprime el bloque en consola y lo guarda en nn_inference_esp32.py.
#
# El bloque generado se pega en esp32/main.py reemplazando el stub marcado
# entre "INICIO BLOQUE IA" y "FIN BLOQUE IA".
#
# Uso:
#   python entrenar_ia.py
#   python entrenar_ia.py --csv ../dataset_levitador.csv --neuronas 8
#
# Dependencias (solo PC): pip install scikit-learn numpy
# =============================================================================

import argparse
import math
import os
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ------------------------------- Parámetros ----------------------------------
DELTA_PWM_MAX = 4000          # Autoridad máxima de la IA (igual que el difuso)
ARCHIVO_SALIDA = "nn_inference_esp32.py"
MARGEN_CLIP = 1.0             # Holgura sobre el rango de entrenamiento para
                              # el recorte anti-extrapolación de las entradas


# =============================================================================
#                              CARGA DEL DATASET
# =============================================================================

def cargar_dataset(ruta_csv):
    if not os.path.exists(ruta_csv):
        print("ERROR: no se encontró '{}'.".format(ruta_csv))
        print("Ejecuta primero la Etapa 1 para cosechar el dataset.")
        sys.exit(1)

    datos = np.genfromtxt(ruta_csv, delimiter=",", skip_header=1)
    datos = datos[~np.isnan(datos).any(axis=1)]   # Filas corruptas: fuera
    if datos.shape[0] < 100 or datos.shape[1] != 3:
        print("ERROR: dataset inválido ({} filas, {} columnas).".format(
            datos.shape[0], datos.shape[1] if datos.ndim > 1 else 0))
        sys.exit(1)

    X = datos[:, 0:2]                              # error, delta_error
    y = datos[:, 2:3]                              # delta_pwm
    print("[DATOS] {} muestras válidas cargadas de '{}'.".format(
        X.shape[0], ruta_csv))
    print("[DATOS] error:       [{:+8.2f}, {:+8.2f}] cm".format(
        X[:, 0].min(), X[:, 0].max()))
    print("[DATOS] delta_error: [{:+8.2f}, {:+8.2f}] cm/s".format(
        X[:, 1].min(), X[:, 1].max()))
    print("[DATOS] delta_pwm:   [{:+8.0f}, {:+8.0f}] cuentas".format(
        y.min(), y.max()))
    return X, y


# =============================================================================
#                               ENTRENAMIENTO
# =============================================================================

def entrenar(X, y, neuronas, semilla):
    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X, y, test_size=0.20, random_state=semilla, shuffle=True)

    # Estandarización: media 0, desviación 1 (entradas Y salida).
    esc_x = StandardScaler().fit(X_ent)
    esc_y = StandardScaler().fit(y_ent)

    Xe = esc_x.transform(X_ent)
    ye = esc_y.transform(y_ent).ravel()

    print("\n[RED] Entrenando MLP 2-{}-1 (tanh, LBFGS)...".format(neuronas))
    mlp = MLPRegressor(
        hidden_layer_sizes=(neuronas,),
        activation="tanh",
        solver="lbfgs",            # Ideal para redes diminutas: converge fino
        alpha=1e-4,                # Regularización L2 leve
        max_iter=8000,
        tol=1e-7,
        random_state=semilla,
    )
    mlp.fit(Xe, ye)

    # ---- Evaluación en unidades físicas reales (cuentas de PWM) ----
    def predecir_fisico(X_crudo):
        y_norm = mlp.predict(esc_x.transform(X_crudo)).reshape(-1, 1)
        return esc_y.inverse_transform(y_norm).ravel()

    pred_pru = predecir_fisico(X_pru)
    mae = mean_absolute_error(y_pru.ravel(), pred_pru)
    r2 = r2_score(y_pru.ravel(), pred_pru)
    print("[RED] Evaluación sobre el 20% de prueba ({} muestras):".format(
        len(y_pru)))
    print("[RED]   MAE = {:.1f} cuentas de PWM  (escala: ±{})".format(
        mae, DELTA_PWM_MAX))
    print("[RED]   R²  = {:.4f}".format(r2))
    if r2 < 0.90:
        print("[RED] AVISO: R² < 0.90. Considera más neuronas (--neuronas 8) "
              "o revisar la riqueza del dataset.")

    return mlp, esc_x, esc_y, predecir_fisico, (X_pru, y_pru)


# =============================================================================
#                  TRANSPILACIÓN A MICROPYTHON PURO (sin librerías)
# =============================================================================

def _fmt(valor):
    """Formato compacto y de precisión completa para constantes float."""
    return "{:.10g}".format(float(valor))


def _fmt_tupla(vector):
    return "(" + ", ".join(_fmt(v) for v in vector) + ")"


def generar_micropython(mlp, esc_x, esc_y, X, neuronas):
    """Construye el bloque MicroPython con la matemática completa de la red:
    recorte de entradas -> estandarización -> capa oculta tanh -> capa de
    salida lineal -> des-estandarización -> saturación ±DELTA_PWM_MAX."""

    W1 = mlp.coefs_[0]              # forma (2, H)
    b1 = mlp.intercepts_[0]         # forma (H,)
    W2 = mlp.coefs_[1][:, 0]        # forma (H,)
    b2 = float(mlp.intercepts_[1][0])

    mu_e, mu_de = esc_x.mean_
    sd_e, sd_de = esc_x.scale_
    mu_y = float(esc_y.mean_[0])
    sd_y = float(esc_y.scale_[0])

    # Recorte anti-extrapolación: la red solo conoce el rango entrenado.
    e_min = float(X[:, 0].min()) - MARGEN_CLIP
    e_max = float(X[:, 0].max()) + MARGEN_CLIP
    de_min = float(X[:, 1].min()) - MARGEN_CLIP
    de_max = float(X[:, 1].max()) + MARGEN_CLIP

    lineas = []
    a = lineas.append
    a("# ===== INICIO BLOQUE IA (generado por entrenar_ia.py — NO editar) =====")
    a("# Red MLP 2-{}-1 (tanh) entrenada sobre dataset_levitador.csv".format(
        neuronas))
    a("# Entradas: error (cm), delta_error (cm/s). Salida: delta_pwm (int).")
    a("import math as _math_ia")
    a("")
    a("_IA_MU_E = {}".format(_fmt(mu_e)))
    a("_IA_SD_E = {}".format(_fmt(sd_e)))
    a("_IA_MU_DE = {}".format(_fmt(mu_de)))
    a("_IA_SD_DE = {}".format(_fmt(sd_de)))
    a("_IA_MU_Y = {}".format(_fmt(mu_y)))
    a("_IA_SD_Y = {}".format(_fmt(sd_y)))
    a("_IA_W1_E = {}".format(_fmt_tupla(W1[0, :])))
    a("_IA_W1_DE = {}".format(_fmt_tupla(W1[1, :])))
    a("_IA_B1 = {}".format(_fmt_tupla(b1)))
    a("_IA_W2 = {}".format(_fmt_tupla(W2)))
    a("_IA_B2 = {}".format(_fmt(b2)))
    a("_IA_E_MIN = {}".format(_fmt(e_min)))
    a("_IA_E_MAX = {}".format(_fmt(e_max)))
    a("_IA_DE_MIN = {}".format(_fmt(de_min)))
    a("_IA_DE_MAX = {}".format(_fmt(de_max)))
    a("_IA_DELTA_MAX = {}".format(int(DELTA_PWM_MAX)))
    a("")
    a("")
    a("def calcular_pwm_ia(error, delta_error):")
    a("    \"\"\"Inferencia MLP en MicroPython puro. Retorna Delta PWM (int)")
    a("    saturado a +/-{}. Cero dependencias externas.\"\"\"".format(
        int(DELTA_PWM_MAX)))
    a("    # 1) Recorte anti-extrapolación al rango de entrenamiento")
    a("    if error < _IA_E_MIN:")
    a("        error = _IA_E_MIN")
    a("    elif error > _IA_E_MAX:")
    a("        error = _IA_E_MAX")
    a("    if delta_error < _IA_DE_MIN:")
    a("        delta_error = _IA_DE_MIN")
    a("    elif delta_error > _IA_DE_MAX:")
    a("        delta_error = _IA_DE_MAX")
    a("    # 2) Estandarización de entradas (media 0, desviación 1)")
    a("    xe = (error - _IA_MU_E) / _IA_SD_E")
    a("    xde = (delta_error - _IA_MU_DE) / _IA_SD_DE")
    a("    # 3) Capa oculta (tanh) + capa de salida (lineal)")
    a("    acumulado = _IA_B2")
    a("    for i in range({}):".format(neuronas))
    a("        acumulado += _IA_W2[i] * _math_ia.tanh(")
    a("            _IA_W1_E[i] * xe + _IA_W1_DE[i] * xde + _IA_B1[i])")
    a("    # 4) Des-estandarización a cuentas de PWM")
    a("    delta = acumulado * _IA_SD_Y + _IA_MU_Y")
    a("    # 5) Saturación de la autoridad de control")
    a("    if delta > _IA_DELTA_MAX:")
    a("        delta = _IA_DELTA_MAX")
    a("    elif delta < -_IA_DELTA_MAX:")
    a("        delta = -_IA_DELTA_MAX")
    a("    return int(delta)")
    a("# ====== FIN BLOQUE IA ======")
    return "\n".join(lineas)


# =============================================================================
#               VERIFICACIÓN NUMÉRICA: MicroPython vs scikit-learn
# =============================================================================

def verificar_equivalencia(codigo, predecir_fisico, X_pru):
    """Ejecuta la función generada en CPython y la compara contra el modelo
    sklearn original sobre el conjunto de prueba completo."""
    entorno = {}
    exec(codigo, entorno)                          # Compila el bloque generado
    fn = entorno["calcular_pwm_ia"]

    ref = predecir_fisico(X_pru)
    ref = np.clip(ref, -DELTA_PWM_MAX, DELTA_PWM_MAX)
    gen = np.array([fn(float(e), float(de)) for e, de in X_pru])
    dif_max = float(np.max(np.abs(gen - ref)))
    print("[VERIFICACIÓN] Diferencia máxima |MicroPython - sklearn| sobre "
          "{} muestras: {:.3f} cuentas".format(len(X_pru), dif_max))
    if dif_max > 1.5:
        print("[VERIFICACIÓN] ERROR: la transpilación no es equivalente.")
        sys.exit(1)
    print("[VERIFICACIÓN] Transpilación EQUIVALENTE (solo redondeo a int).")


# =============================================================================
#                                    MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Etapa 2: entrena la MLP y genera calcular_pwm_ia() "
                    "en MicroPython puro")
    parser.add_argument("--csv", default="dataset_levitador.csv",
                        help="Ruta al dataset (defecto: dataset_levitador.csv)")
    parser.add_argument("--neuronas", type=int, default=6,
                        help="Neuronas de la capa oculta, 4 a 8 (defecto: 6)")
    parser.add_argument("--semilla", type=int, default=42,
                        help="Semilla de reproducibilidad (defecto: 42)")
    args = parser.parse_args()

    if not (4 <= args.neuronas <= 8):
        print("ERROR: --neuronas debe estar entre 4 y 8 (red ultra-ligera "
              "para el ESP32).")
        sys.exit(1)

    X, y = cargar_dataset(args.csv)
    mlp, esc_x, esc_y, predecir_fisico, (X_pru, y_pru) = entrenar(
        X, y, args.neuronas, args.semilla)

    codigo = generar_micropython(mlp, esc_x, esc_y, X, args.neuronas)
    verificar_equivalencia(codigo, predecir_fisico, X_pru)

    # ---- Persistencia + impresión en consola ----
    with open(ARCHIVO_SALIDA, "w") as f:
        f.write(codigo + "\n")
    print("\n[SALIDA] Bloque guardado en '{}'.".format(ARCHIVO_SALIDA))
    print("[SALIDA] Pega TODO el bloque siguiente en esp32/main.py, "
          "reemplazando el stub\n[SALIDA] marcado entre 'INICIO BLOQUE IA' "
          "y 'FIN BLOQUE IA':\n")
    print("=" * 79)
    print(codigo)
    print("=" * 79)

    # Sanidad física rápida de la red ya transpilada.
    entorno = {}
    exec(codigo, entorno)
    fn = entorno["calcular_pwm_ia"]
    print("\n[SANIDAD] calcular_pwm_ia(-10, 0) = {:+d}  (esfera abajo -> "
          "empuje +)".format(fn(-10.0, 0.0)))
    print("[SANIDAD] calcular_pwm_ia(  0, 0) = {:+d}  (en setpoint -> "
          "~neutro)".format(fn(0.0, 0.0)))
    print("[SANIDAD] calcular_pwm_ia(  5, 0) = {:+d}  (esfera arriba -> "
          "recorte -)".format(fn(5.0, 0.0)))


if __name__ == "__main__":
    main()