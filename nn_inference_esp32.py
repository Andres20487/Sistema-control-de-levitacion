# ===== INICIO BLOQUE IA (generado por entrenar_ia.py — NO editar) =====
# Red MLP 2-6-1 (tanh) entrenada sobre dataset_levitador.csv
# Entradas: error (cm), delta_error (cm/s). Salida: delta_pwm (int).
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
    """Inferencia MLP en MicroPython puro. Retorna Delta PWM (int)
    saturado a +/-4000. Cero dependencias externas."""
    # 1) Recorte anti-extrapolación al rango de entrenamiento
    if error < _IA_E_MIN:
        error = _IA_E_MIN
    elif error > _IA_E_MAX:
        error = _IA_E_MAX
    if delta_error < _IA_DE_MIN:
        delta_error = _IA_DE_MIN
    elif delta_error > _IA_DE_MAX:
        delta_error = _IA_DE_MAX
    # 2) Estandarización de entradas (media 0, desviación 1)
    xe = (error - _IA_MU_E) / _IA_SD_E
    xde = (delta_error - _IA_MU_DE) / _IA_SD_DE
    # 3) Capa oculta (tanh) + capa de salida (lineal)
    acumulado = _IA_B2
    for i in range(6):
        acumulado += _IA_W2[i] * _math_ia.tanh(
            _IA_W1_E[i] * xe + _IA_W1_DE[i] * xde + _IA_B1[i])
    # 4) Des-estandarización a cuentas de PWM
    delta = acumulado * _IA_SD_Y + _IA_MU_Y
    # 5) Saturación de la autoridad de control
    if delta > _IA_DELTA_MAX:
        delta = _IA_DELTA_MAX
    elif delta < -_IA_DELTA_MAX:
        delta = -_IA_DELTA_MAX
    return int(delta)
# ====== FIN BLOQUE IA ======
