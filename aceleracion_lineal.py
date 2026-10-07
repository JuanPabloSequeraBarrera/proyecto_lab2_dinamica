"""Estimación de gravedad en los ejes del sensor; necesita calibración en reposo.

Usa integración del giroscopio y corrección gradual por el acelerómetro.
No reproduce exactamente el algoritmo de aceleración lineal del teléfono.
La calibración en una sola postura aproxima los sesgos; no sustituye una
calibración completa del acelerómetro en varias orientaciones.
"""
import numpy as np
from scipy.spatial.transform import Rotation

G = 9.80665


class EstimadorLineal:
    def __init__(self, aceleraciones, giroscopios):
        a = np.asarray(aceleraciones, dtype=float)
        w = np.asarray(giroscopios, dtype=float)
        media = a.mean(axis=0)
        norma = np.linalg.norm(media)
        if not 6 < norma < 14:
            raise ValueError("Aceleración de reposo incompatible con m/s² y gravedad.")
        if np.max(a.std(axis=0)) > 0.35 or np.max(w.std(axis=0)) > 2:
            raise ValueError("Hubo movimiento en la calibración. Repite estando quieto.")
        direccion = media / norma
        self.sesgo_acel = media - G * direccion
        self.sesgo_gyro = w.mean(axis=0)
        self.orientacion = Rotation.align_vectors([[0, 0, 1]], [direccion])[0]
        self.ultimo_t = None

    def actualizar(self, t_ms, aceleracion, giroscopio):
        a = np.asarray(aceleracion, dtype=float) - self.sesgo_acel
        if self.ultimo_t is not None:
            dt = (t_ms - self.ultimo_t) / 1000.0
            if dt <= 0:
                raise ValueError("Tiempo repetido o reinicio de la ESP32.")
            if dt > 0.25:
                # Sin el gyro de ese intervalo no se puede reconstruir el giro.
                raise ValueError("Interrupción mayor de 250 ms: reinicia y calibra de nuevo.")
            gravedad_estimada = self.orientacion.inv().apply([0, 0, 1])
            omega = np.deg2rad(np.asarray(giroscopio) - self.sesgo_gyro)
            norma = np.linalg.norm(a)
            # Reducir la corrección cuando la aceleración dinámica es grande.
            peso = max(0.0, 1.0 - abs(norma - G) / (0.2 * G))
            if norma > 1e-9:
                error = np.cross(a / norma, gravedad_estimada)
                omega = omega + 0.5 * peso * error
            self.orientacion = self.orientacion * Rotation.from_rotvec(omega * dt)
        self.ultimo_t = t_ms
        gravedad = self.orientacion.inv().apply([0, 0, G])
        return a - gravedad
