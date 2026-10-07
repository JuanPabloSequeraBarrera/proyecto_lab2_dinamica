import argparse
import csv
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.signal import butter, sosfiltfilt


BASE = Path(__file__).resolve().parent
ARCHIVO_MODELO = BASE / "modelo" / "modelo_movimientos.joblib"
IP_ESP32 = "172.20.10.2"
DURACION = 90

#el modelo recibe tres valores por muestra.
CONTEXTO_S = 3.0
VENTANA_VOTACION_S = 1.0
PASO_PREDICCION_S = 1.0

MARGEN_FINAL_S = 0.25
MAX_HUECO_S = 0.25

# Si los ejes del wearable difieren de los del celular, ajustar estos valores.
# Ejemplo: EJE_ORIGEN = ("ay", "ax", "az"), SIGNOS = (1, -1, 1).
# Determinar la correspondencia físicamente; no adivinarla.
EJE_ORIGEN = ("ax", "ay", "az")
SIGNOS = (1, 1, 1)


def cargar_modelo(ruta=ARCHIVO_MODELO):
    guardado = joblib.load(ruta)
    if guardado["entradas"] != ["ax", "ay", "az"]:
        raise ValueError("Este receptor espera el modelo de tres ejes de lab2.py.")
    return guardado


def clasificar_ventana(buffer, guardado):
    """Mismos filtros y escalador entrenado; votación del segundo reciente."""
    datos = pd.DataFrame(buffer, columns=["t_ms", "ax", "ay", "az"])
    datos["t"] = datos["t_ms"] / 1000.0
    intervalos = np.diff(datos["t"].to_numpy())
    if len(intervalos) < 30 or np.any(intervalos <= 0):
        return None
    fs = 1.0 / np.median(intervalos)
    if fs <= 10:
        raise ValueError("El filtro de 5 Hz necesita muestreo mayor que 10 Hz.")

    filtro = butter(N=4, Wn=5, btype="lowpass", fs=fs, output="sos")
    originales = datos[["ax", "ay", "az"]].copy()
    for eje, origen, signo in zip(guardado["entradas"], EJE_ORIGEN, SIGNOS):
        filtrado = sosfiltfilt(filtro, signo * originales[origen].to_numpy())
        datos[eje] = pd.Series(filtrado).rolling(
            window=5, center=True, min_periods=5
        ).mean()

    fin = datos["t"].iloc[-1] - MARGEN_FINAL_S
    inicio = fin - VENTANA_VOTACION_S
    ventana = datos.loc[(datos["t"] >= inicio) & (datos["t"] < fin)]
    ventana = ventana.dropna(subset=guardado["entradas"])
    if ventana.empty:
        return None

    # Usar transform, nunca volver a ajustar el escalador con datos nuevos.
    X = guardado["escalador"].transform(ventana[guardado["entradas"]])
    etiquetas = guardado["modelo"].predict(X)
    votos = pd.Series(etiquetas).value_counts()
    # Mismo desempate que grupo.mode().iloc[0] en evaluacion.py.
    movimiento = pd.Series(etiquetas).mode().iloc[0]
    acuerdo = float(votos.loc[movimiento] / len(etiquetas))
    # Acuerdo es la fracción de votos; no es probabilidad de estar acertando.
    return inicio, fin, movimiento, acuerdo, len(etiquetas)


def recibir(ip=IP_ESP32, duracion=DURACION):
    guardado = cargar_modelo()
    carpeta = BASE / "datos"
    carpeta.mkdir(exist_ok=True)
    nombre = f"toma_{time.time_ns()}"
    ruta = carpeta / f"{nombre}.csv"
    ruta_predicciones = carpeta / f"{nombre}_predicciones.csv"
    buffer = deque()
    muestras = bloques = omitidas = predicciones = 0
    ultimo_t_ms = origen_t_ms = ultima_prediccion_ms = None
    creado = False

    try:
        with requests.Session() as conexion:
            print(f"Conectando con la ESP32: {ip}...")
            respuesta = conexion.get(f"http://{ip}/", timeout=5)
            respuesta.raise_for_status()
            print("ESP32 encontrada.")
            print(respuesta.text)
            print("Usando aceleración sin gravedad en m/s².")
            print("Grabación iniciada. Ctrl+C para detener.")
            print(f"Primera predicción tras unos {CONTEXTO_S:g} s de datos continuos.")

            with ruta.open("x", newline="", encoding="utf-8") as archivo, \
                    ruta_predicciones.open("x", newline="", encoding="utf-8") as salida:
                creado = True
                escritor = csv.writer(archivo)
                escritor.writerow(["t_ms", "ax", "ay", "az", "gx", "gy", "gz"])
                resultados = csv.writer(salida)
                resultados.writerow([
                    "inicio_s", "fin_s", "movimiento", "acuerdo_votos", "muestras"
                ])
                archivo.flush()
                salida.flush()
                inicio_grabacion = time.monotonic()

                while time.monotonic() - inicio_grabacion < duracion:
                    try:
                        respuesta = conexion.get(f"http://{ip}/data", timeout=3)
                        respuesta.raise_for_status()
                    except requests.RequestException as error:
                        print("Error de comunicación:", error)
                        time.sleep(0.1)
                        continue

                    nuevas = 0
                    for linea in respuesta.text.strip().splitlines():
                        valores = linea.strip().split(",")
                        if len(valores) != 7:
                            omitidas += 1
                            continue
                        try:
                            numeros = [float(v) for v in valores]
                        except ValueError:
                            omitidas += 1
                            continue
                        if not np.all(np.isfinite(numeros)) or numeros[0] < 0:
                            omitidas += 1
                            continue
                        t_ms = numeros[0]
                        # Evitar contar muestras repetidas si /data reenvía un bloque.
                        if ultimo_t_ms is not None and t_ms <= ultimo_t_ms:
                            omitidas += 1
                            continue
                        if ultimo_t_ms is not None and t_ms - ultimo_t_ms > MAX_HUECO_S * 1000:
                            buffer.clear()
                            ultima_prediccion_ms = None
                            print("Hueco en los datos: esperando una ventana nueva.")
                        if origen_t_ms is None:
                            origen_t_ms = t_ms
                        ultimo_t_ms = t_ms
                        escritor.writerow(valores)
                        muestras += 1
                        nuevas += 1
                        buffer.append(numeros[:4])
                        # Mantener al menos tres segundos, incluido su primer punto.
                        while len(buffer) > 2 and buffer[1][0] < t_ms - CONTEXTO_S * 1000:
                            buffer.popleft()

                    archivo.flush()
                    bloques += 1
                    listo = buffer and buffer[-1][0] - buffer[0][0] >= CONTEXTO_S * 1000
                    toca = ultima_prediccion_ms is None or (
                        ultimo_t_ms - ultima_prediccion_ms >= PASO_PREDICCION_S * 1000
                    )
                    if nuevas and listo and toca:
                        resultado = clasificar_ventana(buffer, guardado)
                        if resultado is not None:
                            desde, hasta, movimiento, acuerdo, cantidad = resultado
                            desde -= origen_t_ms / 1000
                            hasta -= origen_t_ms / 1000
                            resultados.writerow([desde, hasta, movimiento, acuerdo, cantidad])
                            salida.flush()
                            predicciones += 1
                            ultima_prediccion_ms = ultimo_t_ms
                            print(
                                f"[{desde:6.2f}–{hasta:6.2f} s] "
                                f"Movimiento: {movimiento.upper():8s} | "
                                f"Votos: {acuerdo:.0%} | Muestras guardadas: {muestras}"
                            )
                    if nuevas and not listo:
                        print(f"Muestras: {muestras} | Acumulando datos para filtrar...")
                    if not nuevas:
                        time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nGrabación detenida por el usuario.")
    except requests.RequestException as error:
        print("No se pudo conectar con la ESP32:", error)
    finally:
        print(f"\nMuestras guardadas: {muestras} | Bloques recibidos: {bloques}")
        print(f"Filas omitidas o repetidas: {omitidas} | Predicciones: {predicciones}")
        if creado:
            print(f"Datos: {ruta}")
            print(f"Predicciones: {ruta_predicciones}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default=IP_ESP32)
    parser.add_argument("--duracion", type=float, default=DURACION)
    args = parser.parse_args()
    if args.duracion <= 0:
        parser.error("La duración debe ser positiva.")
    if not ARCHIVO_MODELO.is_file():
        parser.error(f"Falta el modelo: {ARCHIVO_MODELO}. Ejecuta lab2.py primero.")
    recibir(args.ip, args.duracion)
