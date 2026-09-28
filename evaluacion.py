"""Evaluar una grabación nueva después de terminar el circuito."""

from pathlib import Path
from zipfile import ZipFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# 1. CONFIGURACIÓN: cambia la ruta y completa los intervalos de tu toma.
BASE = Path(__file__).resolve().parent
ARCHIVO_CIRCUITO = BASE / "evaluacion" / "circuito.csv"
# También puedes indicar un ZIP que contenga "Raw Data.csv".
ARCHIVO_MODELO = BASE / "modelo" / "modelo_movimientos.joblib"
CARPETA_RESULTADOS = BASE / "evaluacion" / "resultados"

# (inicio en segundos, fin en segundos, movimiento real).
# Usa el tiempo del CSV y lo observado en el video; no las predicciones.
# El inicio se incluye y el fin se excluye: inicio <= t < fin.
# Ejemplo de formato: (5, 15, "caminata")
# Nombres válidos: "caminata", "correr", "parado", "saltar".
INTERVALOS = [
    # Completa aquí todos los tramos evaluables de tu circuito.
]


def main():
    if not INTERVALOS:
        raise ValueError("Completa INTERVALOS con los tiempos reales de tu circuito.")
    if not ARCHIVO_MODELO.is_file():
        raise FileNotFoundError("Ejecuta primero lab2.py para guardar el modelo y el escalador.")
    if not ARCHIVO_CIRCUITO.is_file():
        raise FileNotFoundError(f"Coloca tu toma en {ARCHIVO_CIRCUITO} o cambia ARCHIVO_CIRCUITO.")

    # 2. CARGAR EL MODELO: aquí no se vuelve a entrenar ni a ajustar el escalador.
    guardado = joblib.load(ARCHIVO_MODELO)
    modelo = guardado["modelo"]
    escalador = guardado["escalador"]
    entradas = guardado["entradas"]

    # 3. LEER LA TOMA: mismo formato de cinco columnas que en lab2.py.
    if ARCHIVO_CIRCUITO.suffix.lower() == ".zip":
        with ZipFile(ARCHIVO_CIRCUITO) as archivo:
            datos = pd.read_csv(archivo.open("Raw Data.csv"))
    else:
        datos = pd.read_csv(ARCHIVO_CIRCUITO)

    if len(datos.columns) != 5:
        raise ValueError("Se esperan cinco columnas: tiempo, ax, ay, az y magnitud, como en Raw Data.csv.")
    datos.columns = ["t", "ax", "ay", "az", "magnitud"]
    datos[["t", *entradas]] = datos[["t", *entradas]].apply(pd.to_numeric, errors="raise")
    if len(datos) < 20 or not np.isfinite(datos[["t", *entradas]].to_numpy()).all():
        raise ValueError("La toma necesita al menos 20 filas y valores de tiempo/aceleración finitos.")
    if (np.diff(datos["t"]) <= 0).any():
        raise ValueError("El tiempo del CSV debe ser estrictamente creciente.")

    # 4. PREPROCESAR: mismo filtro y promedio móvil del entrenamiento.
    # Se filtra toda la grabación, sin el recorte de 2.5 a 7.5 s de lab2.py.
    fs = 1 / np.median(np.diff(datos["t"]))
    if fs <= 10:
        raise ValueError("El filtro de 5 Hz requiere una frecuencia de muestreo mayor que 10 Hz.")
    filtro = butter(N=4, Wn=5, btype="lowpass", fs=fs, output="sos")
    for eje in entradas:
        datos[eje] = sosfiltfilt(filtro, datos[eje].to_numpy())
        datos[eje] = datos[eje].rolling(window=5, center=True, min_periods=5).mean()
    datos = datos.dropna(subset=entradas).copy()

    # 5. PREDECIR: el modelo recibe solo las aceleraciones escaladas.
    datos["prediccion"] = modelo.predict(escalador.transform(datos[entradas]))

    # 6. ETIQUETAR PARA COMPARAR: tiempos reales establecidos por ustedes.
    datos["etiqueta_real"] = pd.Series(index=datos.index, dtype="object")
    for inicio, fin, movimiento in INTERVALOS:
        if not np.isfinite([inicio, fin]).all() or inicio >= fin:
            raise ValueError(f"Intervalo inválido: {inicio}, {fin}.")
        if movimiento not in modelo.classes_:
            raise ValueError(f"Movimiento desconocido: {movimiento}. Usa {list(modelo.classes_)}.")
        tramo = (datos["t"] >= inicio) & (datos["t"] < fin)
        if not tramo.any():
            raise ValueError(f"No hay muestras para el intervalo {inicio}–{fin} s.")
        if datos.loc[tramo, "etiqueta_real"].notna().any():
            raise ValueError("Hay intervalos que se superponen. Revisa sus tiempos.")
        datos.loc[tramo, "etiqueta_real"] = movimiento

    # Solo se califican las filas con referencia real. Las demás se conservan.
    evaluados = datos.dropna(subset=["etiqueta_real"]).copy()
    evaluados["acierto"] = evaluados["etiqueta_real"] == evaluados["prediccion"]
    exactitud = accuracy_score(evaluados["etiqueta_real"], evaluados["prediccion"])
    reporte = classification_report(
        evaluados["etiqueta_real"], evaluados["prediccion"], digits=3, zero_division=0
    )
    resumen = (
        f"Muestras evaluadas: {len(evaluados)} de {len(datos)} procesadas\n"
        f"Muestras sin etiqueta (no evaluadas): {len(datos) - len(evaluados)}\n"
        f"Exactitud del circuito: {exactitud:.2%}\n"
        f"Criterio de al menos 80%: {'CUMPLE' if exactitud >= 0.8 else 'NO CUMPLE'}\n\n"
        f"{reporte}"
    )
    print(resumen)

    # 7. GUARDAR LOS RESULTADOS.
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    salida = datos[["t", "etiqueta_real", "prediccion"]].copy()
    salida["acierto"] = evaluados["acierto"].astype("boolean")
    salida.to_csv(CARPETA_RESULTADOS / "predicciones.csv", index=False)
    (CARPETA_RESULTADOS / "metricas.txt").write_text(resumen, encoding="utf-8")

    ConfusionMatrixDisplay.from_predictions(
        evaluados["etiqueta_real"], evaluados["prediccion"],
        labels=modelo.classes_, cmap="Blues", values_format="d"
    )
    plt.title("Evaluación del circuito")
    plt.xlabel("Movimiento predicho")
    plt.ylabel("Movimiento real")
    plt.tight_layout()
    plt.savefig(CARPETA_RESULTADOS / "matriz_confusion.png", dpi=150)
    plt.close()
    print("Resultados guardados en:", CARPETA_RESULTADOS)


if __name__ == "__main__":
    main()
