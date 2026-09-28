from pathlib import Path
from zipfile import ZipFile
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay



BASE = Path(__file__).resolve().parent
ARCHIVO_CIRCUITO = BASE / "evaluacion" / "circuito.zip"
ARCHIVO_MODELO = BASE / "modelo" / "modelo_movimientos.joblib"


INTERVALOS = [
    # Completa aquí todos los tramos evaluables de tu circuito.
]


guardado = joblib.load(ARCHIVO_MODELO)
modelo = guardado["modelo"]
escalador = guardado["escalador"]
entradas = guardado["entradas"]

# 3. LEER LA TOMA: mismo formato de cinco columnas que en lab2.py.
with ZipFile(ARCHIVO_CIRCUITO) as archivo:
    datos = pd.read_csv(archivo.open("Raw Data.csv"))



datos.columns = ["t", "ax", "ay", "az", "magnitud"]

#mismo filtro y promedio movil del entrenamiento.
fs = 1 / np.median(np.diff(datos["t"]))

filtro = butter(N=4,
Wn=5, 
btype="lowpass", 
fs=fs, 
output="sos")

for eje in entradas:
    # Filtro de altas y bajas
    datos[eje] = sosfiltfilt(filtro, datos[eje].to_numpy())
    
    #Promedio movil
    datos[eje] = datos[eje].rolling(window=5, center=True, min_periods=5).mean()
    
datos = datos.dropna(subset=entradas).copy()


datos["prediccion"] = modelo.predict(escalador.transform(datos[entradas]))

#Etiquetado
datos["etiqueta_real"] = pd.Series(index=datos.index, dtype="object")
for inicio, fin, movimiento in INTERVALOS:
    tramo = (datos["t"] >= inicio) & (datos["t"] < fin)
    datos.loc[tramo, "etiqueta_real"] = movimiento

# Clasificación con referencial real
evaluados = datos.dropna(subset=["etiqueta_real"]).copy()
evaluados["acierto"] = evaluados["etiqueta_real"] == evaluados["prediccion"]
exactitud = accuracy_score(evaluados["etiqueta_real"], evaluados["prediccion"])
reporte = classification_report(
    evaluados["etiqueta_real"], evaluados["prediccion"], digits=3, zero_division=0
)

print(f"Muestras evaluadas: {len(evaluados)} de {len(datos)} procesadas\n")
print(f"Muestras sin etiqueta (no evaluadas): {len(datos) - len(evaluados)}\n")
print(f"Exactitud del circuito: {exactitud:.2%}\n")
print(f"Criterio de al menos 80%: {'CUMPLE' if exactitud >= 0.8 else 'NO CUMPLE'}\n\n")
print(f"{reporte}")

ConfusionMatrixDisplay.from_predictions(
    evaluados["etiqueta_real"], evaluados["prediccion"],
    labels=modelo.classes_, cmap="Blues", values_format="d"
)
plt.title("Evaluación del circuito")
plt.xlabel("Movimiento predicho")
plt.ylabel("Movimiento real")
plt.tight_layout()
plt.close()
