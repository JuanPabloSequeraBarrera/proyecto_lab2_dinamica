from pathlib import Path
from zipfile import ZipFile
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay



BASE = Path(__file__).resolve().parent
ARCHIVO_CIRCUITO = BASE / "evaluacion" / "circuito_1.zip"
ARCHIVO_MODELO = BASE / "modelo" / "modelo_movimientos.joblib"
#esablezco las rutas de los archivos


INTERVALOS = [
    (5.5, 16.0, "caminata"),
    (16.0, 25.5, "parado"),
    (25.5, 35.5, "correr"),
    (35.5, 45.5, "saltar"),
    (45.5, 55.0, "parado"),
    (55.0, 65.5, "caminata"),
    (65.5, 76.0, "saltar"),
    (76.0, 90.0, "correr"),
] #intervalos sacados del video, 

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

#zona de filtrado
for eje in entradas:
    # Filtro de altas y bajas
    datos[eje] = sosfiltfilt(filtro, datos[eje].to_numpy())
    
    #Promedio movil
    datos[eje] = datos[eje].rolling(window=5, center=True, min_periods=5).mean()
    
datos = datos.dropna(subset=entradas).copy()


datos["prediccion"] = modelo.predict(escalador.transform(datos[entradas]))
# Conservo la salida orginal de mis predicciones
datos["prediccion_original"] = datos["prediccion"]

# Agrupar por segundos: [0, 1), [1, 2), etc.
datos["segundo"] = np.floor(datos["t"]).astype(int)

#Uso el valor que más se repite durante un segundo de mi toma
datos["prediccion"] = (datos.groupby("segundo")["prediccion_original"].transform(lambda grupo: grupo.mode().iloc[0]))
#en esta linea busco la etiqueta mas frecuente en cada grupo (segundo) o más conocida como la moda. y la pongo en todas las filas de esa etiqueta
#esto se llama votación. si por ejemplo en un segundo de caminata el modelo acierta 70 de 100 muestras este metodo asigna a todo ese segundo caminata.


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

print(f"muestras evaluadas: {len(evaluados)} de {len(datos)} procesadas\n")
print(f"muestras (no evaludas): {len(datos) - len(evaluados)}\n")
print(f"Exactitud del circuito: {exactitud:.2%}\n")
print(f"Criterio de al menos 80%: {'CUMPLE' if exactitud >= 0.8 else 'NO CUMPLE'}\n\n")
print(f"{reporte}")

ConfusionMatrixDisplay.from_predictions(
    evaluados["etiqueta_real"], evaluados["prediccion"],
    labels=modelo.classes_, cmap="Blues", values_format="d"
)
plt.title("Matriz de confusión del cirucito")
plt.xlabel("Movimiento predicho")
plt.ylabel("Movimiento real")
plt.tight_layout()
plt.show()

