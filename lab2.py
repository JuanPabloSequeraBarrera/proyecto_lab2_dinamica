import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from zipfile import ZipFile
from scipy.signal import butter, sosfiltfilt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Carpeta data 
carpeta_data = Path(__file__).resolve().parent / "data"
movimientos = ["caminata", "correr", "parado", "saltar"]

tablas = []

for movimiento in movimientos:
    
    archivos = sorted((carpeta_data / movimiento).glob("*.zip"))
    
    print(f"{movimiento}: {len(archivos)} archivos")
    
    # Leer cada toma
    for ruta in archivos:
        
        # CARGA DE DATOS
        with ZipFile(ruta) as archivo:
            datos = pd.read_csv(archivo.open("Raw Data.csv"))
        
        datos.columns = ["t", "ax", "ay", "az", "magnitud"]
        
        # Frecuencia de muestreo
        fs = 1 / np.median(np.diff(datos["t"]))
        
        # Filtro pasa bajas
        filtro = butter(
            N=4,
            Wn=5,
            btype="lowpass",
            fs=fs,
            output="sos"
        )
        
        # Filtrar y suavizar los tres ejes
        for eje in ["ax", "ay", "az"]:
            
            # Filtro de altas y bajas
            datos[eje] = sosfiltfilt(filtro,datos[eje].to_numpy())
            
            # Promedio móvil
            datos[eje] = (datos[eje].rolling(window=5,center=True,min_periods=5).mean())
        
        # Filtro de tiempo 
        datos = datos[(datos["t"] >= 2.5) & (datos["t"] < 7.5)].copy()
        
        # Eliminar NaN producidos por promedio móvil
        datos = datos.dropna(subset=["ax", "ay", "az"])
        
        # Magnitud después de suavizar los datos
        datos["magnitud"] = np.sqrt(
            datos["ax"]**2 +
            datos["ay"]**2 +
            datos["az"]**2
        )
        
        # Etiquetado
        datos["etiqueta"] = movimiento
        
        # Identificador de la toma
        datos["registro"] = movimiento + "/" + ruta.stem
        
        # Guardar esta toma
        tablas.append(datos)


# Unir todas las tomas
dataset = pd.concat(tablas,ignore_index=True)

print("\nSeries procesadas por movimiento:")

print(dataset.groupby("etiqueta")["registro"].nunique())

dataset.to_csv(
    carpeta_data / "datos_procesados.csv",
    index=False
)

#SEPARAR LAS SERIES 

#Tengo una fila por cada toma con su etiqueta
series = dataset[["registro", "etiqueta"]].drop_duplicates()

#20% de las tomas para la prueba 
series_resto, series_test = train_test_split(
    series,
    test_size=0.20,
    stratify=series["etiqueta"],
    random_state=42
)

#20% de las tomas para la validacion 
series_train, series_val = train_test_split(
    series_resto,
    test_size=0.25,
    stratify=series_resto["etiqueta"],
    random_state=42
)

train = dataset[dataset["registro"].isin(series_train["registro"])].copy()

val = dataset[dataset["registro"].isin(series_val["registro"])].copy()

test = dataset[dataset["registro"].isin(series_test["registro"])].copy()

print("Series de entrenamiento:", train["registro"].nunique())
print("Series de validación:", val["registro"].nunique())
print("Series de prueba:", test["registro"].nunique())

entradas = ["ax", "ay", "az"]

x_train = train[entradas]
y_train = train["etiqueta"]

x_val = val[entradas]
y_val = val["etiqueta"]

x_test = test[entradas]
y_test = test["etiqueta"]

escalador = StandardScaler()

X_train_escalado = escalador.fit_transform(x_train)

X_val_escalado = escalador.transform(x_val)
X_test_escalado = escalador.transform(x_test)

