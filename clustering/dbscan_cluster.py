import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from collections import Counter
from utils.device_utils import get_device, prepare_for_device
device = get_device()
print(f"Usando dispositivo: {device}")


# Rutas
EMBEDDINGS_NPY = 'data/embeddings.npy'
RESULTADOS_CSV = 'data/clustering_resultados.csv'

def agrupar_rostros():
    print("Cargando embeddings...")

    if not os.path.exists(EMBEDDINGS_NPY):
        print(f"Archivo '{EMBEDDINGS_NPY}' no encontrado. Ejecuta primero la fase de embeddings.")
        return False

    embeddings_dict = np.load(EMBEDDINGS_NPY, allow_pickle=True).item()
    if not embeddings_dict:
        print("No se encontraron embeddings en el archivo.")
        return False

    X = np.array(list(embeddings_dict.values()))
    paths = list(embeddings_dict.keys())

    print(f"{len(X)} embeddings cargados. Ejecutando DBSCAN...")

    # Clustering
    dbscan = DBSCAN(eps=0.9, min_samples=2, metric='euclidean')
    labels = dbscan.fit_predict(X)

    # Resultados por imagen
    for path, label in zip(paths, labels):
        print(f"{os.path.basename(path)} → Cluster {label}")

    # Resumen
    conteo = Counter(labels)
    print("Resumen de clusters:", dict(conteo))

    # Guardar CSV
    df = pd.DataFrame({'Imagen': paths, 'Grupo': labels})
    df.to_csv(RESULTADOS_CSV, index=False)
    print(f"Resultados guardados en '{RESULTADOS_CSV}'")
    return True