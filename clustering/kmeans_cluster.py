import os
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from collections import Counter

# Rutas
EMBEDDINGS_NPY = 'data/embeddings.npy'
RESULTADOS_CSV = 'data/kmeans_resultados.csv'
MODELO_KMEANS  = 'data/modelo_kmeans.pkl'


def agrupar_kmeans(n_clusters=2):
    """Aplica K-Means sobre los embeddings faciales generados."""
    print("Cargando embeddings para K-Means...")

    if not os.path.exists(EMBEDDINGS_NPY):
        print(f"Archivo '{EMBEDDINGS_NPY}' no encontrado. Ejecuta primero la fase de embeddings.")
        return False

    embeddings_dict_raw = np.load(EMBEDDINGS_NPY, allow_pickle=True).item()
    
    # Filtrar solo datos de entrenamiento (excluir obstruidos e impostores)
    embeddings_dict = {}
    for path, emb in embeddings_dict_raw.items():
        p = path.lower().replace('\\', '/')
        if 'obstruido' not in p and 'obstructed' not in p and 'impostor' not in p:
            embeddings_dict[path] = emb

    if not embeddings_dict:
        print("No se encontraron embeddings de entrenamiento en el archivo.")
        return False

    X = np.array(list(embeddings_dict.values()))
    paths = list(embeddings_dict.keys())

    print(f"{len(X)} embeddings cargados. Ejecutando K-Means con k={n_clusters}...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Guardar modelo entrenado
    with open(MODELO_KMEANS, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"Modelo K-Means guardado en '{MODELO_KMEANS}'")

    # Resultados por imagen
    for path, label in zip(paths, labels):
        print(f"{os.path.basename(path)} -> Cluster {label}")

    # Resumen
    conteo = Counter(labels)
    print("Resumen de clusters:", dict(conteo))

    # Guardar CSV
    df = pd.DataFrame({'Imagen': paths, 'Grupo': labels})
    df.to_csv(RESULTADOS_CSV, index=False)
    print(f"Resultados guardados en '{RESULTADOS_CSV}'")
    return True
