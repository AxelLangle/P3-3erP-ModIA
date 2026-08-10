import os
import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP
from utils.device_utils import get_device, prepare_for_device
device = get_device()
print(f"Usando dispositivo: {device}")


# Rutas
EMBEDDINGS_NPY = 'data/embeddings.npy'
CLUSTERING_CSV = 'data/clustering_resultados.csv'
UMAP_CSV = 'data/umap_3d_resultados.csv'
UMAP_HTML = 'data/umap_3d.html'

def visualizar_espacio_latente():
    print("Cargando embeddings...")

    if not os.path.exists(EMBEDDINGS_NPY):
        print(f"Archivo '{EMBEDDINGS_NPY}' no encontrado.")
        return

    embeddings_dict = np.load(EMBEDDINGS_NPY, allow_pickle=True).item()
    embeddings = np.array(list(embeddings_dict.values()))
    file_names = [os.path.basename(f) for f in embeddings_dict.keys()]

    print("Cargando etiquetas de clustering...")

    if not os.path.exists(CLUSTERING_CSV):
        print(f"Archivo '{CLUSTERING_CSV}' no encontrado. Se usarán etiquetas vacías.")
        labels = ['Sin etiqueta'] * len(file_names)
    else:
        df_clusters = pd.read_csv(CLUSTERING_CSV)
        cluster_map = dict(zip(df_clusters['Imagen'].apply(os.path.basename), df_clusters['Grupo']))
        # Convertir etiquetas a cadenas para que Plotly las trate como categorías
        labels = [str(cluster_map.get(name, 'Sin etiqueta')) for name in file_names]

    print("Reduciendo dimensiones con UMAP...")

    reducer = UMAP(n_components=3, random_state=42)
    emb_3d = reducer.fit_transform(embeddings)

    print("Graficando espacio latente en 3D...")

    fig = px.scatter_3d(
        x=emb_3d[:, 0], y=emb_3d[:, 1], z=emb_3d[:, 2],
        color=labels, text=file_names,
        title='Espacio Latente Facial',
        labels={'x': 'UMAP-1', 'y': 'UMAP-2', 'z': 'UMAP-3'},
        color_discrete_sequence=px.colors.qualitative.Vivid  # Paleta más contrastante
    )

    fig.write_html(UMAP_HTML)
    print(f"Visualización guardada en '{UMAP_HTML}'")
    fig.show()

    # Guardar coordenadas y etiquetas
    df_umap = pd.DataFrame({
        'Imagen': file_names,
        'UMAP-1': emb_3d[:, 0],
        'UMAP-2': emb_3d[:, 1],
        'UMAP-3': emb_3d[:, 2],
        'Grupo': labels
    })
    df_umap.to_csv(UMAP_CSV, index=False)
    print(f"Coordenadas UMAP guardadas en '{UMAP_CSV}'")

    return emb_3d