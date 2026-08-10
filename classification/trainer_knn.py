import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from utils.device_utils import get_device, prepare_for_device
device = get_device()
print(f"Usando dispositivo: {device}")


MODELO_PATH = 'data/modelo_knn.pkl'

def entrenar_clasificador(path_embeddings='data/embeddings.npy',
                          etiquetas_manual=None,
                          guardar_modelo=False,
                          embeddings_directos=None):

    # Cargar embeddings
    if embeddings_directos is not None:
        embeddings = np.array(embeddings_directos)
    else:
        if not os.path.exists(path_embeddings):
            print(f"Archivo '{path_embeddings}' no encontrado.")
            return None
        embeddings_dict = np.load(path_embeddings, allow_pickle=True).item()
        embeddings = np.array(list(embeddings_dict.values()))

    # Validar etiquetas
    if etiquetas_manual is None or len(etiquetas_manual) != len(embeddings):
        print("Se requiere una lista de etiquetas del mismo tamaño que los embeddings.")
        return None

    print(f"Entrenando clasificador KNN con {len(embeddings)} muestras...")
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(embeddings, etiquetas_manual)

    if guardar_modelo:
        with open(MODELO_PATH, 'wb') as f:
            pickle.dump(clf, f)
        print(f"Modelo guardado en '{MODELO_PATH}'")

    return clf