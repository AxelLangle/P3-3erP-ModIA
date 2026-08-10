import numpy as np
import pickle
import os
from utils.device_utils import get_device, prepare_for_device
device = get_device()
print(f"Usando dispositivo: {device}")


def cargar_modelo(path_modelo='data/modelo_knn.pkl'):
    if not os.path.exists(path_modelo):
        print(f"Modelo no encontrado en '{path_modelo}'")
        return None
    with open(path_modelo, 'rb') as f:
        return pickle.load(f)

def predecir_identidad(embedding_nuevo, path_modelo='data/modelo_knn.pkl'):
    modelo = cargar_modelo(path_modelo)
    if modelo is None:
        return None
    nombre_predicho = modelo.predict([embedding_nuevo])[0]
    print(f"Identidad estimada: {nombre_predicho}")
    return nombre_predicho

def predecir_identidad_con_confianza(embedding, path_modelo='data/modelo_knn.pkl'):
    modelo = cargar_modelo(path_modelo)
    if modelo is None:
        return None, 0
    nombre = modelo.predict([embedding])[0]
    distancia, _ = modelo.kneighbors([embedding], n_neighbors=1)
    confianza = max(0, round(100 - distancia[0][0] * 100, 2))  # Escala ajustable
    print(f"Identidad: {nombre} | Confianza: {confianza}%")
    return nombre, confianza

def predecir_top_n(embedding, path_modelo='data/modelo_knn.pkl', n=5):
    modelo = cargar_modelo(path_modelo)
    if modelo is None:
        return []
    distancias, indices = modelo.kneighbors([embedding], n_neighbors=n)
    nombres = modelo.predict([embedding] * n)
    resultados = [(nombres[i], round(distancias[0][i], 4)) for i in range(n)]
    print("Top-N predicciones:")
    for nombre, distancia in resultados:
        print(f" - {nombre} (distancia: {distancia})")
    return resultados