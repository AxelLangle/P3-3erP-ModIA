import os
import json
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Mantenemos tus inicializaciones originales
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Rutas originales
CARPETA_IMAGENES = 'data/images/'
EMBEDDINGS_NPY = 'data/embeddings.npy'
HISTORICO_JSON = 'data/historico_embeddings.json'

def generar_embeddings(ruta_dataset='dataset_practica'):
    """
    Función modificada para procesar por lotes (recomendación del profesor).
    La estructura esperada es:
    dataset_practica/
        Donald_Trump/ -> (Tus 140 fotos)
        Donald_Trump_Obstruido/ -> (Tus 10 fotos de prueba de robustez)
        Impostor/ -> (Tus 10 fotos del doble)
    """
    print(f"Iniciando procesamiento automático por lotes desde: {ruta_dataset}...")
    
    embeddings_dict_npy = {} # Formato para clustering (.npy)
    historico = {} # Formato para clasificación (.json)

    # Si ya hay un histórico, lo cargamos
    if os.path.exists(HISTORICO_JSON):
        with open(HISTORICO_JSON, 'r') as f:
            historico = json.load(f)

    # Validar que la carpeta exista
    if not os.path.exists(ruta_dataset):
        print(f"ERROR: No se encontró la carpeta {ruta_dataset}. Asegúrate de crearla y meter tus subcarpetas ahí.")
        return False

    # Iterar sobre las subcarpetas (Lotes)
    for nombre_lote in os.listdir(ruta_dataset):
        ruta_lote = os.path.join(ruta_dataset, nombre_lote)
        
        # Ignorar si no es una carpeta
        if not os.path.isdir(ruta_lote): continue
        
        print(f"Procesando lote: {nombre_lote}...")
        
        # Definir la etiqueta (si es la carpeta de obstruido, la etiqueta sigue siendo Trump)
        etiqueta_real = "Donald Trump" if "Trump" in nombre_lote else "Impostor"

        # Procesar cada imagen dentro del lote
        for nombre_img in os.listdir(ruta_lote):
            if not nombre_img.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            ruta_completa = os.path.join(ruta_lote, nombre_img)
            
            try:
                img = Image.open(ruta_completa).convert('RGB')
                face = mtcnn(img)
                
                if face is not None:
                    with torch.no_grad():
                        # Generar el embedding de 512 dimensiones
                        emb = resnet(face.unsqueeze(0)).numpy().flatten()
                        
                        # 1. Guardar para DBSCAN (Solo requiere la ruta y el embedding)
                        embeddings_dict_npy[ruta_completa] = emb.tolist()
                        
                        # 2. Guardar para KNN (Requiere el embedding y la etiqueta)
                        historico[ruta_completa] = {
                            "embedding": emb.tolist(),
                            "nombre": etiqueta_real
                        }
                        
                else:
                    print(f"  -> Rostro no detectado en: {nombre_img}")
                    
            except Exception as e:
                print(f"  -> Error procesando {nombre_img}: {e}")

    # --- GUARDADO FINAL ---
    # Guardar para DBSCAN
    np.save(EMBEDDINGS_NPY, embeddings_dict_npy)
    print(f"Embeddings listos para Clustering guardados en '{EMBEDDINGS_NPY}'")

    # Guardar para KNN
    with open(HISTORICO_JSON, 'w') as f:
        json.dump(historico, f, indent=2)
    print(f"Histórico etiquetado listo para Clasificación guardado en '{HISTORICO_JSON}'")

    return True