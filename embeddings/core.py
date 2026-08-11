import os
import json
import torch
import numpy as np
import cv2
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

    total_exitos = 0
    total_fallos = 0

    # Iterar sobre las subcarpetas (Lotes)
    for nombre_lote in os.listdir(ruta_dataset):
        ruta_lote = os.path.join(ruta_dataset, nombre_lote)
        
        # Ignorar si no es una carpeta
        if not os.path.isdir(ruta_lote): continue
        
        print(f"\nProcesando lote: {nombre_lote}...")
        
        # Definir la etiqueta (si es la carpeta de obstruido, la etiqueta sigue siendo Trump)
        etiqueta_real = "Donald Trump" if "Trump" in nombre_lote else "Impostor"

        exitos_lote = 0
        fallos_lote = 0

        # Ignorar archivos del sistema (dejar que PIL/cv2 manejen cualquier formato)
        ARCHIVOS_IGNORAR = {'thumbs.db', 'desktop.ini', '.ds_store', '.gitkeep'}

        # Procesar cada imagen dentro del lote
        for nombre_img in os.listdir(ruta_lote):
            if nombre_img.lower() in ARCHIVOS_IGNORAR or nombre_img.startswith('.'):
                continue
            ruta_completa = os.path.join(ruta_lote, nombre_img)
            
            try:
                # Intento primario con PIL
                try:
                    img = Image.open(ruta_completa).convert('RGB')
                except Exception as e_pil:
                    # Fallback robusto con OpenCV
                    print(f"  [DEBUG] PIL no pudo leer {nombre_img} ({e_pil}). Intentando con cv2...")
                    img_cv = cv2.imread(ruta_completa)
                    if img_cv is None:
                        raise ValueError("No se pudo leer la imagen ni con PIL ni con cv2.")
                    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

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
                    exitos_lote += 1
                        
                else:
                    print(f"  [DEBUG] Rostro no detectado por MTCNN en: {nombre_img}")
                    fallos_lote += 1
                    
            except Exception as e:
                print(f"  [DEBUG] Error procesando archivo {nombre_img}: {e}")
                fallos_lote += 1

        print(f"Resumen del lote '{nombre_lote}': {exitos_lote} exitosos, {fallos_lote} fallidos.")
        total_exitos += exitos_lote
        total_fallos += fallos_lote
        
    print(f"\n--- RESUMEN FINAL ---")
    print(f"Total imágenes procesadas con éxito: {total_exitos}")
    print(f"Total imágenes con fallo: {total_fallos}")
    print(f"---------------------\n")

    # Validar que se haya generado al menos un embedding
    if total_exitos == 0:
        print("ERROR: No se generó ningún embedding. Revisa que las imágenes contengan rostros detectables.")
        return False

    # --- GUARDADO FINAL ---
    # Guardar para DBSCAN
    np.save(EMBEDDINGS_NPY, embeddings_dict_npy)
    print(f"Embeddings listos para Clustering guardados en '{EMBEDDINGS_NPY}'")

    # Guardar para KNN
    with open(HISTORICO_JSON, 'w') as f:
        json.dump(historico, f, indent=2)
    print(f"Histórico etiquetado listo para Clasificación guardado en '{HISTORICO_JSON}'")

    return total_exitos