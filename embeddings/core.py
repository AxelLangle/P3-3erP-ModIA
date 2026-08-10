import os
import json
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils.device_utils import get_device, prepare_for_device
device = get_device()
print(f"Usando dispositivo: {device}")


# Inicialización de modelos
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Rutas
CARPETA_IMAGENES = 'data/images/'
EMBEDDINGS_NPY = 'data/embeddings.npy'
LOTE_JSON = 'data/lote_actual_embeddings.json'
HISTORICO_JSON = 'data/historico_embeddings.json'
OUTLIERS_JSON = 'data/outliers_embeddings.json'

def generar_embeddings(imagenes_subidas=None):
    """
    Si se proporcionan imágenes subidas (modo automático), se procesan directamente.
    Si no, se procesan las imágenes en la carpeta por defecto (modo manual).
    """
    embeddings_dict = {}

    if imagenes_subidas:
        print("Procesando imágenes subidas por el usuario...")
        for img_file in imagenes_subidas:
            try:
                # Obtener ruta temporal si existe
                ruta = getattr(img_file, 'name', None)
                if not ruta:
                    continue

                img = Image.open(img_file).convert('RGB')
                face = mtcnn(img)
                if face is not None:
                    with torch.no_grad():
                        emb = resnet(face.unsqueeze(0)).numpy().flatten()
                        embeddings_dict[ruta] = emb.tolist()
                else:
                    print(f"Rostro no detectado: {ruta}")
            except Exception as e:
                print(f"Error procesando {ruta}: {e}")
    else:
        print("Procesando imágenes desde carpeta por defecto...")
        image_paths = [
            os.path.join(CARPETA_IMAGENES, f)
            for f in os.listdir(CARPETA_IMAGENES)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                face = mtcnn(img)
                if face is not None:
                    with torch.no_grad():
                        emb = resnet(face.unsqueeze(0)).numpy().flatten()
                        embeddings_dict[path] = emb.tolist()
                else:
                    print(f"Rostro no detectado: {os.path.basename(path)}")
            except Exception as e:
                print(f"Error procesando {path}: {e}")

    # Guardar .npy
    np.save(EMBEDDINGS_NPY, embeddings_dict)
    print(f"Embeddings guardados en '{EMBEDDINGS_NPY}'")

    # Guardar lote actual en JSON
    with open(LOTE_JSON, 'w') as f:
        json.dump(embeddings_dict, f)
    print(f"Lote actual guardado en '{LOTE_JSON}'")

    # Actualizar histórico
    historico = {}
    if os.path.exists(HISTORICO_JSON):
        with open(HISTORICO_JSON, 'r') as f:
            historico = json.load(f)
    historico.update(embeddings_dict)
    with open(HISTORICO_JSON, 'w') as f:
        json.dump(historico, f)
    print(f"Histórico actualizado en '{HISTORICO_JSON}'")

    # Inicializar outliers si no existe
    if not os.path.exists(OUTLIERS_JSON):
        with open(OUTLIERS_JSON, 'w') as f:
            json.dump({}, f)
        print(f"Archivo de outliers inicializado en '{OUTLIERS_JSON}'")

    return embeddings_dict