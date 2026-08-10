import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import streamlit as st
from PIL import Image
from utils.device_utils import get_device, prepare_for_device
device = get_device()
print(f"Usando dispositivo: {device}")

# Importar módulos del pipeline
from embeddings.core import generar_embeddings
from clustering.dbscan_cluster import agrupar_rostros
from visualization.umap_3d import visualizar_espacio_latente
from classification.trainer_knn import entrenar_clasificador
from classification.predict_identity import predecir_identidad_con_confianza

# Nuevos imports para métricas
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Rutas
LOTE_JSON = 'data/lote_actual_embeddings.json'
HISTORICO_JSON = 'data/historico_embeddings.json'
CLUSTERING_CSV = 'data/clustering_resultados.csv'
UMAP_HTML = 'data/umap_3d.html'
MODELO_KNN = 'data/modelo_knn.pkl'
UPLOAD_FOLDER = 'data/uploads'
METRICAS_CSV = 'data/metricas_knn.csv'

# Configuración de la app
st.set_page_config(page_title="Laboratorio de Identidades", layout="wide")
st.title("Laboratorio de Identidades Faciales")

# Paso 1: Subir imágenes y generar embeddings
st.subheader("Paso 1: Subir imágenes para generar embeddings")
uploaded_images = st.file_uploader("Sube imágenes faciales", type=['jpg', 'png'], accept_multiple_files=True)

if uploaded_images:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    rutas_temporales = []

    for img_file in uploaded_images:
        ruta = os.path.join(UPLOAD_FOLDER, img_file.name)
        with open(ruta, "wb") as f:
            f.write(img_file.getbuffer())
        rutas_temporales.append(ruta)

    imagenes_para_embeddings = [open(ruta, "rb") for ruta in rutas_temporales]
    embeddings_generados = generar_embeddings(imagenes_subidas=imagenes_para_embeddings)

    for f in imagenes_para_embeddings:
        f.close()

    if embeddings_generados:
        st.success("Embeddings generados y guardados correctamente.")
    else:
        st.warning("No se detectaron rostros en las imágenes subidas.")

# Paso 2: Agrupar rostros
st.subheader("Paso 2: Agrupar rostros")
if st.button("Ejecutar clustering DBSCAN"):
    if os.path.exists(LOTE_JSON):
        agrupar_rostros()
        st.success("Clustering ejecutado correctamente.")
    else:
        st.warning("Primero debes subir imágenes para generar embeddings.")

# Paso 3: Generar visualización UMAP 3D
st.subheader("Paso 3: Visualizar espacio latente")
if st.button("Generar visualización"):
    if os.path.exists(CLUSTERING_CSV):
        visualizar_espacio_latente()
        st.success("Visualización generada correctamente.")
    else:
        st.warning("Primero debes ejecutar el clustering.")

# Paso 4: Etiquetado manual
st.subheader("Paso 4: Etiquetado de rostros")
if os.path.exists(LOTE_JSON):
    with open(LOTE_JSON, 'r') as f:
        lote = json.load(f)

    etiquetas = {}
    paths = list(lote.keys())
    cols = st.columns(5)

    for i, path in enumerate(paths):
        if os.path.exists(path):
            with cols[i % 5]:
                st.image(path, caption=os.path.basename(path), width=150)
                etiqueta = st.text_input(f"Etiqueta para {os.path.basename(path)}", key=path)
                if etiqueta:
                    etiquetas[path] = {
                        "embedding": lote[path],
                        "nombre": etiqueta
                    }
        else:
            st.warning(f"No se encontró la imagen: {path}")

    if etiquetas:
        if os.path.exists(HISTORICO_JSON):
            with open(HISTORICO_JSON, 'r') as f:
                historico = json.load(f)
        else:
            historico = {}

        historico.update(etiquetas)
        with open(HISTORICO_JSON, 'w') as f:
            json.dump(historico, f, indent=2)
        st.success("Etiquetas guardadas en el histórico.")
else:
    st.info("Primero debes subir imágenes para generar embeddings.")

# Paso 5: Entrenar clasificador KNN
st.subheader("Paso 5: Entrenar modelo KNN")
if st.button("Entrenar modelo"):
    if os.path.exists(HISTORICO_JSON):
        with open(HISTORICO_JSON, 'r') as f:
            historico = json.load(f)

        embeddings = []
        nombres = []
        for datos in historico.values():
            if "nombre" in datos and datos["nombre"] != "desconocido":
                embeddings.append(datos["embedding"])
                nombres.append(datos["nombre"])

        if len(nombres) >= 5:
            entrenar_clasificador(
                path_embeddings=None,
                etiquetas_manual=nombres,
                guardar_modelo=True,
                embeddings_directos=embeddings
            )
            st.success("Modelo KNN entrenado correctamente.")

            # Evaluación del modelo
            st.subheader("Métricas de evaluación del modelo")
            X_train, X_test, y_train, y_test = train_test_split(embeddings, nombres, test_size=0.3, random_state=42)
            modelo = joblib.load(MODELO_KNN)
            y_pred = modelo.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report)
            df_report.to_csv(METRICAS_CSV, index=True)
            st.info(f"Métricas guardadas en '{METRICAS_CSV}'")

            # Botón para descargar CSV
            with open(METRICAS_CSV, "rb") as f:
                st.download_button(
                    label="Descargar métricas en CSV",
                    data=f,
                    file_name="metricas_knn.csv",
                    mime="text/csv"
                )

            # Mostrar matriz de confusión
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Real")
            st.pyplot(fig)
        else:
            st.warning("Se necesitan al menos 5 imágenes etiquetadas para entrenar el modelo.")
    else:
        st.warning("No hay histórico de etiquetas disponible.")

# Paso 6: Clasificar nueva imagen
st.subheader("Paso 6: Clasificar nueva imagen")
uploaded_file = st.file_uploader("Sube una imagen para clasificar", type=['jpg', 'png'])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Imagen cargada", width=200)

    from facenet_pytorch import MTCNN, InceptionResnetV1
    import numpy as np
    import torch

    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            emb = resnet(face.unsqueeze(0)).numpy().flatten()

        if emb.shape == (512,):
            if os.path.exists(MODELO_KNN):
                nombre, confianza = predecir_identidad_con_confianza(emb, MODELO_KNN)
                st.success(f"Identidad estimada: **{nombre}** con {confianza:.2f}% de confianza")
            else:
                st.warning("El modelo aún no ha sido entrenado.")
        else:
            st.error("Embedding inválido. No se puede realizar la predicción.")
    else:
        st.warning("No se detectó rostro en la imagen.")

# Paso 7: Visualización UMAP 3D
st.subheader("Paso 7: Visualización interactiva")
if os.path.exists(UMAP_HTML):
    with open(UMAP_HTML, 'r', encoding='utf-8') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600)
else:
    st.info("La visualización aún no ha sido generada.")