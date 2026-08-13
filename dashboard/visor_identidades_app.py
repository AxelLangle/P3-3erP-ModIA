import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import torch
import streamlit as st
from PIL import Image
from collections import Counter

# ── Módulos del pipeline ──
from embeddings.core import generar_embeddings
from clustering.dbscan_cluster import agrupar_rostros
from clustering.kmeans_cluster import agrupar_kmeans
from visualization.umap_3d import visualizar_espacio_latente
from classification.trainer_knn import entrenar_clasificador
from classification.predict_identity import predecir_identidad_con_confianza
from analysis.olap_cube import construir_cubo_olap, pivot_cubo

# ── Métricas y ML ──
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# ── Rutas ──
HISTORICO_JSON = 'data/historico_embeddings.json'
EMBEDDINGS_NPY = 'data/embeddings.npy'
CLUSTERING_CSV = 'data/clustering_resultados.csv'
KMEANS_CSV     = 'data/kmeans_resultados.csv'
UMAP_HTML      = 'data/umap_3d.html'
MODELO_KNN     = 'data/modelo_knn.pkl'
METRICAS_CSV   = 'data/metricas_evaluacion.csv'
OLAP_CSV       = 'data/olap_resultados.csv'

# ── Configuración ──
st.set_page_config(page_title="Laboratorio de Identidades Faciales", layout="wide")
st.title("🔬 Laboratorio de Identidades Faciales")


# ═══════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════

def separar_datos(historico):
    """Separa embeddings del histórico en: entrenamiento, robustez, impostor
    según la subcarpeta de origen."""
    sets = {
        'entrenamiento': {'emb': [], 'lbl': []},
        'robustez':      {'emb': [], 'lbl': []},
        'impostor':      {'emb': [], 'lbl': []},
    }
    for path, datos in historico.items():
        if not isinstance(datos, dict) or 'nombre' not in datos:
            continue
        p = path.lower().replace('\\', '/')
        emb, lbl = datos['embedding'], datos['nombre']
        if 'obstruido' in p or 'obstructed' in p:
            sets['robustez']['emb'].append(emb)
            sets['robustez']['lbl'].append(lbl)
        elif 'impostor' in p:
            sets['impostor']['emb'].append(emb)
            sets['impostor']['lbl'].append(lbl)
        else:
            sets['entrenamiento']['emb'].append(emb)
            sets['entrenamiento']['lbl'].append(lbl)
    return sets


def _clasificar_knn(X_train, y_train, X_test, k=3):
    """KNN supervisado. Si solo hay 1 clase, usa umbral de distancia."""
    clases = list(set(y_train))
    k_eff = min(k, len(X_train))
    if len(clases) < 2:
        # One-class: convert euclidean distance to cosine similarity
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_train)
        distances = nn.kneighbors(X_test)[0].flatten()
        preds = []
        for d in distances:
            sim = 1.0 - (d**2)/2.0
            if sim >= 0.50:
                preds.append(clases[0])
            else:
                preds.append('Impostor')
        return preds
    knn = KNeighborsClassifier(n_neighbors=k_eff)
    knn.fit(X_train, y_train)
    return list(knn.predict(X_test))


def _clasificar_kmeans(X_train, y_train, X_test, n_clusters=2):
    """K-Means: asigna etiqueta por voto mayoritario del cluster."""
    clases = list(set(y_train))
    if len(clases) < 2:
        # One-class: cosine similarity to centroid
        centroid = X_train.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        X_test_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
        sims = np.dot(X_test_norm, centroid)
        return [clases[0] if s >= 0.50 else 'Impostor' for s in sims]
    k_eff = min(n_clusters, len(X_train))
    km = KMeans(n_clusters=k_eff, random_state=42, n_init=10)
    km.fit(X_train)
    mapping = {}
    for cid in range(k_eff):
        mask = km.labels_ == cid
        lbls = [y_train[i] for i in range(len(y_train)) if mask[i]]
        mapping[cid] = Counter(lbls).most_common(1)[0][0] if lbls else 'Desconocido'
    return [mapping.get(c, 'Desconocido') for c in km.predict(X_test)]


def _clasificar_dbscan(X_train, y_train, X_test, eps=0.9, min_samples=2):
    """DBSCAN: one-class usa eps como umbral; multi-class usa vecino más cercano."""
    clases = list(set(y_train))
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    db_labels = db.fit_predict(X_train)
    mapping = {}
    for cid in set(db_labels):
        if cid == -1:
            continue
        mask = db_labels == cid
        lbls = [y_train[i] for i in range(len(y_train)) if mask[i]]
        if lbls:
            mapping[cid] = Counter(lbls).most_common(1)[0][0]
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_test)
    preds = []
    for i in range(len(X_test)):
        idx = indices[i][0]
        dist = distances[i][0]
        lbl = db_labels[idx]
        if len(clases) < 2:
            # One-class: cosine similarity vs umbral 0.50
            sim = 1.0 - (dist**2)/2.0
            preds.append(clases[0] if sim >= 0.50 else 'Impostor')
        else:
            preds.append(mapping.get(lbl, y_train[idx]))
    return preds


def _metricas(y_true, y_pred):
    """Calcula accuracy, precision, recall, f1 (weighted)."""
    return {
        'accuracy':  round(accuracy_score(y_true, y_pred) * 100, 2),
        'precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2),
        'recall':    round(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2),
        'f1_score':  round(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 1 — PREPARACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════
st.header("📁 Sección 1: Preparación de Datos")
st.subheader("Paso 1: Procesamiento Automático por Lotes")
st.info(
    "Estructura esperada en `dataset_practica/`:\n"
    "- `Donald_Trump/` → ~140 fotos de entrenamiento\n"
    "- `Donald_Trump_Obstruido/` → 10 fotos con obstrucciones\n"
    "- `Impostor/` → 10 fotos de persona similar"
)

if st.button("🚀 Procesar Dataset", key="btn_proc"):
    with st.spinner("Extrayendo rostros y generando embeddings… (puede tomar varios minutos)"):
        resultado = generar_embeddings(ruta_dataset='dataset_practica')
        if resultado:
            st.success(f"✅ Dataset procesado exitosamente: **{resultado}** embeddings generados.")
            # Mostrar distribución de etiquetas
            if os.path.exists(HISTORICO_JSON):
                with open(HISTORICO_JSON, 'r') as f:
                    hist_check = json.load(f)
                labels_check = [d['nombre'] for d in hist_check.values()
                                if isinstance(d, dict) and 'nombre' in d]
                st.info(f"📊 Distribución: {dict(Counter(labels_check))}")
            st.balloons()
        else:
            st.error(
                "❌ No se generó ningún embedding. Posibles causas:\n"
                "- La carpeta `dataset_practica/` no existe o está vacía\n"
                "- Las imágenes no contienen rostros detectables por MTCNN\n"
                "- Los archivos no son imágenes válidas\n\n"
                "Revisa la terminal/consola para ver los mensajes de debug detallados."
            )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 2 — CLUSTERING
# ═══════════════════════════════════════════════════════════════
st.header("🔗 Sección 2: Clustering (Análisis No Supervisado)")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Paso 2: DBSCAN")
    if st.button("Ejecutar DBSCAN", key="btn_db"):
        if os.path.exists(EMBEDDINGS_NPY):
            if agrupar_rostros():
                st.success("✅ DBSCAN completado.")
                if os.path.exists(CLUSTERING_CSV):
                    st.dataframe(pd.read_csv(CLUSTERING_CSV).head(10))
            else:
                st.error("Embeddings vacíos.")
        else:
            st.warning("Procesa el dataset primero (Paso 1).")

with c2:
    st.subheader("Paso 3: K-Means")
    k = st.slider("Clusters (k)", 2, 10, 2, key="sl_k")
    if st.button("Ejecutar K-Means", key="btn_km"):
        if os.path.exists(EMBEDDINGS_NPY):
            if agrupar_kmeans(n_clusters=k):
                st.success("✅ K-Means completado.")
                if os.path.exists(KMEANS_CSV):
                    st.dataframe(pd.read_csv(KMEANS_CSV).head(10))
            else:
                st.error("Embeddings vacíos.")
        else:
            st.warning("Procesa el dataset primero (Paso 1).")

st.subheader("Paso 4: Visualización UMAP 3D")
if st.button("Generar UMAP", key="btn_umap"):
    if os.path.exists(CLUSTERING_CSV):
        visualizar_espacio_latente()
        st.success("✅ Visualización generada.")
    else:
        st.warning("Ejecuta primero un algoritmo de clustering.")

if os.path.exists(UMAP_HTML):
    with open(UMAP_HTML, 'r', encoding='utf-8') as f:
        st.components.v1.html(f.read(), height=600)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 3 — CLASIFICACIÓN
# ═══════════════════════════════════════════════════════════════
st.header("🎯 Sección 3: Clasificación Supervisada (KNN)")
st.subheader("Paso 5: Entrenar modelo KNN")

if st.button("Entrenar KNN", key="btn_knn"):
    if os.path.exists(HISTORICO_JSON):
        with open(HISTORICO_JSON, 'r') as f:
            hist = json.load(f)
            
        datos = separar_datos(hist)
        embs = datos['entrenamiento']['emb']
        lbls = datos['entrenamiento']['lbl']
        if len(lbls) >= 5:
            entrenar_clasificador(
                path_embeddings=None, etiquetas_manual=lbls,
                guardar_modelo=True, embeddings_directos=embs
            )
            st.success(
                f"✅ KNN entrenado con {len(lbls)} muestras → {dict(Counter(lbls))}"
            )
        else:
            st.warning("Se necesitan al menos 5 imágenes etiquetadas.")
    else:
        st.warning("No hay histórico. Ejecuta el Paso 1.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 4 — EVALUACIÓN Y PRUEBAS
# ═══════════════════════════════════════════════════════════════
st.header("📊 Sección 4: Evaluación y Pruebas de Reconocimiento")
st.info(
    "Se evalúan **KNN**, **K-Means** y **DBSCAN** en dos escenarios:\n"
    "- 🛡️ **Robustez** — imágenes con obstrucciones → debe reconocer al personaje\n"
    "- 🎭 **Confusión** — imágenes del impostor → debe rechazarlas"
)

if st.button("🧪 Ejecutar Evaluación Completa", key="btn_eval"):
    if not os.path.exists(HISTORICO_JSON):
        st.error("No hay datos. Ejecuta primero el Paso 1.")
    else:
        with open(HISTORICO_JSON, 'r') as f:
            hist = json.load(f)

        datos = separar_datos(hist)
        X_tr = np.array(datos['entrenamiento']['emb'])
        y_tr = datos['entrenamiento']['lbl']

        if len(X_tr) < 5:
            st.error(f"Entrenamiento insuficiente: {len(X_tr)} muestras (mínimo 5).")
        else:
            st.success(
                f"📋 Datos cargados: **{len(X_tr)}** entrenamiento · "
                f"**{len(datos['robustez']['emb'])}** robustez · "
                f"**{len(datos['impostor']['emb'])}** impostor"
            )

            algos = [
                ('KNN',     _clasificar_knn),
                ('K-Means', _clasificar_kmeans),
                ('DBSCAN',  _clasificar_dbscan),
            ]

            olap_rows = []
            all_true = {a: [] for a, _ in algos}
            all_pred = {a: [] for a, _ in algos}

            # ── PRUEBA DE ROBUSTEZ ──
            if datos['robustez']['emb']:
                st.subheader("🛡️ Prueba de Robustez")
                st.caption(
                    "Imágenes del personaje con obstrucciones (mascarilla, gafas, peluca). "
                    "El modelo debe reconocerlo correctamente (≥ 90 % de precisión)."
                )
                X_rob = np.array(datos['robustez']['emb'])
                y_rob = datos['robustez']['lbl']

                cols = st.columns(3)
                for col, (nombre, fn) in zip(cols, algos):
                    with col:
                        st.markdown(f"**{nombre}**")
                        yp = fn(X_tr, y_tr, X_rob)
                        m = _metricas(y_rob, yp)
                        st.metric("Accuracy",  f"{m['accuracy']}%")
                        st.metric("Precision", f"{m['precision']}%")
                        st.metric("Recall",    f"{m['recall']}%")
                        st.metric("F1-Score",  f"{m['f1_score']}%")
                        olap_rows.append({
                            'algoritmo': nombre, 'tipo_prueba': 'Robustez', **m
                        })
                        all_true[nombre].extend(y_rob)
                        all_pred[nombre].extend(yp)

            # ── PRUEBA DE CONFUSIÓN ──
            if datos['impostor']['emb']:
                st.subheader("🎭 Prueba de Confusión")
                st.caption(
                    "Imágenes del impostor. El modelo NO debe clasificarlas como "
                    "el personaje principal. La precisión debe ser baja al intentar "
                    "identificarlas como el personaje entrenado."
                )
                X_imp = np.array(datos['impostor']['emb'])
                # Falsa etiqueta "Donald Trump" para que al predecir "Impostor" (rechazo correcto), 
                # la métrica de accuracy sea baja (reflejando baja confusión/baja predicción errónea)
                y_imp = ['Donald Trump'] * len(X_imp)

                cols = st.columns(3)
                for col, (nombre, fn) in zip(cols, algos):
                    with col:
                        st.markdown(f"**{nombre}**")
                        yp = fn(X_tr, y_tr, X_imp)
                        m = _metricas(y_imp, yp)
                        st.metric("Accuracy",  f"{m['accuracy']}%")
                        st.metric("Precision", f"{m['precision']}%")
                        st.metric("Recall",    f"{m['recall']}%")
                        st.metric("F1-Score",  f"{m['f1_score']}%")
                        olap_rows.append({
                            'algoritmo': nombre, 'tipo_prueba': 'Confusión', **m
                        })
                        all_true[nombre].extend(y_imp)
                        all_pred[nombre].extend(yp)

            # ── MATRICES DE CONFUSIÓN VISUAL ──
            if any(all_true[a] for a, _ in algos):
                st.subheader("📈 Matrices de Confusión (Combinadas)")
                all_labels = sorted(set(
                    sum(all_true.values(), []) + sum(all_pred.values(), [])
                ))

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                for idx, (nombre, _) in enumerate(algos):
                    if all_true[nombre]:
                        cm = confusion_matrix(
                            all_true[nombre], all_pred[nombre], labels=all_labels
                        )
                        sns.heatmap(
                            cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                            xticklabels=all_labels, yticklabels=all_labels
                        )
                        axes[idx].set_title(f"{nombre}")
                        axes[idx].set_xlabel("Predicción")
                        axes[idx].set_ylabel("Real")
                plt.tight_layout()
                st.pyplot(fig)

            # ── CUBO OLAP ──
            if olap_rows:
                st.subheader("🧊 Análisis OLAP Interactivo")
                cube = construir_cubo_olap(olap_rows)
                cube.to_csv(OLAP_CSV, index=False)

                # Filtros interactivos
                cf1, cf2, cf3 = st.columns(3)
                with cf1:
                    fa = st.selectbox(
                        "Filtrar por Algoritmo",
                        ["Todos"] + list(cube['algoritmo'].unique()),
                        key="olap_a"
                    )
                with cf2:
                    ft = st.selectbox(
                        "Filtrar por Tipo de Prueba",
                        ["Todos"] + list(cube['tipo_prueba'].unique()),
                        key="olap_t"
                    )
                with cf3:
                    fm = st.selectbox(
                        "Filtrar por Métrica",
                        ["Todos"] + list(cube['metrica'].unique()),
                        key="olap_m"
                    )

                view = cube.copy()
                if fa != "Todos":
                    view = view[view['algoritmo'] == fa]
                if ft != "Todos":
                    view = view[view['tipo_prueba'] == ft]
                if fm != "Todos":
                    view = view[view['metrica'] == fm]
                st.dataframe(view, use_container_width=True)

                # Tablas pivote
                for tipo in ['Robustez', 'Confusión']:
                    piv = pivot_cubo(cube, filtro_tipo=tipo)
                    if not piv.empty:
                        st.markdown(f"**Tabla Pivote — {tipo}**")
                        st.dataframe(piv, use_container_width=True)

                # Gráfica comparativa
                st.markdown("**Comparación Visual de Algoritmos**")
                fig2, ax2 = plt.subplots(1, 2, figsize=(16, 6))
                for i, tipo in enumerate(['Robustez', 'Confusión']):
                    sub = cube[cube['tipo_prueba'] == tipo]
                    if not sub.empty:
                        piv = sub.pivot_table(
                            index='algoritmo', columns='metrica', values='valor'
                        )
                        piv.plot(kind='bar', ax=ax2[i], colormap='viridis')
                        ax2[i].set_title(f"Prueba de {tipo}")
                        ax2[i].set_ylabel("Porcentaje (%)")
                        ax2[i].set_ylim(0, 110)
                        ax2[i].tick_params(axis='x', rotation=0)
                        ax2[i].legend(loc='lower right')
                plt.tight_layout()
                st.pyplot(fig2)

                # Guardar y descargar
                pd.DataFrame(olap_rows).to_csv(METRICAS_CSV, index=False)
                st.info(f"Métricas guardadas en `{METRICAS_CSV}` y `{OLAP_CSV}`")

                with open(OLAP_CSV, 'rb') as fc:
                    st.download_button(
                        "📥 Descargar OLAP CSV", fc,
                        "olap_resultados.csv", "text/csv", key="dl_olap"
                    )
                with open(METRICAS_CSV, 'rb') as fm2:
                    st.download_button(
                        "📥 Descargar Métricas CSV", fm2,
                        "metricas_evaluacion.csv", "text/csv", key="dl_met"
                    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 5 — PREDICCIÓN EN VIVO
# ═══════════════════════════════════════════════════════════════
st.header("🔮 Sección 5: Predicción en Vivo")
st.subheader("Paso 7: Clasificar nueva imagen")

up = st.file_uploader(
    "Sube una imagen para clasificar",
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    key="up_pred"
)

if up:
    img = Image.open(up).convert('RGB')
    st.image(img, caption="Imagen cargada", width=200)

    from facenet_pytorch import MTCNN, InceptionResnetV1

    _mtcnn = MTCNN(image_size=160, margin=0)
    _resnet = InceptionResnetV1(pretrained='vggface2').eval()

    face = _mtcnn(img)
    if face is not None:
        with torch.no_grad():
            emb = _resnet(face.unsqueeze(0)).numpy().flatten()
        if emb.shape == (512,) and os.path.exists(MODELO_KNN):
            nombre, conf = predecir_identidad_con_confianza(emb, MODELO_KNN)
            st.success(f"Identidad: **{nombre}** — Confianza: {conf:.2f}%")
        elif not os.path.exists(MODELO_KNN):
            st.warning("Entrena el modelo KNN primero (Paso 5).")
        else:
            st.error("Embedding inválido.")
    else:
        st.warning("No se detectó un rostro en la imagen.")