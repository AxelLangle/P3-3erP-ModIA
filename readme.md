#Laboratorio de Reconocimiento Facial:

Este laboratorio funciona para el análisis, visualización y predicción de identidades faciales mediante inteligencia artificial. Este proyecto permite generar embeddings, agrupar rostros, visualizar espacios latentes y realizar predicciones sobre nuevas imágenes.

#Instalación:

Este laboratorio funciona con python 3.10.X
Para el uso de la GPU instala (verifica la plataforma adecuada para tu tarjeta gráfica antes de continuar, esta versión es la 12.1, la última compatible de pytorch con CUDA):

	pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Usa Visual Studio Code para la ejecución del entorno virtual
Procura instalar la extensión Python de Microsoft para VSCode

1. Crear un entorno virtual:

Instala la versión mencionada de Python y agrega al sistema su variable global.
Crea en tu gestor de arhivos una carpeta donde se almacenará la estructura de este laboratorio.
Abre la carpeta que creaste en VSCode.
Abre una terminal cuta ruta de ejecución pertenezca a la carpeta creada.
En caso de ser necesario, actualiza pip:
	
	python.exe -m pip install --upgrade pip  

Fuerza la creación del entorno virtual con esta versión de Python:

	py -3.10 -m venv nombre_de_tu_entorno

Instala los requerimientos necesarios desde el archivo requirements.txt:

	pip install -r requirements.txt
	
#Ejecución del modo manual:
Creación de embeddings
Genera vectores numéricos (embeddings) a partir de las imágenes faciales ubicadas en data/images. Estos vectores representan características únicas de cada rostro y se almacenan en formato JSON y .npy.

	python main.py --fase embeddings
	
Clasificación por cluster
Agrupa los rostros según similitud usando el algoritmo DBSCAN.
Esto permite identificar grupos de personas similares y detectar posibles outliers.
	
	python main.py --fase clustering
	
Visualización Latente
Este módulo aplica UMAP para proyectar los embeddings faciales en un espacio 3D visual. Permite auditar agrupamientos y detectar patrones visuales en los vectores de rostro.
	
	python main.py --fase visualization
	
#Ejecución del modo automático:
Dashboard interactivo
Permite subir imágenes, generar embeddings automáticamente, etiquetar rostros, entrenar el modelo y realizar predicciones desde una interfaz visual. En la terminal ejecuta el siguiente comando:

	streamlit run dashboard/visor_identidades_app.py

#Estructura del Proyecto
 
nombre_de_tu_carpeta/
├── data/                   # Carpeta de imágenes, embeddings, modelos y resultados
├── embeddings/             # Generación de embeddings faciales
├── clustering/             # Agrupamiento de rostros
├── visualization/          # Visualización UMAP 3D
├── classification/         # Entrenamiento y predicción
├── dashboard/              # Interfaz gráfica con Streamlit
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación del laboratorio

#Contribuciones
Este laboratorio está diseñado para la práctica de minería de datos de los estudiantes de la Universidad Politécnica de Tecámac. Siéntete libre de explorar, modificar y mejorar. Los usuarios deben aprender a instalar, ejecutar y experimentar por sí mismos. ¡El conocimiento se construye con práctica!

