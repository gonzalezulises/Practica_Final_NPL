# \# Análisis de Sentimiento de Reseñas de Productos de Amazon

# 

# !\[Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)

# !\[Libraries](https://img.shields.io/badge/Librerías-Scikit--learn%20%7C%20Pandas%20%7C%20NLTK-orange.svg)

# !\[License](https://img.shields.io/badge/Licencia-MIT-green.svg)

# 

# Este repositorio contiene un proyecto completo de Procesamiento de Lenguaje Natural (NLP) enfocado en la clasificación de sentimiento (positivo vs. negativo) de reseñas de productos de Amazon. El proyecto abarca todas las etapas de un flujo de trabajo de Machine Learning, desde la exploración inicial de los datos hasta la evaluación y conclusión de un modelo final.

# 

# \## 📝 Descripción del Proyecto

# 

# El objetivo principal es construir y evaluar un clasificador de sentimiento utilizando un enfoque de clasificación binaria supervisada. El proyecto se centra no solo en el rendimiento final del modelo, sino también en el análisis, la justificación de las decisiones técnicas y la interpretación de los resultados, siguiendo las directrices de una práctica realista de NLP.

# 

# ---

# 

# \## ✨ Características Principales

# 

# \* \*\*Análisis Exploratorio de Datos (EDA):\*\*

# &nbsp;   \* Análisis de la distribución de calificaciones para identificar la estructura de los datos.

# &nbsp;   \* Identificación y visualización de un fuerte desbalanceo de clases.

# &nbsp;   \* Extracción y análisis de N-grams (bigramas) para encontrar las frases más comunes en cada sentimiento.

# &nbsp;   \* Generación de nubes de palabras para visualizar los términos más frecuentes.

# &nbsp;   \* Entrenamiento de un modelo Word2Vec y visualización de embeddings con t-SNE para explorar relaciones semánticas.

# 

# \* \*\*Preprocesado de Texto:\*\*

# &nbsp;   \* Creación de una pipeline de limpieza robusta y encapsulada en una única función.

# &nbsp;   \* Pasos incluidos: conversión a minúsculas, eliminación de HTML/URLs, eliminación de puntuación y números, tokenización, eliminación de \*stop words\* y lematización.

# 

# \* \*\*Modelado y Evaluación:\*\*

# &nbsp;   \* Vectorización de texto usando \*\*TF-IDF\*\* con unigramas y bigramas.

# &nbsp;   \* Entrenamiento y comparación de dos modelos: \*\*Regresión Logística\*\* y \*\*Naive Bayes Multinomial\*\*.

# &nbsp;   \* Aplicación de la técnica `class\_weight='balanced'` para gestionar el desbalanceo de clases.

# &nbsp;   \* Evaluación de modelos utilizando reportes de clasificación (precisión, recall, f1-score) y matrices de confusión.

# 

# ---

# 

# \## 📊 Dataset

# 

# El proyecto utiliza el dataset \*\*"Automotive\_5"\*\* del conjunto de datos de reseñas de productos de Amazon, recopilado por J. McAuley de la UCSD. Este es un dataset "5-core", lo que garantiza que cada usuario y producto tiene al menos cinco reseñas.

# 

# \* \*\*Fuente:\*\* \[Amazon Review Data (2014)](http://jmcauley.ucsd.edu/data/amazon/)

# \* \*\*Archivo:\*\* `reviews\_Automotive\_5.json.gz`

# 

# ---

# 

# \## 🚀 Resultados Clave

# 

# El análisis y modelado arrojaron dos conclusiones principales:

# 

# 1\.  \*\*Fuerte Desbalanceo de Clases:\*\* El análisis exploratorio reveló una proporción de reseñas positivas a negativas de casi \*\*16 a 1\*\*. Este fue el desafío técnico más importante a superar.

# 

# 2\.  \*\*Rendimiento del Modelo Final:\*\* El modelo de \*\*Regresión Logística\*\* fue el claro ganador, principalmente por su capacidad para manejar el desbalanceo de clases. El modelo Naive Bayes, sin este ajuste, resultó inútil al predecir únicamente la clase mayoritaria.

# 

# &nbsp;   \*\*Métricas del Modelo de Regresión Logística (en el conjunto de prueba):\*\*

# 

# &nbsp;   | Métrica               | Clase 'Negativo' | Clase 'Positivo' | General (Accuracy) |

# &nbsp;   | --------------------- | :--------------: | :--------------: | :----------------: |

# &nbsp;   | \*\*Precision\*\* |       0.35       |       0.98       |                    |

# &nbsp;   | \*\*Recall\*\* |     \*\*0.65\*\* |       0.92       |      \*\*0.91\*\* |

# &nbsp;   | \*\*F1-Score\*\* |       0.46       |       0.95       |                    |

# 

# &nbsp;   La métrica más importante, el \*\*recall de 0.65 para la clase negativa\*\*, indica que el modelo es capaz de identificar correctamente el 65% de todas las quejas de los clientes, proporcionando un valor real para un caso de uso de negocio.

# 

# ---

# 

# \## 🛠️ Tecnologías Utilizadas

# 

# \* \*\*Lenguaje:\*\* Python 3

# \* \*\*Librerías Principales:\*\*

# &nbsp;   \* Pandas: Manipulación y análisis de datos.

# &nbsp;   \* Scikit-learn: Modelado de Machine Learning y métricas.

# &nbsp;   \* NLTK: Preprocesado de texto (stop words, lematización).

# &nbsp;   \* Gensim: Modelado de tópicos (Word2Vec).

# &nbsp;   \* Matplotlib y Seaborn: Visualización de datos.

# &nbsp;   \* WordCloud: Creación de nubes de palabras.

# \* \*\*Entorno:\*\* Jupyter Notebook (desarrollado en Google Colab).

# 

# ---

# 

# \## 📂 Estructura del Repositorio

├── Práctica\_final\_NLP\_Ulises\_González.ipynb

└── README.md



\## ⚙️ Instalación y Uso



Para ejecutar este proyecto localmente, sigue estos pasos:



1\.  \*\*Clona el repositorio:\*\*

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)

&nbsp;   cd tu-repositorio

&nbsp;   ```



2\.  \*\*Crea un entorno virtual (recomendado):\*\*

&nbsp;   ```bash

&nbsp;   python -m venv venv

&nbsp;   source venv/bin/activate  # En Windows: venv\\Scripts\\activate

&nbsp;   ```



3\.  \*\*Instala las dependencias:\*\*

&nbsp;   Crea un archivo `requirements.txt` con el siguiente contenido y luego ejecuta `pip install -r requirements.txt`.



&nbsp;   \*\*Contenido para `requirements.txt`:\*\*

&nbsp;   ```text

&nbsp;   pandas

&nbsp;   numpy

&nbsp;   requests

&nbsp;   matplotlib

&nbsp;   seaborn

&nbsp;   wordcloud

&nbsp;   nltk

&nbsp;   scikit-learn

&nbsp;   gensim

&nbsp;   jupyter

&nbsp;   ```



&nbsp;   \*\*Comando de instalación:\*\*

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



4\.  \*\*Ejecuta el Notebook:\*\*

&nbsp;   Inicia Jupyter Notebook y abre el archivo `Práctica\_final\_NLP\_Ulises\_González.ipynb`.

&nbsp;   ```bash

&nbsp;   jupyter notebook

&nbsp;   ```

&nbsp;   Al ejecutar el notebook por primera vez, se descargarán automáticamente los datos de Amazon y los paquetes necesarios de NLTK.



---



\## 📄 Licencia



Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles (si aplica).

















