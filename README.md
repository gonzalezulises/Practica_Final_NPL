# Análisis de Sentimiento de Reseñas de Productos de Amazon

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Librerías-Scikit--learn%20%7C%20Pandas%20%7C%20NLTK-orange.svg)
![License](https://img.shields.io/badge/Licencia-MIT-green.svg)

Este repositorio contiene un proyecto completo de Procesamiento de Lenguaje Natural (NLP) enfocado en la clasificación de sentimiento (positivo vs. negativo) de reseñas de productos de Amazon. El proyecto abarca todas las etapas de un flujo de trabajo de Machine Learning, desde la exploración inicial de los datos hasta la evaluación y conclusión de un modelo final.

## 📝 Descripción del Proyecto

El objetivo principal es construir y evaluar un clasificador de sentimiento utilizando un enfoque de clasificación binaria supervisada. El proyecto se centra no solo en el rendimiento final del modelo, sino también en el análisis, la justificación de las decisiones técnicas y la interpretación de los resultados, siguiendo las directrices de una práctica realista de NLP.

---

## ✨ Características Principales

* **Análisis Exploratorio de Datos (EDA):**
    * Análisis de la distribución de calificaciones para identificar la estructura de los datos.
    * Identificación y visualización de un fuerte desbalanceo de clases.
    * Extracción y análisis de N-grams (bigramas) para encontrar las frases más comunes en cada sentimiento.
    * Generación de nubes de palabras para visualizar los términos más frecuentes.
    * Entrenamiento de un modelo Word2Vec y visualización de embeddings con t-SNE para explorar relaciones semánticas.

* **Preprocesado de Texto:**
    * Creación de una pipeline de limpieza robusta y encapsulada en una única función.
    * Pasos incluidos: conversión a minúsculas, eliminación de HTML/URLs, eliminación de puntuación y números, tokenización, eliminación de *stop words* y lematización.

* **Modelado y Evaluación:**
    * Vectorización de texto usando **TF-IDF** con unigramas y bigramas.
    * Entrenamiento y comparación de dos modelos: **Regresión Logística** y **Naive Bayes Multinomial**.
    * Aplicación de la técnica `class_weight='balanced'` para gestionar el desbalanceo de clases.
    * Evaluación de modelos utilizando reportes de clasificación (precisión, recall, f1-score) y matrices de confusión.

---

## 📊 Dataset

El proyecto utiliza el dataset **"Automotive_5"** del conjunto de datos de reseñas de productos de Amazon, recopilado por J. McAuley de la UCSD. Este es un dataset "5-core", lo que garantiza que cada usuario y producto tiene al menos cinco reseñas.

* **Fuente:** [Amazon Review Data (2014)](http://jmcauley.ucsd.edu/data/amazon/)
* **Archivo:** `reviews_Automotive_5.json.gz`

---

## 🚀 Resultados Clave

El análisis y modelado arrojaron dos conclusiones principales:

1.  **Fuerte Desbalanceo de Clases:** El análisis exploratorio reveló una proporción de reseñas positivas a negativas de casi **16 a 1**. Este fue el desafío técnico más importante a superar.

2.  **Rendimiento del Modelo Final:** El modelo de **Regresión Logística** fue el claro ganador, principalmente por su capacidad para manejar el desbalanceo de clases. El modelo Naive Bayes, sin este ajuste, resultó inútil al predecir únicamente la clase mayoritaria.

    **Métricas del Modelo de Regresión Logística (en el conjunto de prueba):**

    | Métrica | Clase 'Negativo' | Clase 'Positivo' | General (Accuracy) |
    | :--- | :---: | :---: | :---: |
    | **Precision** | 0.35 | 0.98 | |
    | **Recall** | **0.65** | 0.92 | **0.91** |
    | **F1-Score** | 0.46 | 0.95 | |

    La métrica más importante, el **recall de 0.65 para la clase negativa**, indica que el modelo es capaz de identificar correctamente el 65% de todas las quejas de los clientes, proporcionando un valor real para un caso de uso de negocio.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3
* **Librerías Principales:**
    * Pandas: Manipulación y análisis de datos.
    * Scikit-learn: Modelado de Machine Learning y métricas.
    * NLTK: Preprocesado de texto (stop words, lematización).
    * Gensim: Modelado de tópicos (Word2Vec).
    * Matplotlib y Seaborn: Visualización de datos.
    * WordCloud: Creación de nubes de palabras.
* **Entorno:** Jupyter Notebook (desarrollado en Google Colab).

---

## 📂 Estructura del Repositorio

├── Práctica_final_NLP_Ulises_González.ipynb
└── README.md
Of course. Here is the corrected and cleaned-up version of your Markdown text.

```markdown
# Análisis de Sentimiento de Reseñas de Productos de Amazon

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Librerías-Scikit--learn%20%7C%20Pandas%20%7C%20NLTK-orange.svg)
![License](https://img.shields.io/badge/Licencia-MIT-green.svg)

Este repositorio contiene un proyecto completo de Procesamiento de Lenguaje Natural (NLP) enfocado en la clasificación de sentimiento (positivo vs. negativo) de reseñas de productos de Amazon. El proyecto abarca todas las etapas de un flujo de trabajo de Machine Learning, desde la exploración inicial de los datos hasta la evaluación y conclusión de un modelo final.

## 📝 Descripción del Proyecto

El objetivo principal es construir y evaluar un clasificador de sentimiento utilizando un enfoque de clasificación binaria supervisada. El proyecto se centra no solo en el rendimiento final del modelo, sino también en el análisis, la justificación de las decisiones técnicas y la interpretación de los resultados, siguiendo las directrices de una práctica realista de NLP.

---

## ✨ Características Principales

* **Análisis Exploratorio de Datos (EDA):**
    * Análisis de la distribución de calificaciones para identificar la estructura de los datos.
    * Identificación y visualización de un fuerte desbalanceo de clases.
    * Extracción y análisis de N-grams (bigramas) para encontrar las frases más comunes en cada sentimiento.
    * Generación de nubes de palabras para visualizar los términos más frecuentes.
    * Entrenamiento de un modelo Word2Vec y visualización de embeddings con t-SNE para explorar relaciones semánticas.

* **Preprocesado de Texto:**
    * Creación de una pipeline de limpieza robusta y encapsulada en una única función.
    * Pasos incluidos: conversión a minúsculas, eliminación de HTML/URLs, eliminación de puntuación y números, tokenización, eliminación de *stop words* y lematización.

* **Modelado y Evaluación:**
    * Vectorización de texto usando **TF-IDF** con unigramas y bigramas.
    * Entrenamiento y comparación de dos modelos: **Regresión Logística** y **Naive Bayes Multinomial**.
    * Aplicación de la técnica `class_weight='balanced'` para gestionar el desbalanceo de clases.
    * Evaluación de modelos utilizando reportes de clasificación (precisión, recall, f1-score) y matrices de confusión.

---

## 📊 Dataset

El proyecto utiliza el dataset **"Automotive_5"** del conjunto de datos de reseñas de productos de Amazon, recopilado por J. McAuley de la UCSD. Este es un dataset "5-core", lo que garantiza que cada usuario y producto tiene al menos cinco reseñas.

* **Fuente:** [Amazon Review Data (2014)](http://jmcauley.ucsd.edu/data/amazon/)
* **Archivo:** `reviews_Automotive_5.json.gz`

---

## 🚀 Resultados Clave

El análisis y modelado arrojaron dos conclusiones principales:

1.  **Fuerte Desbalanceo de Clases:** El análisis exploratorio reveló una proporción de reseñas positivas a negativas de casi **16 a 1**. Este fue el desafío técnico más importante a superar.

2.  **Rendimiento del Modelo Final:** El modelo de **Regresión Logística** fue el claro ganador, principalmente por su capacidad para manejar el desbalanceo de clases. El modelo Naive Bayes, sin este ajuste, resultó inútil al predecir únicamente la clase mayoritaria.

    **Métricas del Modelo de Regresión Logística (en el conjunto de prueba):**

    | Métrica | Clase 'Negativo' | Clase 'Positivo' | General (Accuracy) |
    | :--- | :---: | :---: | :---: |
    | **Precision** | 0.35 | 0.98 | |
    | **Recall** | **0.65** | 0.92 | **0.91** |
    | **F1-Score** | 0.46 | 0.95 | |

    La métrica más importante, el **recall de 0.65 para la clase negativa**, indica que el modelo es capaz de identificar correctamente el 65% de todas las quejas de los clientes, proporcionando un valor real para un caso de uso de negocio.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3
* **Librerías Principales:**
    * Pandas: Manipulación y análisis de datos.
    * Scikit-learn: Modelado de Machine Learning y métricas.
    * NLTK: Preprocesado de texto (stop words, lematización).
    * Gensim: Modelado de tópicos (Word2Vec).
    * Matplotlib y Seaborn: Visualización de datos.
    * WordCloud: Creación de nubes de palabras.
* **Entorno:** Jupyter Notebook (desarrollado en Google Colab).

---

## 📂 Estructura del Repositorio

```

.
├── Práctica\_final\_NLP\_Ulises\_González.ipynb
└── README.md

````

---

## ⚙️ Instalación y Uso

Para ejecutar este proyecto localmente, sigue estos pasos:

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
    cd tu-repositorio
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    Crea un archivo `requirements.txt` con el siguiente contenido y luego ejecuta `pip install -r requirements.txt`.

    **Contenido para `requirements.txt`:**
    ```text
    pandas
    numpy
    requests
    matplotlib
    seaborn
    wordcloud
    nltk
    scikit-learn
    gensim
    jupyter
    ```

    **Comando de instalación:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecuta el Notebook:**
    Inicia Jupyter Notebook y abre el archivo `Práctica_final_NLP_Ulises_González.ipynb`.
    ```bash
    jupyter notebook
    ```
    Al ejecutar el notebook por primera vez, se descargarán automáticamente los datos de Amazon y los paquetes necesarios de NLTK.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles (si aplica).
````







