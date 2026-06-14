# Hotel Cancellation Prediction ML

Proyecto final de Machine Learning orientado a predecir cancelaciones hoteleras a partir del dataset **Hotel Booking Demand** de Kaggle.

El objetivo es construir un modelo supervisado capaz de estimar si una reserva será cancelada antes de la estancia, usando únicamente información disponible antes de la llegada del cliente.

## Dataset

- Fuente: [Hotel Booking Demand - Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Target: `is_canceled`
- Problema: clasificación binaria
- Dataset bruto: 119.390 reservas y 32 columnas
- Dataset limpio: 118.564 reservas y 34 columnas
- Split: train/test estratificado 80/20

## Estructura del repositorio

```text
Proyecto_ML/
|-- data/
|   |-- raw/
|   |   `-- hotel_bookings.csv
|   |-- processed/
|   |   `-- hotel_bookings_clean.csv
|   |-- train/
|   |   `-- train.csv
|   `-- test/
|       `-- test.csv
|-- docs/
|   |-- ds_tecnica.pdf
|   |-- memoria.ipynb
|   `-- ML_primera_presentacion.ipynb
|-- models/
|   |-- final_model.pkl
|   |-- model_config.yaml
|   `-- trained_model_1.pkl
|-- notebooks/
|   |-- 01_fuentes.ipynb
|   |-- 02_limpiezaEDA.ipynb
|   `-- 03_entrenamiento_evaluacion.ipynb
|-- src/
|   |-- __init__.py
|   |-- data_processing.py
|   |-- evaluation.py
|   `-- training.py
`-- README.md
```

## Flujo de trabajo

1. `notebooks/01_fuentes.ipynb`: carga inicial, fuente del dataset y contexto del problema.
2. `notebooks/02_limpiezaEDA.ipynb`: limpieza, control de nulos, filtrado de anomalías, análisis exploratorio y generación de datasets procesados.
3. `notebooks/03_entrenamiento_evaluacion.ipynb`: comparación de modelos, evaluación, tratamiento de duplicados, optimización de XGBoost y selección del modelo final.

## Preparación de datos

La limpieza prioriza evitar data leakage y trabajar con variables disponibles antes de la estancia:

- Eliminación de variables con posible fuga de información, como `reservation_status` y `reservation_status_date`.
- Exclusión de `assigned_room_type` por posible información posterior a la reserva.
- Conversión de identificadores con muchos nulos (`company`, `agent`) en flags binarias: `has_company` y `has_agent`.
- Creación de variables derivadas como `total_nights`, `total_guests` y señales binarias de reserva.
- Filtrado de registros anómalos: reservas sin huéspedes, sin noches o con `adr` negativo.

## Modelado

Se compararon distintos modelos supervisados de clasificación:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

El modelo final seleccionado es **XGBoost**, optimizado mediante búsqueda de hiperparámetros y evaluado sobre el conjunto de test.

## Resultados principales

Métricas del modelo final:

- Accuracy: 0,881
- Precision: 0,830
- Recall: 0,856
- F1-score: 0,842
- ROC-AUC: 0,953

La métrica principal utilizada para la selección del modelo fue **F1-score**, al equilibrar precision y recall sobre la clase positiva: reservas canceladas.

## Modelos guardados

- `models/trained_model_1.pkl`: modelo entrenado intermedio.
- `models/final_model.pkl`: modelo final XGBoost.
- `models/model_config.yaml`: configuración y métricas principales del modelo final.

## Scripts

- `src/data_processing.py`: funciones de carga, limpieza, feature engineering y generación de datasets procesados.
- `src/training.py`: definición del preprocesado, modelos y entrenamiento.
- `src/evaluation.py`: funciones de evaluación para clasificación binaria.

## Documentación

La carpeta `docs/` incluye material de apoyo del proyecto, como la presentación técnica en PDF y notebooks de documentación.

## Demo Streamlit

La demo operativa está desarrollada localmente en la carpeta `app_streamlit/`, pero **no está incluida todavía en esta versión del repositorio**.

Se añadirá en una segunda subida junto con la presentación de negocio.

Cuando esté disponible, se ejecutará desde la raíz del proyecto con:

```bash
streamlit run app_streamlit/app.py
```

## Próximos pasos

- Añadir `booking_id` o una regla auditable para separar duplicados reales de reservas similares.
- Realizar validación temporal entrenando con histórico y evaluando en fechas posteriores.
- Ajustar el threshold de decisión con costes reales de falsos positivos y falsos negativos.
