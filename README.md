# Hotel_cancellation_prediction_ML

Proyecto de Machine Learning para predecir cancelaciones hoteleras usando el dataset Hotel Booking Demand.

## Estructura del proyecto

```text
Proyecto_ML/
|-- app_streamlit/
|   `-- app.py
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
|-- models/
|   |-- trained_model_1.pkl
|   |-- final_model.pkl
|   `-- model_config.yaml
|-- notebooks/
|   |-- 01_fuentes.ipynb
|   |-- 02_limpiezaEDA.ipynb
|   `-- 03_entrenamiento_evaluacion.ipynb
|-- src/
|   |-- data_processing.py
|   |-- evaluation.py
|   `-- training.py
`-- README.md
```

## Dataset

Fuente: Kaggle - Hotel Booking Demand.

El objetivo es predecir la variable `is_canceled`, que indica si una reserva hotelera fue cancelada o no.

## Flujo de trabajo

1. `01_fuentes.ipynb`: fuente del dataset y contexto inicial.
2. `02_limpiezaEDA.ipynb`: limpieza, EDA, tratamiento de variables, decision sobre duplicados y guardado del dataset limpio con su split train/test.
3. `03_entrenamiento_evaluacion.ipynb`: comparacion de modelos, evaluacion, hiperparametrizacion de XGBoost y seleccion del modelo final.

## Modelos guardados

- `models/trained_model_1.pkl`: modelo entrenado intermedio.
- `models/final_model.pkl`: modelo final tras la hiperparametrizacion.
- `models/model_config.yaml`: configuracion y metricas del modelo final.

## Scripts

- `src/data_processing.py`: funciones de carga, limpieza y generacion de `processed`, `train` y `test`.
- `src/training.py`: preprocesado, definicion de modelos y entrenamiento.
- `src/evaluation.py`: metricas de evaluacion para clasificacion binaria.

## Demo Streamlit

La demo interactiva se encuentra en `app_streamlit/app.py`.

Ejecutar desde la raiz del proyecto:

```bash
streamlit run app_streamlit/app.py
```
