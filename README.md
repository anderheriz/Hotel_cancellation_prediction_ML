# Hotel_cancellation_prediction_ML

Proyecto de Machine Learning para predecir cancelaciones hoteleras usando el dataset Hotel Booking Demand.

## Objetivo

Predecir si una reserva hotelera sera cancelada (`is_canceled`) a partir de variables de la reserva, cliente, canal de venta, deposito, fechas y precio.

## Estructura actual

```text
data/
  raw/hotel_bookings.csv
  processed/hotel_bookings_clean.csv
  train/train.csv
  test/test.csv
notebooks/
  01_fuentes.ipynb
  02_limpiezaEDA.ipynb
  03_baseline_random_forest.ipynb
src/
  data_processing.py
  training.py
  evaluation.py
  train_baseline.py
models/
  baseline_random_forest.pkl
```

## Dataset

Fuente: Kaggle - Hotel Booking Demand.

Target: `is_canceled`.

Problema: clasificacion binaria supervisada.

## Baseline

La primera iteracion del modelo usa Random Forest como baseline. El entrenamiento se encuentra en:

```text
notebooks/03_baseline_random_forest.ipynb
```

El modelo baseline guardado es:

```text
models/baseline_random_forest.pkl
```

## Ejecucion

Generar datos procesados:

```bash
python src/data_processing.py
```

Entrenar baseline:

```bash
python src/train_baseline.py
```
