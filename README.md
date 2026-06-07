# Jakarta Flood Prediction Model

A machine learning pipeline for predicting flood events across Jakarta's administrative regions using historical weather data and real-time weather forecasts from OpenWeatherMap.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Model Training & Tuning](#model-training--tuning)
- [Inference Pipeline](#inference-pipeline)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [Configuration](#configuration)
- [Model Checkpoint](#model-checkpoint)

---

## Overview

This project builds a binary flood classification model for four Jakarta regions ,**Jakarta Pusat, Jakarta Selatan, Jakarta Timur, and Jakarta Utara**, using XGBoost trained on daily weather station records. A separate inference notebook fetches live 5-day forecasts from the OpenWeatherMap API and generates next-day flood probability scores and alerts.

**Prediction task:** Given daily weather conditions for a region, predict whether a flood event will occur (`flood = 1`) or not (`flood = 0`).

---

## Project Structure

```
├── Raw_Weather_Jakarta.csv          # Raw weather station data (2016–present)
├── training.ipynb                   # Full training pipeline (EDA, model, checkpoint)
├── inference.ipynb                  # Live forecast + flood alert generation
└── model_data.pkl                   # Serialized model checkpoint (see below)
```

---

## Dataset

**Source:** Indonesian weather station records (`Raw_Weather_Jakarta.csv`)

**Coverage:** Daily observations per region, starting from January 2016.

**Raw columns used:**

| Column     | Description                              |
|------------|------------------------------------------|
| `date`     | Observation date                         |
| `Tn`       | Minimum temperature (°C)                 |
| `Tx`       | Maximum temperature (°C)                 |
| `Tavg`     | Average temperature (°C)                 |
| `RH_avg`   | Average relative humidity (%)            |
| `RR`       | Daily rainfall (mm)                      |
| `ff_avg`   | Average wind speed (m/s)                 |
| `ddd_x`    | Wind direction (degrees, 0–360)          |
| `region_name` | Administrative region name            |
| `flood`    | Binary flood label (0 = No Flood, 1 = Flood) |

**Dropped columns:** `ss`, `ff_x`, `ddd_car`, `station_name`, `station_id`

**Class imbalance:** Flood events are a minority class; the model uses `sample_weight="balanced"` and Optuna-tuned `scale_pos_weight` to handle this.

---

## Methodology

### 1. Preprocessing

- Rows sorted chronologically per region to preserve temporal order
- `region_name` label-encoded to `region_id`
- `RR` (rainfall) missing values filled with `0.0` (no rain)
- All other numeric columns imputed via per-region forward/backward fill, then global mean
- `ddd_x` (wind direction in degrees) decomposed into `wind_direction_sin` and `wind_direction_cos` to preserve circular continuity

### 2. Train / Val / Test Split

Temporal split, no shuffling, to prevent data leakage:

| Split      | Proportion | Date Range                     |
|------------|------------|--------------------------------|
| Train      | 70%        | Earliest (~70th percentile)    |
| Validation | 15%        | 70th (85th percentile)         |
| Test       | 15%        | 85th percentile (Latest)       |

Features are scaled using `RobustScaler` fitted exclusively on the training set.

---

## Feature Engineering

All features are computed in strict temporal order, grouped by `region_id`, to avoid leakage.

### Temporal Cyclical Encoding

| Feature      | Formula                              |
|--------------|--------------------------------------|
| `month_sin`  | `sin(2π × month / 12)`               |
| `month_cos`  | `cos(2π × month / 12)`               |
| `doy_sin`    | `sin(2π × day_of_year / 365)`        |
| `doy_cos`    | `cos(2π × day_of_year / 365)`        |

### Rainfall Lag Features

| Feature           | Description                   |
|-------------------|-------------------------------|
| `rainfall_lag1d`  | Rainfall 1 day prior          |
| `rainfall_lag3d`  | Rainfall 3 days prior         |
| `rainfall_lag7d`  | Rainfall 7 days prior         |

### Rainfall Rolling Window Features

| Feature                  | Window | Aggregation |
|--------------------------|--------|-------------|
| `rainfall_rolling3d_sum` | 3 days | Sum         |
| `rainfall_rolling7d_sum` | 7 days | Sum         |
| `rainfall_rolling14d_sum`| 14 days| Sum         |
| `rainfall_rolling3d_max` | 3 days | Max         |
| `rainfall_rolling7d_max` | 7 days | Max         |
| `rainfall_rolling7d_std` | 7 days | Std Dev     |

### Humidity Rolling Window Features

| Feature                    | Window | Aggregation |
|----------------------------|--------|-------------|
| `humidity_rolling3d_mean`  | 3 days | Mean        |
| `humidity_rolling7d_mean`  | 7 days | Mean        |

All rainfall-based features are log-transformed via `log1p` to reduce skewness.

---

## Model Training & Tuning

**Model:** `XGBClassifier` (XGBoost)

**Baseline:** A fixed-parameter model trained with `compute_sample_weight("balanced")` is used to verify the pipeline before tuning.

**Hyperparameter Tuning:** [Optuna](https://optuna.org/) with TPE sampler over 60 trials.

**Objective function:** F₂ score at threshold 0.5 on the validation set.
F₂ is chosen over F₁ to weight recall more heavily, missing a flood event (false negative) is costlier than a false alarm.

**Search space:**

| Parameter           | Range / Type              |
|---------------------|---------------------------|
| `n_estimators`      | 100 – 2000                |
| `learning_rate`     | 0.005 – 0.15 (log)        |
| `max_depth`         | 3 – 7                     |
| `min_child_weight`  | 1 – 30                    |
| `subsample`         | 0.6 – 1.0                 |
| `colsample_bytree`  | 0.5 – 1.0                 |
| `colsample_bylevel` | 0.5 – 1.0                 |
| `colsample_bynode`  | 0.5 – 1.0                 |
| `gamma`             | 0.0 – 15.0                |
| `reg_alpha`         | 1e-4 – 50.0 (log)         |
| `reg_lambda`        | 0.01 – 20.0 (log)         |
| `scale_pos_weight`  | 1.0 – 15.0                |
| `max_delta_step`    | 0 – 10                    |

**Final model:** Trained on the combined train + validation set using the best Optuna parameters, evaluated on the held-out test set.

---

## Inference Pipeline

The inference notebook (`inference.ipynb`) runs a live prediction for the next 5 days across all four Jakarta regions.

**Steps:**

1. Fetch 5-day / 3-hour forecast from OpenWeatherMap API for each region's coordinates
2. Aggregate 3-hour slots into daily records (min/max/mean temperature, total rainfall, mean humidity, mean wind speed, circular-mean wind direction)
3. Apply the same feature engineering logic used in training (lag features, rolling windows, log transforms)
4. Scale features with the saved `RobustScaler`
5. Predict flood probability using the saved `XGBClassifier`
6. Apply flood alert threshold (`FLOOD_THRESHOLD = 0.45`) to generate binary alerts

**Output columns:**

| Column               | Description                           |
|----------------------|---------------------------------------|
| `region_name`        | Jakarta administrative region         |
| `date`               | Forecast date                         |
| `RR`                 | Forecasted daily rainfall (mm)        |
| `RH_avg`             | Forecasted average humidity (%)       |
| `Tavg`               | Forecasted average temperature (°C)   |
| `flood_probability`  | Model output probability (0.0 – 1.0)  |
| `alert_label`        | `No Flood` or `Flood`                 |

---

## Results

Evaluation metrics are computed on the temporally held-out **test set** using the final model. Key metrics reported:

- ROC-AUC
- Average Precision (PR-AUC)
- Classification Report (Precision, Recall, F1 per class)
- Confusion Matrix

> Detailed metric values are logged in the training notebook output cells.

---

## Setup & Usage

### Requirements

```bash
pip install xgboost optuna scikit-learn pandas numpy matplotlib seaborn requests
```

### Training

1. Place `Raw_Weather_Jakarta.csv` in the working directory (or update `DATA_DIR`)
2. Mount Google Drive and set `MODEL_DIR` to your checkpoint folder
3. Run all cells in `training.ipynb`

The notebook will:
- Perform EDA and visualizations
- Preprocess and engineer features
- Train a baseline XGBoost model
- Run Optuna hyperparameter search (60 trials)
- Retrain the final model on train + val
- Save `model_data.pkl` to `MODEL_DIR`

### Inference

1. Copy `model_data.pkl` to `/content/` (or update `MODEL_PATH`)
2. Set your OpenWeatherMap API key in `OWM_API_KEY`
3. Run all cells in `inference.ipynb`

Output is a DataFrame showing flood probability and alert status per region per day for the next 5 days.

---

## Configuration

| Variable          | Location           | Description                                     |
|-------------------|--------------------|-------------------------------------------------|
| `DATA_DIR`        | `training.ipynb`   | Path to raw CSV data file                       |
| `MODEL_DIR`       | `training.ipynb`   | Directory to save model checkpoint              |
| `MODEL_PATH`      | `inference.ipynb`  | Path to `model_data.pkl`                        |
| `OWM_API_KEY`     | `inference.ipynb`  | OpenWeatherMap API key                          |
| `OWM_UNITS`       | `inference.ipynb`  | Unit system (`metric` = °C, mm, m/s)            |
| `REGIONS`         | `inference.ipynb`  | Dict of region name (lat, lon) coordinates      |
| `FLOOD_THRESHOLD` | `inference.ipynb`  | Probability threshold for flood alert (default: 0.45) |

---

## Model Checkpoint

`model_data.pkl` is a Python pickle file containing the following keys:

| Key            | Type                    | Description                                      |
|----------------|-------------------------|--------------------------------------------------|
| `le_region`    | `LabelEncoder`          | Fitted encoder for `region_name to region_id`    |
| `region_map`   | `dict`                  | Human-readable `{region_name: region_id}` map    |
| `feature_cols` | `list[str]`             | Ordered list of 24 feature column names          |
| `scaler`       | `RobustScaler`          | Fitted scaler (trained on train split only)      |
| `flood_model`  | `XGBClassifier`         | Final trained XGBoost model                      |
| `best_params`  | `dict`                  | Best Optuna hyperparameters used for final model |

---

## Limitations & Future Work

**Current limitations:**

- Inference lag features (`lag1d`, `lag3d`, `lag7d`) are derived only from the 5-day forecast window, not from actual historical station data, which may reduce accuracy for the first few forecast days
- The model is trained on station-level data but inference uses grid-point API data; systematic biases may exist between the two
- Jakarta Barat is not included in the current region set