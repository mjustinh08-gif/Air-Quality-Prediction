# Air Quality Prediction (CO Concentration)

A Machine Learning regression project to predict Carbon Monoxide (CO) concentration based on chemical multisensor responses and ambient meteorological conditions.

---

## Project Overview

- **Project Title:** Air Quality Prediction (CO Concentration)
- **Domain:** Environmental Machine Learning / Air Quality Forecasting
- **Target Variable:** CO(GT) (Ground Truth CO concentration from reference analyzer in mg/m³)
- **Feature Set:** Chemical sensor responses (PT08.S1 through PT08.S5, NMHC, C6H6, NOx, NO2), atmospheric conditions (Temperature T, Relative Humidity RH, Absolute Humidity AH), and cyclical time features (Month, Day, DayOfWeek, Hour).

---

## Dataset Description

- **Source:** [UCI Machine Learning Repository - Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
- **Data Scope:** 9,358 hourly recorded observations from an urban monitoring station in an Italian city (March 2004 to April 2005).
- **Missing Value Handling:** The value `-200` represents missing sensor readings and was converted to `NaN`. Rows with missing target values (`CO(GT)`) were removed, retaining 7,674 valid records for modelling (exceeding the 1,000-row course requirement).

---

## Methodology and Data Leakage Prevention

### 1. Chronological Train-Test Split (80:20)

- The `Date` and `Time` columns were parsed into a proper datetime format (`_DateTime`).
- Data was sorted chronologically: the first 80% serves as the training set (6,139 rows), and the remaining 20% serves as the test set (1,535 rows).
- This approach helps prevent look-ahead data leakage in time-series data.

### 2. Preprocessing Pipeline

- Missing feature values were imputed using the median (`SimpleImputer`).
- Feature scaling was applied using `StandardScaler` for Linear Regression.
- All transformations were wrapped inside Scikit-learn `Pipeline` objects to ensure statistical parameters were computed solely from training data.

---

## Evaluated Algorithms

1. Linear Regression (with StandardScaler)
2. Random Forest Regressor (`n_estimators=150`, `random_state=42`)
3. Gradient Boosting Regressor (`random_state=42`)

---

## Model Evaluation and Results

Models were evaluated on the 20% unseen test set (1,535 rows) using four standard regression metrics.

| Rank | Model | MAE | MSE | RMSE | R² |
|:---:|---|---:|---:|---:|---:|
| 1 | Linear Regression | 0.3442 | 0.2714 | 0.5209 | 0.8499 |
| 2 | Random Forest Regressor | 0.3630 | 0.2916 | 0.5400 | 0.8387 |
| 3 | Gradient Boosting Regressor | 0.3868 | 0.3280 | 0.5727 | 0.8185 |

### Analysis

- Linear Regression achieved the best overall performance, with the lowest MAE (0.3442) and the highest R² score (0.8499).
- An R² score of 0.8499 indicates that the model explains approximately 84.99% of the variance in the observed CO concentration values on the test set.
- Linear Regression outperformed the tree-based ensemble models on this chronological test split.

---

## Model Deployment and Monitoring

The trained machine learning pipeline is deployed as a production-ready HTTP REST API service using **Flask**, complete with runtime inference logging and latency monitoring.

### 1. Running the API Service
Start the Flask application server:
```bash
python app.py
