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

```
The service will initialize and bind to http://127.0.0.1:5000.

## 2. Endpoints
A. Health Check (GET /health)

Verifies service availability, loaded model identity, and input feature requirements.
code Bash
```

curl.exe http://127.0.0.1:5000/health
```
## B. Inference Prediction (POST /predict)

Accepts a 16-element feature vector representing sensor responses and atmospheric parameters.

Command (PowerShell / Command Prompt):
code Powershell
```
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "@data.json"
```
Input Payload (data.json):
code JSON
```
{
  "data": [
    [1360.0, 150.0, 11.9, 1046.0, 166.0, 1056.0, 113.0, 1692.0, 1268.0, 13.6, 48.9, 0.7578, 3, 10, 2, 18]
  ]
}
```
Response Payload:
code JSON
```
{
  "latency_ms": 66.07,
  "model_used": "Linear Regression",
  "prediction_co_concentration": [
    2.733033760003181
  ],
  "status": "success"
}
```
## 3. Monitoring & Observability

    Latency Tracking: The API automatically measures request-to-response duration in milliseconds (latency_ms) to detect real-time inference bottlenecks.

    Audit Logging (monitoring.log): Each incoming request, response summary, and exception is persistently recorded with ISO timestamps for post-hoc analysis and data drift diagnosis.

## Loading the Saved Model for Inference

The best-performing model is saved in hasil_air_quality/model_air_quality_terbaik.joblib.

You can load the saved model and inspect predictions using the following Python script:
code Python
```
import joblib
import pandas as pd

# 1. Load the saved model artifact
artifact = joblib.load(
    "hasil_air_quality/model_air_quality_terbaik.joblib"
)

model = artifact["model"]
features = artifact["features"]

print("Model loaded:", artifact["model_name"])
print("Expected features:", features)

# 2. Load test predictions

test_data = pd.read_csv(
    "hasil_air_quality/hasil_prediksi_test.csv"
)

print("\nSample predictions:")
print(test_data.head())
```
## Repository Structure
code Text

```

Air-Quality-Prediction/
│
├── air_quality_ml.ipynb
│   └── End-to-end Jupyter Notebook
│
├── AirQualityUCI.csv
│   └── Raw dataset
│
├── app.py
│   └── Flask REST API for model deployment and monitoring
│
├── data.json
│   └── Sample 16-feature input payload for testing predictions
│
├── monitoring.log
│   └── Runtime inference logs and latency audit trail
│
├── Paper/
│   └── paper_chicco2021_r2_mae_mse.pdf (Reference paper for MAE, MSE, R²)
│
├── hasil_air_quality/
│   ├── model_air_quality_terbaik.joblib
│   ├── hasil_prediksi_test.csv
│   ├── laporan_missing_value.csv
│   ├── laporan_outlier_iqr.csv
│   └── perbandingan_model.csv
│
└── README.md
    └── Project documentation
```

## Technologies Used

    Python

    Flask (REST API Deployment)

    Jupyter Notebook

    Pandas

    NumPy

    Matplotlib

    Seaborn

    Scikit-learn

    Joblib

## Project Objectives

    Analyze air quality data and identify patterns in carbon monoxide concentration.

    Apply data preprocessing techniques to handle missing values.

    Detect potential outliers using the Interquartile Range (IQR) method.

    Develop and compare multiple regression models.

    Evaluate model performance using MAE, MSE, RMSE, and R².

    Save the best-performing model for future inference.

    Deploy the optimal model as an HTTP REST API with latency monitoring and audit logging.

## Conclusion

Based on the evaluation results, Linear Regression achieved the best performance among the three tested algorithms, with an MAE of 0.3442 and an R² score of 0.8499.

The model was successfully operationalized using a Flask web API, delivering real-time predictions with sub-100ms latency (~66 ms) and persistent operational logging, demonstrating the viability of this solution in production environmental monitoring scenarios.
