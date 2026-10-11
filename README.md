# Air Quality Prediction (CO Concentration)

A Machine Learning regression project to predict Carbon Monoxide (CO) concentration based on chemical multisensor responses and ambient meteorological conditions.

---

## Project Overview

- Project Title: Air Quality Prediction (CO Concentration)
- Domain: Environmental Machine Learning / Air Quality Forecasting
- Target Variable: CO(GT) (Ground Truth CO concentration from reference analyzer in mg/m3)
- Feature Set: Chemical sensor responses (PT08.S1 through PT08.S5, NMHC, C6H6, NOx, NO2), atmospheric conditions (Temperature T, Relative Humidity RH, Absolute Humidity AH), and cyclical time features (Month, Day, DayOfWeek, Hour).

---

## Dataset Description

- Source: UCI Machine Learning Repository - Air Quality Dataset (https://archive.ics.uci.edu/dataset/360/air+quality)
- Data Scope: 9,358 hourly recorded observations from an urban monitoring station in an Italian city (March 2004 to April 2005).
- Missing Value Handling: The value -200 represents missing sensor readings and was converted to NaN. Rows with missing target values (CO(GT)) were removed, retaining 7,674 valid records for modelling (exceeding the 1,000-row course requirement).

---

## Methodology and Data Leakage Prevention

1. Chronological Train-Test Split (80:20):
   - The Date and Time columns were parsed into a proper datetime format (_DateTime).
   - Data was sorted chronologically: the first 80% serves as the training set (6,139 rows) and the remaining 20% serves as the test set (1,535 rows). This prevents look-ahead data leakage in time-series data.
2. Preprocessing Pipeline:
   - Missing feature values were imputed using the median (SimpleImputer).
   - Feature scaling was applied using StandardScaler for Linear Regression.
   - All transformations were wrapped inside Scikit-learn Pipeline objects to ensure statistical parameters were computed solely from training data.

---

## Evaluated Algorithms

1. Linear Regression (with StandardScaler)
2. Random Forest Regressor (n_estimators=150, random_state=42)
3. Gradient Boosting Regressor (random_state=42)

---

## Model Evaluation and Results

Models were evaluated on the 20% unseen test set (1,535 rows) across 4 standard regression metrics:

| Rank | Model | MAE | MSE | RMSE | R2 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | Linear Regression | 0.3442 | 0.2714 | 0.5209 | 0.8499 |
| 2 | Random Forest Regressor | 0.3630 | 0.2916 | 0.5400 | 0.8387 |
| 3 | Gradient Boosting Regressor | 0.3868 | 0.3280 | 0.5727 | 0.8185 |

### Analysis:
- Linear Regression achieved the best overall performance with the lowest MAE (0.3442) and the highest R2 score (0.8499).
- An R2 score of 0.8499 indicates that the model accounts for approximately 84.99% of the variance in true CO concentration on the test set.
- Linear Regression outperformed tree-based ensembles because semiconductor oxide sensors exhibit near-linear physical response curves to gas concentration, whereas decision-tree ensembles were more prone to overfitting sensor noise over the chronological test split.

---

## Loading Saved Model for Inference

The best-performing model is saved in `hasil_air_quality/model_air_quality_terbaik.joblib`. You can load the model artifact and run inference on new data using the following Python script:

```python
import joblib
import pandas as pd

# 1. Load saved model artifact
artifact = joblib.load("hasil_air_quality/model_air_quality_terbaik.joblib")
model = artifact["model"]
features = artifact["features"]

print("Model loaded:", artifact["model_name"])
print("Expected features:", features)

# 2. Example: Predict on test data
test_data = pd.read_csv("hasil_air_quality/hasil_prediksi_test.csv")
print("Sample predictions:")
print(test_data.head())

```
## Repository Structure

Air-Quality-Prediction/
├── air_quality_ml.ipynb        # End-to-end Jupyter Notebook
├── AirQualityUCI.csv           # Raw dataset
├── hasil_air_quality/          # Output directory
│   ├── model_air_quality_terbaik.joblib   # Persisted best model artifact
│   ├── hasil_prediksi_test.csv            # Test set actual vs predicted values
│   ├── laporan_missing_value.csv          # Missing value report
│   ├── laporan_outlier_iqr.csv            # IQR outlier report
│   └── perbandingan_model.csv             # Model comparison metrics
└── README.md                   # Project documentation
