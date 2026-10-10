# 🌍 Air Quality Prediction (CO Concentration)

A Machine Learning regression project to predict urban **Carbon Monoxide (CO)** concentration based on chemical multisensor responses and ambient meteorological conditions.

---

## 📌 Project Overview

- **Project Title**: Air Quality Prediction (CO Concentration)
- **Domain**: Environmental Machine Learning / Air Quality Forecasting
- **Target Variable ($y$)**: `CO(GT)` (Ground Truth CO concentration from certified reference analyzer, measured in $mg/m^3$)
- **Feature Set ($X$)**: Chemical sensor responses (`PT08.S1` through `PT08.S5`, `NMHC`, `C6H6`, `NOx`, `NO2`), atmospheric conditions (Temperature `T`, Relative Humidity `RH`, Absolute Humidity `AH`), and cyclical time features (`Month`, `Day`, `Hour`).

---

## 📊 Dataset Description

- **Source**: [UCI Machine Learning Repository - Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
- **Data Scope**: 9,358 hourly recorded observations from an urban monitoring station in an Italian city (March 2004 – April 2005).
- **Missing Value Handling**: The sentinel value `-200` in the raw data represents missing observations and was converted to `NaN`. Rows with missing target values (`CO(GT)`) were dropped to preserve evaluation integrity. The post-cleaning modeling dataset retains over 7,600 valid records (exceeding the 1,000-row course requirement).

---

## ⚙️ Methodology & Data Leakage Prevention

1. **Chronological Train-Test Split (80:20)**:
   - Date and time features were parsed into datetime format (`_DateTime`).
   - Observations were sorted chronologically: the first **80%** serves as the training set (past data) and the remaining **20%** serves as the test set (future data). This eliminates look-ahead bias inherent to random splitting in time-series data.
2. **Preprocessing in Pipelines**:
   - Feature missing values were imputed using the median via `SimpleImputer`.
   - Feature scaling was conducted via `StandardScaler` for the linear model.
   - All transformations were encapsulated inside Scikit-learn `Pipeline` objects to ensure statistical parameters were computed **strictly from training data**, preventing data leakage into the test set.

---

## 🤖 Algorithms Evaluated

1. **Linear Regression** (with `StandardScaler`)
2. **Random Forest Regressor** (`n_estimators=150`, `random_state=42`)
3. **Gradient Boosting Regressor** (`random_state=42`)

---

## 📈 Model Evaluation & Benchmarks

Models were evaluated on the 20% unseen test set across 4 standard regression metrics:
- **MAE** (*Mean Absolute Error*) — Lower is better
- **MSE** (*Mean Squared Error*) — Lower is better
- **RMSE** (*Root Mean Squared Error*) — Lower is better
- **R²** (*Coefficient of Determination*) — Higher is better

| Rank | Model | MAE | MSE | RMSE | R² |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Linear Regression** | **0.3442** | **0.2714** | **0.5209** | **0.8499** |
| 🥈 | **Random Forest** | 0.3630 | 0.2916 | 0.5400 | 0.8397 |
| 🥉 | **Gradient Boosting** | 0.3968 | 0.3280 | 0.5727 | 0.8185 |

### 💡 Key Findings:
- **Linear Regression** demonstrated superior generalizability, achieving the lowest test errors (MAE = 0.3442, RMSE = 0.5209) and the highest explanatory power (R² = 0.8499).
- An **R² score of 0.8499** indicates that the model accounts for approximately **84.99% of the variance** in true CO concentration on unseen evaluation data.
- Linear Regression outperformed tree-based ensembles because semiconductor oxide sensors exhibit near-linear physical response curves to target gas concentrations, whereas ensemble trees were more prone to overfitting high-frequency sensor noise over chronological test splits.

---

## 📁 Repository Structure

```text
Air-Quality-Prediction/
├── air_quality_ml.ipynb        # End-to-end Jupyter Notebook
├── AirQualityUCI.csv           # Raw dataset
├── hasil_air_quality/          # Exported outputs
│   ├── model_air_quality_terbaik.joblib   # Persisted best model
│   └── hasil_prediksi_test.csv            # Test set actual vs predicted values
└── README.md                   # Project documentation
