# 🌍 Air Quality Prediction (CO Concentration)

Proyek *Machine Learning* untuk memprediksi konsentrasi polutan **Karbon Monoksida (CO)** sebagai salah satu indikator kualitas udara menggunakan data respons sensor kimia dan kondisi lingkungan atmosfer.

---

## 📌 Ringkasan Proyek

- **Judul Proyek**: Air Quality Prediction (CO Concentration)
- **Topik**: Regresi Machine Learning untuk Kualitas Udara
- **Target Variabel ($y$)**: `CO(GT)` (Konsentrasi CO dari *reference analyzer*, satuan mg/m³)
- **Fitur Input ($X$)**: Pengukuran respons sensor kimia (`PT08.S1` s.d. `PT08.S5`, `NMHC`, `C6H6`, `NOx`, `NO2`), variabel lingkungan (`T` suhu, `RH` kelembapan relatif, `AH` kelembapan absolut), serta fitur waktu (`Month`, `Day`, `Hour`).

---

## 📊 Dataset

- **Sumber**: [UCI Machine Learning Repository - Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
- **Karakteristik Data**: 9.358 data observasi per jam (*hourly*) yang dicatat pada stasiun pemantau di area perkotaan Italia (Maret 2004 – April 2005).
- **Penanganan Missing Value**: Nilai `-200` pada dataset mentah merepresentasikan data hilang (*missing value*) dan diubah menjadi `NaN`. Baris dengan nilai target `CO(GT)` kosong dihapus untuk menjaga keabsahan evaluasi. Setelah pembersihan, dataset menyisakan lebih dari 7.600 baris (memenuhi syarat minimal 1.000 baris data).

---

## ⚙️ Metodologi & Pencegahan Data Leakage

1. **Pengurutan Waktu & Split Kronologis**:
   - Kolom `Date` dan `Time` dikonversi menjadi format waktu asli (`_DateTime`).
   - Data diurutkan secara kronologis dan dibagi menjadi **80% data training** (masa lalu) dan **20% data testing** (masa depan). Hal ini menghindari kebocoran data (*look-ahead bias*) yang sering terjadi jika menggunakan split acak pada data deret waktu.
2. **Pipeline Preprocessing**:
   - Nilai hilang pada fitur diimputasi menggunakan median (`SimpleImputer`).
   - Standardisasi fitur dilakukan menggunakan `StandardScaler` khusus untuk model Linear Regression.
   - Semua tahapan preprocessing dibungkus dalam `sklearn.pipeline.Pipeline` agar parameter statistik dihitung **hanya dari data training**.

---

## 🤖 Algoritma yang Digunakan

1. **Linear Regression** (dengan `StandardScaler`)
2. **Random Forest Regressor** (`n_estimators=150`, `random_state=42`)
3. **Gradient Boosting Regressor** (`random_state=42`)

---

## 📈 Hasil Evaluasi & Perbandingan Model

Evaluasi model dilakukan pada 20% data uji (*unseen test set*) menggunakan 4 metrik regresi:
- **MAE** (*Mean Absolute Error*) — lebih rendah lebih baik
- **MSE** (*Mean Squared Error*) — lebih rendah lebih baik
- **RMSE** (*Root Mean Squared Error*) — lebih rendah lebih baik
- **R²** (*Coefficient of Determination*) — lebih tinggi lebih baik

| Peringkat | Model | MAE | MSE | RMSE | R² |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Linear Regression** | **0.3442** | **0.2714** | **0.5209** | **0.8499** |
| 🥈 | **Random Forest** | 0.3630 | 0.2916 | 0.5400 | 0.8397 |
| 🥉 | **Gradient Boosting** | 0.3968 | 0.3280 | 0.5727 | 0.8185 |

### 💡 Analisis Temuan:
- **Linear Regression** menjadi model terbaik dengan error terendah (MAE = 0.3442) dan koefisien determinasi tertinggi (R² = 0.8499).
- Nilai **R² = 0.8499** menunjukkan bahwa model mampu menjelaskan sekitar **84.99% variasi konsentrasi CO** pada data uji.
- Model linier unggul karena respons sensor semikonduktor oksida logam pada dataset ini memiliki hubungan fisik yang cenderung linier/proporsional terhadap konsentrasi gas referensi, sedangkan model berbasis pohon (*tree-based*) lebih rentan terhadap *noise* fluktuasi sensor saat diuji pada data masa depan.

---

## 📁 Struktur Direktori

```text
air-quality-prediction/
├── air_quality_ml.ipynb        # Notebook analisis, pelatihan, dan visualisasi
├── AirQualityUCI.csv           # File dataset mentah
├── hasil_air_quality/          # Folder output hasil
│   ├── model_air_quality_terbaik.joblib   # File model tersimpan
│   └── hasil_prediksi_test.csv            # Prediksi aktual vs prediksi
└── README.md                   # Dokumentasi proyek