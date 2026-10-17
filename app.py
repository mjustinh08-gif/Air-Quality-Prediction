import time
import logging
from pathlib import Path
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# 1. SETUP MONITORING & LOGGING
logging.basicConfig(
    filename="monitoring.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)

# 2. INISIALISASI APLIKASI FLASK
app = Flask(__name__)

# 3. MEMUAT MODEL ARTIFACT
MODEL_PATH = Path("hasil_air_quality/model_air_quality_terbaik.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan di {MODEL_PATH}!")

artifact = joblib.load(MODEL_PATH)

# Mendukung jika model disimpan dalam dict artifact atau langsung estimator
if isinstance(artifact, dict) and "model" in artifact:
    model = artifact["model"]
    feature_names = artifact.get("features", None)
    model_name = artifact.get("model_name", "Best Regressor")
else:
    model = artifact
    feature_names = None
    model_name = "Best Regressor"

print(f"🚀 Model '{model_name}' berhasil dimuat!")
if feature_names:
    print(f"📋 Jumlah fitur yang diharapkan: {len(feature_names)}")


# 4. ENDPOINT MONITORING: HEALTH CHECK
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_name": model_name,
        "features_count": len(feature_names) if feature_names else "unknown"
    }), 200


# 5. ENDPOINT DEPLOYMENT: PREDIKSI CO
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    
    try:
        req_json = request.get_json(force=True)
        if "data" not in req_json:
            return jsonify({"error": "Format JSON salah! Gunakan key 'data'"}), 400
        
        raw_data = req_json["data"]
        
        # Konversi ke DataFrame jika daftar fitur diketahui, atau ke numpy array
        if feature_names and isinstance(raw_data, list):
            # Jika user mengirim 2D array baris x fitur
            if len(raw_data) > 0 and len(raw_data[0]) == len(feature_names):
                input_df = pd.DataFrame(raw_data, columns=feature_names)
            else:
                input_df = np.array(raw_data)
        else:
            input_df = np.array(raw_data)

        # Lakukan prediksi
        predictions = model.predict(input_df)
        
        # Hitung waktu respons (Latency Monitoring)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        # Catat ke log monitoring
        logging.info(f"PREDICT_SUCCESS | Rows: {len(input_df)} | Latency: {latency_ms}ms | Pred: {predictions.tolist()[:3]}")

        return jsonify({
            "status": "success",
            "model_used": model_name,
            "latency_ms": latency_ms,
            "prediction_co_concentration": predictions.tolist()
        }), 200

    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logging.error(f"PREDICT_ERROR | Latency: {latency_ms}ms | Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("🌍 Server Flask Air Quality berjalan di http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)