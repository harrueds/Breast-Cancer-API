from flask import Flask, request, Response
import joblib
import logging
import pandas as pd
import json

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/app_breast_cancer.log"),
        logging.StreamHandler(),
    ],
)

# ============================================================================
# FLASK APP + MODEL LOADING
# ============================================================================

app = Flask(__name__)

model = joblib.load("models/model_breast_cancer.pkl")
logging.info("=" * 60)
logging.info("Classification model loaded successfully into memory")
logging.info("=" * 60)


def response_json(data, status=200):
    """Standard JSON response helper (UTF-8 safe)."""
    return Response(
        response=json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        status=status,
        mimetype="application/json; charset=utf-8",
    )


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    logging.info("=" * 60)
    logging.info("Health check requested")
    logging.info("=" * 60)
    return response_json({"status": "OK", "message": "API online and waiting"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Accepts:
    - {"features": [30 values]}
    - {"feature_name": value, ...} (30 total)

    Returns:
    - prediction label + probability
    """
    try:
        # 1. JSON validation
        data = request.get_json()
        if not data:
            logging.info("=" * 60)
            logging.warning("Empty JSON payload")
            logging.info("=" * 60)
            return response_json({"error": "Expected JSON with data"}, 400)

        df = None

        # 2. Format 1: list of features
        if "features" in data:
            features = data.get("features")

            if not isinstance(features, list) or len(features) != 30:
                logging.info("=" * 60)
                logging.warning("Invalid feature list length")
                logging.info("=" * 60)
                return response_json(
                    {"error": "Expected 30 values in 'features' list"}, 400
                )

            df = pd.DataFrame([features])

        # 3. Format 2: named feature dictionary
        else:
            df = pd.DataFrame([data])

            if df.shape[1] != 30:
                logging.info("=" * 60)
                logging.warning("Invalid named feature count")
                logging.info("=" * 60)
                return response_json({"error": "Expected 30 named features"}, 400)

        # 4. Model inference
        pred = model.predict(df)[0]
        prob = round(model.predict_proba(df)[0][pred], 4)

        result = "Malignant" if pred == 1 else "Benign"

        logging.info("=" * 60)
        logging.info(f"Prediction: {result} | prob={prob}")
        logging.info("=" * 60)
        return response_json({"prediction": result, "probability": float(prob)})

    except Exception as e:
        logging.info("=" * 60)
        logging.error(f"Prediction error: {type(e).__name__}: {e}")
        logging.info("=" * 60)

        return response_json({"error": "Internal server error"}, 500)


def run():
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    run()
