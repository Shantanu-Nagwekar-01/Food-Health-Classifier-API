from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and scaler

model = joblib.load("./Model/best_food_health_classifier_xgb.pkl")
scaler = joblib.load("./Scaler/feature_scaler.pkl")

# Expected feature order

FEATURES = [
"Calories", "Protein", "Carbohydrates", "Fat", "Fiber",
"Sugars", "Sodium", "Cholesterol", "diabetes", "obesity", "bp"
]

@app.route("/", methods=["GET"])
def home():
    return "XGBoost: Best Food Health Classifier API is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON
        data = request.get_json()

        # Validate input
        if not all(f in data for f in FEATURES):
            return jsonify({"error": "Missing one or more required features."}), 400

        # Convert to DataFrame
        df = pd.DataFrame([data], columns=FEATURES)

        # Scale data
        scaled = scaler.transform(df)

        # Predict with XGBoost
        pred = int(model.predict(scaled)[0])
        prob = model.predict_proba(scaled)[0]

        # Prepare JSON response
        return jsonify({
            "prediction": pred,
            "probability": {
                "healthy": round(float(prob[1]), 4),
                "unhealthy": round(float(prob[0]), 4)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
