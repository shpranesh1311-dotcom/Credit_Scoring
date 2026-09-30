from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np

app = Flask(__name__)
CORS(app)


# ==========================================
# LOAD FINAL CLEAN MODEL
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "credit_scoring_final_model.pkl"
)

model = joblib.load(MODEL_PATH)


# ==========================================
# FINAL PREDICTION THRESHOLD
# ==========================================

THRESHOLD = 0.65


# ==========================================
# MODEL INFORMATION
# ==========================================

FEATURE_NAMES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]


# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/")
def home():

    return jsonify({
        "message": "Credit Scoring API is running!",
        "model": "Final Clean Random Forest",
        "features": 10,
        "threshold": THRESHOLD
    })


# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # ----------------------------------
        # Extract features in EXACT order
        # used during model training
        # ----------------------------------

        features = [

            data["RevolvingUtilizationOfUnsecuredLines"],

            data["age"],

            data["NumberOfTime30-59DaysPastDueNotWorse"],

            data["DebtRatio"],

            data["MonthlyIncome"],

            data["NumberOfOpenCreditLinesAndLoans"],

            data["NumberOfTimes90DaysLate"],

            data["NumberRealEstateLoansOrLines"],

            data["NumberOfTime60-89DaysPastDueNotWorse"],

            data["NumberOfDependents"]

        ]


        # ----------------------------------
        # Convert to NumPy array
        # ----------------------------------

        features_array = np.array(
            features,
            dtype=float
        ).reshape(1, -1)


        # ----------------------------------
        # Check feature count
        # ----------------------------------

        if features_array.shape[1] != 10:

            return jsonify({
                "error": "Invalid number of features"
            }), 400


        # ----------------------------------
        # Calculate probability
        # ----------------------------------

        probability = model.predict_proba(
            features_array
        )[0][1]


        probability = float(probability)


        # ----------------------------------
        # Apply 0.65 threshold
        # ----------------------------------

        if probability >= THRESHOLD:

            prediction = 1

        else:

            prediction = 0


        # ----------------------------------
        # Determine risk level
        # ----------------------------------

        if probability < 0.30:

            risk_level = "Low Risk"

        elif probability < THRESHOLD:

            risk_level = "Medium Risk"

        else:

            risk_level = "High Risk"


        # ----------------------------------
        # Return JSON response
        # ----------------------------------

        return jsonify({

            "default_probability": round(
                probability,
                4
            ),

            "prediction": prediction,

            "risk_level": risk_level,

            "threshold": THRESHOLD

        })


    except KeyError as e:

        return jsonify({

            "error": f"Missing field: {str(e)}"

        }), 400


    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 400


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )