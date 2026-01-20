from fastapi import FastAPI
import joblib
import json
import numpy as np

app = FastAPI(title="LoL Win Prediction API")

# Load model and feature order
model = joblib.load("model.pkl")

with open("features.json") as f:
    FEATURES = json.load(f)


@app.post("/predict")
def predict(data: dict):
    """
    Expects JSON:
    {
        "killDiff": 3,
        "deathDiff": -1,
        ...
    }
    """

    try:
        x = [data[feature] for feature in FEATURES]
    except KeyError as e:
        return {"error": f"Missing feature: {e}"}

    prediction = model.predict([x])[0]
    probability = model.predict_proba([x])[0][1]

    return {
        "blueWins": int(prediction),
        "probability": float(probability)
    }
