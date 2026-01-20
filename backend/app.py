from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import json

app = FastAPI(title="LoL Win Prediction API")


model = joblib.load("model.pkl")
with open("features.json") as f:
    FEATURES = json.load(f)

app.mount("/static", StaticFiles(directory="../frontend"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("../frontend/index.html")


@app.post("/predict")
def predict(data: dict):
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
