from fastapi import FastAPI
import joblib
import numpy as np

# Load all trained models
models = {
    "RidgeRegression": joblib.load("RidgeRegression.pkl"),
    "LassoRegression": joblib.load("LassoRegression.pkl"),
    "RandomForest": joblib.load("RandomForest.pkl"),
}

# Define API using FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Smart Building Energy Prediction API is Running!"}

@app.post("/predict/")
def predict_energy(data: dict, model_name: str = "RandomForest"):
    """
    Predicts energy consumption per room based on floor, occupancy, and device usage.
    """
    if model_name not in models:
        return {"error": f"Model '{model_name}' not found. Available models: {list(models.keys())}"}

    model = models[model_name]

    # Extract input values
    input_data = np.array([
        data["floor"], data["room"], data["occupancy"], data["device_usage"],
        data["temperature"], data["humidity"], data["windspeed"], data["visibility"]
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Ensure prediction is non-negative
    prediction = max(0, round(prediction, 2))

    return {
        "model": model_name,
        "floor": data["floor"],
        "room": data["room"],
        "predicted_energy_watts": prediction
    }
