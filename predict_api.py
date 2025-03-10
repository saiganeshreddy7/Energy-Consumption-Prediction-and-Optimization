from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

# Load all trained models
models = {
    "LinearRegression": joblib.load("LinearRegression.pkl"),
    "SGDRegressor": joblib.load("SGDRegressor.pkl"),
    "RidgeRegression": joblib.load("RidgeRegression.pkl"),
    "LassoRegression": joblib.load("LassoRegression.pkl"),
    "KNN": joblib.load("KNN.pkl"),
}

# Define API using FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Energy Prediction API is Running!"}

@app.post("/predict/")
def predict_energy(data: dict, model_name: str = "LinearRegression"):
    """
    Accepts input JSON and predicts energy consumption using the selected model.
    Default model: LinearRegression
    """
    if model_name not in models:
        return {"error": f"Model '{model_name}' not found. Available models: {list(models.keys())}"}

    model = models[model_name]

    # Extract input values
    input_data = np.array([
        data["T1"], data["RH_1"], data["T2"], data["RH_2"], data["T3"], data["RH_3"], 
        data["T4"], data["RH_4"], data["T5"], data["RH_5"], data["T6"], data["RH_6"],
        data["T7"], data["RH_7"], data["T8"], data["RH_8"], data["T9"], data["RH_9"], 
        data["T_out"], data["Press_mm_hg"], data["RH_out"], data["Windspeed"], 
        data["Visibility"], data["Tdewpoint"], data["rv1"], data["rv2"],
        data["hour"], data["weekday"], data["month"]
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)
    
    return {"model": model_name, "predicted_appliances": prediction[0]}
