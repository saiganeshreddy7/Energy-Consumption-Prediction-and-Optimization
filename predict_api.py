# predict_api.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional
import json

# Check if models exist
if not os.path.exists("models") or len(os.listdir("models")) == 0:
    raise Exception("Models directory is empty. Please run train_model.py first.")

# Load feature names from training
features = [
    "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5",
    "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out", "Press_mm_hg",
    "RH_out", "Windspeed", "Visibility", "Tdewpoint", "rv1", "rv2",
    "hour", "weekday", "month", "season", "is_weekend", "day_period", 
    "temp_diff_avg", "T_avg", "RH_avg"
]

# Load all trained models
print("Loading trained models...")
models = {}
model_files = [f for f in os.listdir("models") if f.endswith(".pkl") and f != "scaler.pkl"]
for model_file in model_files:
    model_name = model_file.replace(".pkl", "")
    if model_name != "scaler":  # Skip the scaler file
        models[model_name] = joblib.load(f"models/{model_file}")
        print(f"Loaded {model_name}")

# Load the scaler
scaler = joblib.load("models/scaler.pkl")

# Load performance metrics
with open("models/performance_metrics.json", "r") as f:
    performance_metrics = json.load(f)

# Sort models by performance (using R² score)
sorted_models = sorted(performance_metrics.items(), key=lambda x: x[1]["r2"], reverse=True)
best_model_name = sorted_models[0][0]
print(f"Best performing model: {best_model_name} (R² = {performance_metrics[best_model_name]['r2']:.4f})")

# Define API using FastAPI
app = FastAPI(
    title="Energy Consumption Prediction API",
    description="Predicts appliance energy consumption based on environmental data",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data model
class PredictionInput(BaseModel):
    # Temperature sensors
    T1: float = Field(..., description="Temperature in kitchen area (°C)")
    T2: float = Field(..., description="Temperature in living room area (°C)")
    T3: float = Field(..., description="Temperature in laundry room area (°C)")
    T4: float = Field(..., description="Temperature in office area (°C)")
    T5: float = Field(..., description="Temperature in bathroom (°C)")
    T6: float = Field(..., description="Temperature outside bathroom (°C)")
    T7: float = Field(..., description="Temperature in ironing room (°C)")
    T8: float = Field(..., description="Temperature in teenager room 2 (°C)")
    T9: float = Field(..., description="Temperature in parents room (°C)")
    
    # Humidity sensors
    RH_1: float = Field(..., description="Humidity in kitchen area (%)")
    RH_2: float = Field(..., description="Humidity in living room area (%)")
    RH_3: float = Field(..., description="Humidity in laundry room area (%)")
    RH_4: float = Field(..., description="Humidity in office area (%)")
    RH_5: float = Field(..., description="Humidity in bathroom (%)")
    RH_6: float = Field(..., description="Humidity outside bathroom (%)")
    RH_7: float = Field(..., description="Humidity in ironing room (%)")
    RH_8: float = Field(..., description="Humidity in teenager room 2 (%)")
    RH_9: float = Field(..., description="Humidity in parents room (%)")
    
    # Weather data
    T_out: float = Field(..., description="Outside temperature (°C)")
    RH_out: float = Field(..., description="Outside humidity (%)")
    Press_mm_hg: float = Field(..., description="Pressure (mm Hg)")
    Windspeed: float = Field(..., description="Wind speed (m/s)")
    Visibility: float = Field(..., description="Visibility (km)")
    Tdewpoint: float = Field(..., description="Dew point temperature (°C)")
    
    # Random variables from original dataset
    rv1: float = Field(..., description="Random variable 1")
    rv2: float = Field(..., description="Random variable 2")
    
    # Time-related features
    hour: int = Field(..., description="Hour of the day (0-23)", ge=0, le=23)
    weekday: int = Field(..., description="Day of the week (0=Monday, 6=Sunday)", ge=0, le=6)
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)
    
    # Optional fields for additional context
    occupancy: Optional[int] = Field(None, description="Number of people in the house", ge=0)
    device_usage: Optional[int] = Field(None, description="Number of active electrical devices", ge=0)
    lights: Optional[int] = Field(None, description="Light energy consumption (Wh)")

    class Config:
        json_schema_extra = {
            "example": {
                "T1": 20.1, "RH_1": 45.2, "T2": 19.8, "RH_2": 44.3,
                "T3": 19.7, "RH_3": 44.7, "T4": 19.5, "RH_4": 45.1,
                "T5": 18.2, "RH_5": 47.3, "T6": 17.8, "RH_6": 50.2,
                "T7": 17.9, "RH_7": 46.8, "T8": 18.3, "RH_8": 47.9,
                "T9": 18.2, "RH_9": 48.1, "T_out": 8.2, "Press_mm_hg": 733.7,
                "RH_out": 90.1, "Windspeed": 5.2, "Visibility": 62.0, "Tdewpoint": 6.1,
                "rv1": 13.3, "rv2": 13.3, "hour": 17, "weekday": 0, "month": 1,
                "occupancy": 3, "device_usage": 5, "lights": 30
            }
        }
class PredictionOutput(BaseModel):
    model: str
    predicted_appliances: float
    confidence_level: str
    energy_profile: str
    factors: List[Dict[str, float]]
    optimization_tips: List[str]

@app.get("/")
def home():
    """Returns information about the API."""
    return {
        "message": "Energy Prediction API is Running!",
        "available_models": list(models.keys()),
        "best_model": best_model_name,
        "performance_metrics": performance_metrics
    }

@app.get("/models")
def get_models():
    """Returns information about available models and their performance."""
    return {
        "models": list(models.keys()),
        "performance_metrics": performance_metrics,
        "best_model": best_model_name
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_energy(
    data: PredictionInput, 
    model_name: str = Query(best_model_name, description="Model to use for prediction")
):
    """
    Predicts energy consumption using the selected model.
    Returns the prediction along with energy profile and optimization suggestions.
    """
    if model_name not in models:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available models: {list(models.keys())}"
        )

    # Prepare input data
    input_dict = data.dictt()
    
    # Extract optional fields
    occupancy = input_dict.pop("occupancy", None)
    device_usage = input_dict.pop("device_usage", None)
    lights = input_dict.pop("lights", None)
    
    # Calculate additional features
    temp_sensors = [input_dict[f"T{i}"] for i in range(1, 10)]
    rh_sensors = [input_dict[f"RH_{i}"] for i in range(1, 10)]
    
    # Season calculation (0: winter, 1: spring, 2: summer, 3: fall)
    input_dict["season"] = input_dict["month"] % 12 // 3
    
    # Is weekend
    input_dict["is_weekend"] = 1 if input_dict["weekday"] >= 5 else 0
    
    # Day period
    hour = input_dict["hour"]
    input_dict["day_period"] = 0 if 0 <= hour < 6 else (1 if 6 <= hour < 12 else (2 if 12 <= hour < 18 else 3))
    
    # Temperature differential (indoor vs outdoor)
    input_dict["temp_diff_avg"] = input_dict["T_out"] - sum(temp_sensors) / len(temp_sensors)
    
    # Average temperature and humidity
    input_dict["T_avg"] = sum(temp_sensors) / len(temp_sensors)
    input_dict["RH_avg"] = sum(rh_sensors) / len(rh_sensors)
    
    # Create input array with all features
    try:
        input_data = np.array([[float(input_dict[feature]) for feature in features]])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Select model
    model = models[model_name]
    
    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    
    # Ensure prediction is non-negative
    prediction = max(0, prediction)
    
    # Adjust prediction based on occupancy and device usage if provided
    if occupancy is not None and device_usage is not None:
        if occupancy == 0:
            base_consumption = 20  # Minimum standby power when no one is home
            prediction = min(prediction, base_consumption + (10 * device_usage))
        else:
            # Increase prediction slightly for high occupancy or device usage
            occupancy_factor = 1.0 + (0.05 * occupancy)
            device_factor = 1.0 + (0.03 * device_usage)
            prediction = prediction * occupancy_factor * device_factor
    
    # Round prediction
    prediction = round(prediction, 2)
    
    # Determine energy profile
    if prediction < 50:
        energy_profile = "Low"
    elif prediction < 100:
        energy_profile = "Moderate"
    elif prediction < 200:
        energy_profile = "High"
    else:
        energy_profile = "Very High"
    
    # Determine confidence level
    r2_score = performance_metrics[model_name]["r2"]
    if r2_score > 0.8:
        confidence_level = "High"
    elif r2_score > 0.6:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    # Identify key contributing factors
    # For models that support feature importance
    # Identify key contributing factors
    factors = []
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        
        # Only keep numerical features in factors
        top_features = [(features[i], importances[i]) for i in sorted_idx[:5] if isinstance(importances[i], (int, float))]
        factors = [{"factor": feature, "importance": float(importance)} for feature, importance in top_features]

    # For linear models
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        top_features = [(features[i], coefs[i]) for i in sorted_idx[:5]]
        factors = [{"factor": feature, "impact": float(coef)} for feature, coef in top_features]
    
    # Generate optimization tips
    tips = []
    
    # Temperature related tips
    if input_dict["temp_diff_avg"] < -5:  # Indoor much warmer than outdoor
        tips.append("Consider reducing heating - indoor temperature is significantly higher than outdoor.")
    elif input_dict["temp_diff_avg"] > 5:  # Indoor much cooler than outdoor
        tips.append("Consider reducing cooling - indoor temperature is significantly lower than outdoor.")
    
    # Time of day tips
    if 17 <= hour <= 21:  # Peak hours
        tips.append("Current time (17:00-21:00) is typically peak energy pricing - consider delaying high-consumption activities.")
    
    # Humidity related tips
    if input_dict["RH_avg"] > 60:
        tips.append("Indoor humidity is high - this may cause increased energy usage for dehumidifiers or AC.")
    
    # Occupancy tips
    if occupancy is not None and device_usage is not None:
        if device_usage > occupancy + 2:
            tips.append(f"Device usage ({device_usage}) seems high for current occupancy ({occupancy}) - check for unused devices.")
    
    # Season specific tips
    if input_dict["season"] == 0:  # Winter
        tips.append("Winter: Ensure proper insulation around windows and doors to reduce heating costs.")
    elif input_dict["season"] == 2:  # Summer
        tips.append("Summer: Consider using fans instead of air conditioning when possible.")
    
    # If no specific tips are generated, add general tips
    if not tips:
        tips = [
            "Use energy-efficient LED lighting to reduce electricity consumption.",
            "Unplug devices when not in use to eliminate phantom power draw.",
            "Consider using smart power strips to manage multiple devices."
        ]
    
    return {
        "model": model_name,
        "predicted_appliances": prediction,
        "confidence_level": confidence_level,
        "energy_profile": energy_profile,
        "factors": factors,
        "optimization_tips": tips
    }
