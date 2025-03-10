import numpy as np
import pandas as pd
import time
import requests

# FastAPI URL
API_URL = "http://127.0.0.1:8000/predict/"

# Function to generate synthetic data with expanded features
def generate_synthetic_data():
    """Simulates live energy consumption data with expanded features."""
    timestamp = pd.Timestamp.now()
    
    data = {
        "T1": round(np.random.uniform(18, 30), 2),
        "RH_1": round(np.random.uniform(30, 70), 2),
        "T2": round(np.random.uniform(18, 30), 2),
        "RH_2": round(np.random.uniform(30, 70), 2),
        "T3": round(np.random.uniform(18, 30), 2),
        "RH_3": round(np.random.uniform(30, 70), 2),
        "T4": round(np.random.uniform(18, 30), 2),
        "RH_4": round(np.random.uniform(30, 70), 2),
        "T5": round(np.random.uniform(18, 30), 2),
        "RH_5": round(np.random.uniform(30, 70), 2),
        "T6": round(np.random.uniform(18, 30), 2),
        "RH_6": round(np.random.uniform(30, 70), 2),
        "T7": round(np.random.uniform(18, 30), 2),
        "RH_7": round(np.random.uniform(30, 70), 2),
        "T8": round(np.random.uniform(18, 30), 2),
        "RH_8": round(np.random.uniform(30, 70), 2),
        "T9": round(np.random.uniform(18, 30), 2),
        "RH_9": round(np.random.uniform(30, 70), 2),
        "T_out": round(np.random.uniform(10, 25), 2),
        "Press_mm_hg": round(np.random.uniform(720, 750), 2),
        "RH_out": round(np.random.uniform(20, 100), 2),
        "Windspeed": round(np.random.uniform(0, 10), 2),
        "Visibility": round(np.random.uniform(10, 100), 2),
        "Tdewpoint": round(np.random.uniform(0, 10), 2),
        "rv1": round(np.random.uniform(5, 20), 2),
        "rv2": round(np.random.uniform(5, 20), 2),
        "hour": timestamp.hour,
        "weekday": timestamp.weekday(),
        "month": timestamp.month
    }
    return data

# Simulating real-time data feed
if __name__ == "__main__":
    while True:
        synthetic_data = generate_synthetic_data()
        print(f"Generated Data: {synthetic_data}")

        # Send data to API
        response = requests.post(API_URL, json=synthetic_data, params={"model_name": "KNN"})
        if response.status_code == 200:
            prediction = response.json()
            print(f"Prediction Response: {prediction}")
        else:
            print("Error in API request:", response.text)
        
        time.sleep(2)  # Generate new data every 2 seconds
