import numpy as np
import pandas as pd
import time
import requests
from datetime import datetime

# FastAPI URL
API_URL = "http://127.0.0.1:8000/predict"

# Function to generate synthetic data with expanded features
def generate_synthetic_data():
    """Simulates live energy consumption data with expanded features."""
    timestamp = pd.Timestamp.now()
    current_time = datetime.now()
    
    # Time information
    hour = current_time.hour
    weekday = current_time.weekday()
    month = current_time.month
    
    # Determine realistic occupancy based on time
    if 0 <= hour < 6:  # Night
        occupancy = np.random.randint(0, 2)
    elif 6 <= hour < 9:  # Morning
        occupancy = np.random.randint(1, 5)
    elif 9 <= hour < 17:  # Day
        occupancy = np.random.randint(0, 3)
    else:  # Evening
        occupancy = np.random.randint(2, 6)
    
    # Determine realistic device usage based on occupancy
    if occupancy == 0:
        device_usage = np.random.randint(0, 3)
    else:
        device_usage = np.random.randint(occupancy, occupancy + 5)
    
    # Base temperatures and humidity
    base_temp_in = 20  # Default room temperature
    base_humidity_in = 50  # Default indoor humidity
    
    # Adjust outdoor temperature by season
    season = month % 12 // 3  # 0: winter, 1: spring, 2: summer, 3: fall
    if season == 0:  # Winter
        base_temp_out = np.random.uniform(-5, 10)
        base_humidity_out = np.random.uniform(70, 95)
    elif season == 1:  # Spring
        base_temp_out = np.random.uniform(10, 20)
        base_humidity_out = np.random.uniform(50, 80)
    elif season == 2:  # Summer
        base_temp_out = np.random.uniform(20, 35)
        base_humidity_out = np.random.uniform(40, 70)
    else:  # Fall
        base_temp_out = np.random.uniform(5, 20)
        base_humidity_out = np.random.uniform(60, 85)
    
    # Room temperature and humidity variations
    temp_variance = 1.5
    temps = [round(base_temp_in + np.random.uniform(-temp_variance, temp_variance), 2) for _ in range(9)]
    humidities = [round(base_humidity_in + np.random.uniform(-10, 10), 2) for _ in range(9)]
    
    # Lights consumption based on hour and occupancy
    if 7 <= hour <= 21 and occupancy > 0:
        lights = int(occupancy * 10 * np.random.uniform(0.7, 1.3))
    else:
        lights = int(5 * np.random.uniform(0, 1.0))
    
    data = {
        # Temperature sensors
        "T1": temps[0], "T2": temps[1], "T3": temps[2], 
        "T4": temps[3], "T5": temps[4], "T6": temps[5],
        "T7": temps[6], "T8": temps[7], "T9": temps[8],
        
        # Humidity sensors
        "RH_1": humidities[0], "RH_2": humidities[1], "RH_3": humidities[2],
        "RH_4": humidities[3], "RH_5": humidities[4], "RH_6": humidities[5],
        "RH_7": humidities[6], "RH_8": humidities[7], "RH_9": humidities[8],
        
        # Weather data
        "T_out": round(base_temp_out, 2),
        "RH_out": round(base_humidity_out, 2),
        "Press_mm_hg": round(np.random.uniform(730, 760), 2),
        "Windspeed": round(np.random.uniform(0, 15), 2),
        "Visibility": round(np.random.uniform(10, 100), 2),
        "Tdewpoint": round(base_temp_out - np.random.uniform(2, 8), 2),
        
        # Random variables
        "rv1": round(np.random.uniform(10, 30), 2),
        "rv2": round(np.random.uniform(10, 30), 2),
        
        # Time information
        "hour": hour,
        "weekday": weekday,
        "month": month,
        
        # Additional context (optional for API, useful for logging)
        "occupancy": occupancy,
        "device_usage": device_usage,
        "lights": lights
    }
    
    # Calculate additional features used by the model
    # Season calculation
    data["season"] = data["month"] % 12 // 3
    
    # Is weekend
    data["is_weekend"] = 1 if data["weekday"] >= 5 else 0
    
    # Day period
    data["day_period"] = 0 if 0 <= data["hour"] < 6 else (1 if 6 <= data["hour"] < 12 else (2 if 12 <= data["hour"] < 18 else 3))
    
    # Temperature differential
    temp_sensors = [data[f"T{i}"] for i in range(1, 10)]
    data["temp_diff_avg"] = data["T_out"] - sum(temp_sensors) / len(temp_sensors)
    
    # Average temperature and humidity
    data["T_avg"] = sum(temp_sensors) / len(temp_sensors)
    rh_sensors = [data[f"RH_{i}"] for i in range(1, 10)]
    data["RH_avg"] = sum(rh_sensors) / len(rh_sensors)
    
    return data

# Function to store historical data
def store_historical_data(data, prediction, filename="energy_history.csv"):
    """Store data and predictions to a CSV file for historical analysis."""
    timestamp = pd.Timestamp.now()
    
    # Extract key information
    record = {
        "timestamp": timestamp,
        "energy": prediction["predicted_appliances"],
        "occupancy": data["occupancy"],
        "device_usage": data["device_usage"],
        "temp_indoor": data["T_avg"],
        "temp_outdoor": data["T_out"],
        "humidity_indoor": data["RH_avg"],
        "humidity_outdoor": data["RH_out"],
        "energy_profile": prediction["energy_profile"],
        "confidence": prediction["confidence_level"]
    }
    
    # Create DataFrame with a single row
    record_df = pd.DataFrame([record])
    
    try:
        # Check if file exists and append or create new
        try:
            existing_df = pd.read_csv(filename, parse_dates=["timestamp"])
            updated_df = pd.concat([existing_df, record_df], ignore_index=True)
        except FileNotFoundError:
            updated_df = record_df
            
        # Save to CSV
        updated_df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

# Simulating real-time data feed
if __name__ == "__main__":
    print("Starting energy consumption data generator...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Generate synthetic data
            synthetic_data = generate_synthetic_data()
            print(f"Generated Data: {synthetic_data['hour']}h, {synthetic_data['T_avg']:.1f}°C indoor, {synthetic_data['RH_avg']:.1f}% RH")
            
            # Send data to API
            try:
                response = requests.post(API_URL, json=synthetic_data, params={"model_name": "RandomForest"})
                if response.status_code == 200:
                    prediction = response.json()
                    print(f"Prediction: {prediction['predicted_appliances']:.2f} Wh - {prediction['energy_profile']} consumption")
                    
                    # Store historical data
                    store_historical_data(synthetic_data, prediction)
                else:
                    print("Error in API request:", response.text)
            except Exception as e:
                print(f"API request failed: {e}")
            
            time.sleep(10)  # Generate new data every 10 seconds
    
    except KeyboardInterrupt:
        print("\nData generator stopped.")