import streamlit as st
import requests
import time
import pandas as pd
import numpy as np

# FastAPI URL
API_URL = "http://127.0.0.1:8000/predict/"

# Available models
models = ["LinearRegression", "SGDRegressor", "RidgeRegression", "LassoRegression", "KNN"]

# Streamlit UI
st.title("🔌 Real-Time Energy Consumption Prediction & Optimization")
st.write("Streaming live predictions and energy-saving insights...")

# User selects model
selected_model = st.selectbox("Select Prediction Model:", models)

# Simulated data storage
live_data = []
total_energy_kwh = 0  # Track total energy consumption

# Function to generate synthetic data with additional features
def generate_synthetic_data():
    occupancy = np.random.randint(1, 6)  # Number of people in the room
    device_usage = np.random.randint(occupancy, occupancy + 3) # Number of active electrical devices
    return {
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
        "hour": time.localtime().tm_hour,
        "weekday": time.localtime().tm_wday,
        "month": time.localtime().tm_mon,
        "occupancy": occupancy,
        "device_usage": device_usage
    }

# Function to calculate energy consumption (kWh)
def calculate_kwh(predictions, interval_minutes=2):
    return sum(predictions) * (interval_minutes / 60) / 1000  # Convert W to kWh

# Function to generate energy-saving suggestions
def get_energy_tips(appliances, occupancy, device_usage):
    if appliances > 500:
        return "⚡ Very High Usage! Reduce unnecessary appliance use immediately."
    elif appliances > 400:
        return "⚠️ High Usage! Turn off idle devices and optimize heating/cooling."
    elif appliances > 200:
        return f"🔋 Moderate Usage! You have {device_usage} active devices. Consider switching to energy-efficient ones."
    elif occupancy > 3 and appliances > 150:
        return f"👥 High occupancy detected ({occupancy} people). Ensure energy usage is optimized."
    else:
        return "✅ Good Usage! Keep maintaining efficiency."

# Real-time data streaming
placeholder = st.empty()

while True:
    # Generate new data
    data = generate_synthetic_data()
    
    # Send data to API
    response = requests.post(API_URL, json=data, params={"model_name": selected_model})
    prediction = response.json().get("predicted_appliances", 0)
    
    # Append to live data storage
    data["predicted_appliances"] = prediction
    live_data.append(data)
    
    # Compute kWh consumption
    total_energy_kwh += calculate_kwh([prediction])  # Add new reading
    
    # Convert to DataFrame for display
    df = pd.DataFrame(live_data[-10:])  # Show last 10 readings
    
    # Update UI
    with placeholder.container():
        st.write(df)
        st.line_chart(df.set_index("hour")["predicted_appliances"])  # Show trendline
        
        st.subheader("⚡ Total Energy Consumption (kWh)")
        st.write(f"{total_energy_kwh:.4f} kWh")  # Display total energy used
        
        # Display energy-saving tips
        if prediction != "N/A":
            st.subheader("💡 Optimization Suggestion:")
            st.write(get_energy_tips(prediction, data["occupancy"], data["device_usage"]))
    
    time.sleep(2)  # Update every 2 seconds
