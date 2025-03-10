import streamlit as st
import requests
import time
import pandas as pd
import numpy as np

# FastAPI URL
API_URL = "http://127.0.0.1:8000/predict/"

# Available models
models = ["RidgeRegression", "LassoRegression", "RandomForest"]

# Streamlit UI
st.title("🏢 Smart Building Energy Monitoring & Optimization")
st.write("Live energy predictions per floor & room with optimization insights.")

# User selects model
selected_model = st.selectbox("Select Prediction Model:", models)

# Simulated data storage
live_data = []
total_energy_kwh = 0  # Track total energy consumption

# Function to generate synthetic building data
def generate_synthetic_data():
    floor = np.random.randint(1, 11)  # 10 floors
    room = np.random.randint(1, 6)  # 5 rooms per floor
    occupancy = np.random.randint(0, 5)  # Number of people in the room
    device_usage = np.random.randint(max(occupancy, 1), occupancy + 3)  # Devices >= occupancy
    temperature = round(np.random.uniform(18, 30), 2)
    humidity = round(np.random.uniform(30, 70), 2)
    windspeed = round(np.random.uniform(0, 10), 2)
    visibility = round(np.random.uniform(10, 100), 2)
    
    return {
        "floor": floor,
        "room": room,
        "occupancy": occupancy,
        "device_usage": device_usage,
        "temperature": temperature,
        "humidity": humidity,
        "windspeed": windspeed,
        "visibility": visibility
    }

# Function to calculate energy consumption (kWh)
def calculate_kwh(predictions, interval_minutes=2):
    return sum(predictions) * (interval_minutes / 60) / 1000  # Convert W to kWh

# Function to generate energy-saving suggestions
def get_energy_tips(occupancy, device_usage):
    if occupancy == 1 and device_usage > 2:
        return "⚡ Too many devices for one person. Turn off a fan or light!"
    elif occupancy > 3 and device_usage < occupancy:
        return f"🔋 More people but fewer devices on? Ensure proper lighting & cooling."
    elif device_usage > 5:
        return "⚠️ High device usage! Consider turning off unnecessary appliances."
    else:
        return "✅ Good energy efficiency!"

# Real-time data streaming
placeholder = st.empty()

while True:
    # Generate new building data
    data = generate_synthetic_data()
    
    # Send data to API
    response = requests.post(API_URL, json=data, params={"model_name": selected_model})
    prediction = response.json().get("predicted_energy_watts", 0)
    
    # Append to live data storage
    data["predicted_energy_watts"] = prediction
    live_data.append(data)
    
    # Compute kWh consumption
    total_energy_kwh += calculate_kwh([prediction])  # Add new reading
    
    # Convert to DataFrame for display
    df = pd.DataFrame(live_data[-10:])  # Show last 10 readings
    
    # Update UI
    with placeholder.container():
        st.write(df)
        st.line_chart(df.set_index("floor")["predicted_energy_watts"])  # Show energy per floor
        
        st.subheader("⚡ Total Energy Consumption (kWh)")
        st.write(f"{total_energy_kwh:.4f} kWh")  # Display total energy used
        
        # Display energy-saving tips
        if prediction != "N/A":
            st.subheader("💡 Optimization Suggestion:")
            st.write(get_energy_tips(data["occupancy"], data["device_usage"]))
    
    time.sleep(2)  # Update every 2 seconds
