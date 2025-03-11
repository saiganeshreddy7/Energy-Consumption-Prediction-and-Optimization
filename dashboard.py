# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Energy Consumption Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load theme CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #34495e;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #3498db;
    }
    .metric-label {
        font-size: 16px;
        color: #7f8c8d;
    }
    .high-energy {
        color: #e74c3c;
    }
    .medium-energy {
        color: #f39c12;
    }
    .low-energy {
        color: #27ae60;
    }
    .tip-card {
        background-color: #eafaf1;
        border-left: 5px solid #27ae60;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .section-divider {
        margin-top: 30px;
        margin-bottom: 30px;
        border-top: 1px solid #ecf0f1;
    }
</style>
""", unsafe_allow_html=True)

# API settings
API_URL = "http://127.0.0.1:8000"

# Function to call the API
def call_prediction_api(data, model_name):
    try:
        response = requests.post(f"{API_URL}/predict", json=data, params={"model_name": model_name})
        return response.json()
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return None

# Function to get available models
def get_available_models():
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get models: {response.text}")
            return {"models": ["LinearRegression"], "best_model": "LinearRegression"}
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return {"models": ["LinearRegression"], "best_model": "LinearRegression"}

# Function to generate synthetic data with expanded features
def generate_synthetic_data(scenario=None, occupancy=None, device_usage=None, hour=None):
    """Simulates live energy consumption data with expanded features."""
    current_time = datetime.now()
    
    if hour is None:
        hour = current_time.hour
        
    # Default values
    if occupancy is None:
        # Realistic occupancy pattern based on time of day
        if 0 <= hour < 6:  # Night
            occupancy = np.random.randint(0, 2)
        elif 6 <= hour < 9:  # Morning
            occupancy = np.random.randint(1, 5)
        elif 9 <= hour < 17:  # Day
            occupancy = np.random.randint(0, 3)
        else:  # Evening
            occupancy = np.random.randint(2, 6)
    
    if device_usage is None:
        # Base device usage on occupancy
        if occupancy == 0:
            device_usage = np.random.randint(0, 3)  # Low when nobody home
        else:
            device_usage = np.random.randint(occupancy, occupancy + 5)  # More devices than people
    
    # Base temperatures - different scenarios
    base_temp_in = 20  # Default room temperature
    base_humidity_in = 50  # Default indoor humidity
    
    # Weather/outdoor conditions based on month
    month = current_time.month
    weekday = current_time.weekday()
    
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
    
    # Scenario-specific adjustments
    if scenario == "energy_efficient":
        # Energy efficient home - good insulation, moderate temperature
        temp_variance = 1.0  # Low variance between rooms
        base_temp_in = 20 if season in [0, 3] else 24  # Moderate heating/cooling
        device_efficiency = 0.8  # More efficient devices
        
    elif scenario == "high_consumption":
        # High consumption - large temperature differentials, many devices
        temp_variance = 3.0  # High variance between rooms
        base_temp_in = 23 if season in [0, 3] else 19  # More extreme heating/cooling
        device_usage = max(device_usage, 8)  # More devices running
        device_efficiency = 1.2  # Less efficient devices
        
    elif scenario == "unoccupied":
        # Nobody home - minimal device usage
        occupancy = 0
        device_usage = np.random.randint(0, 3)  # Only essential devices
        temp_variance = 0.5  # Lower variance as HVAC is likely off or reduced
        device_efficiency = 1.0
    # Completing the dashboard.py file where it was cut off:
    elif scenario == "peak_evening":
        # Peak evening usage - high occupancy, many devices
        hour = hour if hour is not None else np.random.randint(18, 22)
        occupancy = max(occupancy, np.random.randint(3, 6))
        device_usage = np.random.randint(occupancy + 3, occupancy + 8)
        temp_variance = 2.0
        device_efficiency = 1.1
    else:
        # Default behavior
        temp_variance = 1.5
        device_efficiency = 1.0
    
    # Generate temperature and humidity values with variance between rooms
    temps = [round(base_temp_in + np.random.uniform(-temp_variance, temp_variance), 2) for _ in range(9)]
    humidities = [round(base_humidity_in + np.random.uniform(-10, 10), 2) for _ in range(9)]
    
    # Lights consumption based on hour and occupancy
    if 7 <= hour <= 21 and occupancy > 0:
        lights = int(occupancy * 10 * np.random.uniform(0.7, 1.3))
    else:
        lights = int(5 * np.random.uniform(0, 1.0))  # Minimal usage at night
    
    # Create data dictionary
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
        
        # Additional context
        "occupancy": occupancy,
        "device_usage": device_usage,
        "lights": lights
    }
    
    return data

# Function to generate historical data for trends
def generate_historical_data(days=7, interval_hours=1):
    """Generate synthetic historical data for trend analysis."""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Create time range
    times = []
    current = start_time
    while current <= end_time:
        times.append(current)
        current += timedelta(hours=interval_hours)
    
    # Generate data for each time point
    data_points = []
    for t in times:
        data = generate_synthetic_data(
            hour=t.hour,
            occupancy=None,  # Let the function determine based on time
            device_usage=None
        )
        
        # Call API to get prediction
        api_data = {k: v for k, v in data.items() if k not in ["occupancy", "device_usage", "lights"]}
        try:
            response = requests.post(f"{API_URL}/predict", json=api_data, params={"model_name": "RandomForest"})
            if response.status_code == 200:
                prediction = response.json()
                energy = prediction["predicted_appliances"]
            else:
                energy = np.random.uniform(30, 200)  # Fallback if API fails
        except:
            energy = np.random.uniform(30, 200)  # Fallback if API fails
        
        data_points.append({
            "timestamp": t,
            "energy": energy,
            "occupancy": data["occupancy"],
            "device_usage": data["device_usage"],
            "temp_indoor": data["T_avg"] if "T_avg" in data else sum([data[f"T{i}"] for i in range(1, 10)]) / 9,
            "temp_outdoor": data["T_out"],
            "humidity_indoor": sum([data[f"RH_{i}"] for i in range(1, 10)]) / 9,
            "humidity_outdoor": data["RH_out"]
        })
    
    return pd.DataFrame(data_points)

# Main dashboard layout
def main():
    # Get available models
    model_info = get_available_models()
    available_models = model_info.get("models", ["LinearRegression"])
    best_model = model_info.get("best_model", available_models[0])
    
    # Header
    st.markdown('<div class="main-header">⚡ Smart Home Energy Consumption Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    selected_model = st.sidebar.selectbox("Select Model", options=available_models, index=available_models.index(best_model) if best_model in available_models else 0)
    
    scenario = st.sidebar.selectbox(
        "Simulation Scenario",
        options=["default", "energy_efficient", "high_consumption", "unoccupied", "peak_evening"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    # Optional manual overrides
    st.sidebar.subheader("Manual Overrides (Optional)")
    use_manual = st.sidebar.checkbox("Use manual inputs")
    
    occupancy = None
    device_usage = None
    hour = None
    
    if use_manual:
        occupancy = st.sidebar.slider("Occupancy", 0, 10, 2)
        device_usage = st.sidebar.slider("Active Devices", 0, 20, 5)
        hour = st.sidebar.slider("Hour of Day", 0, 23, datetime.now().hour)
    
    # Generate current data
    current_data = generate_synthetic_data(scenario, occupancy, device_usage, hour)
    
    # Call API for prediction
    with st.spinner("Calculating energy consumption..."):
        prediction = call_prediction_api(current_data, selected_model)
    
    # Display current metrics
    st.markdown('<div class="sub-header">Current Energy Status</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    energy_value = prediction["predicted_appliances"] if prediction else 0
    energy_profile = prediction["energy_profile"] if prediction else "Unknown"
    
    energy_class = ""
    if energy_profile == "Low":
        energy_class = "low-energy"
    elif energy_profile == "Moderate":
        energy_class = "medium-energy"
    else:
        energy_class = "high-energy"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {energy_class}">{energy_value:.1f} Wh</div>
            <div class="metric-label">Current Consumption</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{current_data["occupancy"]}</div>
            <div class="metric-label">Occupants</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{current_data["device_usage"]}</div>
            <div class="metric-label">Active Devices</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{energy_profile}</div>
            <div class="metric-label">Energy Profile</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display optimization tips
    if prediction and "optimization_tips" in prediction:
        st.markdown('<div class="sub-header">Energy Optimization Tips</div>', unsafe_allow_html=True)
        for tip in prediction["optimization_tips"]:
            st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)
    
    # Display key factors
    if prediction and "factors" in prediction and prediction["factors"]:
        st.markdown('<div class="sub-header">Key Factors Affecting Consumption</div>', unsafe_allow_html=True)
        
        factors_df = pd.DataFrame(prediction["factors"])
        factor_col = factors_df.columns[0]  # Either "factor" or "impact"
        value_col = factors_df.columns[1]  # The other column
        
        # Create bar chart
        fig = px.bar(
            factors_df, 
            x=factor_col, 
            y=value_col,
            title="Top Factors Impacting Energy Consumption",
            labels={factor_col: "Factor", value_col: "Impact"},
            color=value_col,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig)
    
    # Historical data and trends (simulated)
    st.markdown('<div class="sub-header">Energy Consumption Trends</div>', unsafe_allow_html=True)
    
    # Check if historical data exists in session state
    if 'historical_data' not in st.session_state:
        with st.spinner("Generating historical data..."):
            st.session_state.historical_data = generate_historical_data()
    
    # Generate trend visualizations
    hist_data = st.session_state.historical_data
    
    # Daily pattern
    daily_pattern = hist_data.copy()
    daily_pattern['hour'] = daily_pattern['timestamp'].dt.hour
    hourly_avg = daily_pattern.groupby('hour')['energy'].mean().reset_index()
    
    fig1 = px.line(
        hourly_avg, 
        x='hour', 
        y='energy',
        title="Average Energy Consumption by Hour of Day",
        labels={"hour": "Hour of Day", "energy": "Energy (Wh)"}
    )
    st.plotly_chart(fig1)
    
    # Weekly pattern
    hist_data['weekday'] = hist_data['timestamp'].dt.weekday
    hist_data['weekday_name'] = hist_data['weekday'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    
    weekly_avg = hist_data.groupby('weekday_name')['energy'].mean().reset_index()
    # Ensure correct order of days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg['weekday_name'] = pd.Categorical(weekly_avg['weekday_name'], categories=day_order, ordered=True)
    weekly_avg = weekly_avg.sort_values('weekday_name')
    
    fig2 = px.bar(
        weekly_avg,
        x='weekday_name',
        y='energy',
        title="Average Energy Consumption by Day of Week",
        labels={"weekday_name": "Day of Week", "energy": "Energy (Wh)"}
    )
    st.plotly_chart(fig2)
    
    # Temperature vs Energy scatter plot
    fig3 = px.scatter(
        hist_data,
        x='temp_outdoor',
        y='energy',
        title="Energy Consumption vs Outdoor Temperature",
        labels={"temp_outdoor": "Outdoor Temperature (°C)", "energy": "Energy (Wh)"},
        trendline="ols"
    )
    st.plotly_chart(fig3)
    
    # Room temperature map
    st.markdown('<div class="sub-header">Current Indoor Temperature Map</div>', unsafe_allow_html=True)
    
    # Room layout
    rooms = [
        ["T3 (Laundry)", "T2 (Living)", "T1 (Kitchen)"],
        ["T7 (Ironing)", "T6 (Bath Out)", "T5 (Bathroom)"],
        ["T8 (Teen Room)", "T9 (Parents)", "T4 (Office)"]
    ]
    
    # Create temperature heatmap data
    temp_data = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            room_id = rooms[i][j].split()[0].replace("T", "")
            temp_data[i][j] = current_data[f"T{room_id}"]
    
    # Create heatmap
    fig4 = go.Figure(data=go.Heatmap(
        z=temp_data,
        text=[[f"{temp:.1f}°C" for temp in row] for row in temp_data],
        texttemplate="%{text}",
        x=["Kitchen", "Living Room", "Laundry"],
        y=["Office/Parents", "Bathroom", "Ironing/Teen"],
        colorscale="Viridis",
        hoverongaps=False
    ))
    
    fig4.update_layout(
        title="Room Temperature Distribution (°C)",
        xaxis_title="",
        yaxis_title="",
    )
    
    st.plotly_chart(fig4)
    
    # About the dashboard
    with st.expander("About this Dashboard"):
        st.markdown("""
        This smart home energy consumption dashboard uses machine learning to predict and analyze 
        energy usage. The system monitors temperature, humidity, occupancy, and other factors across
        the home to provide real-time insights and energy-saving recommendations.
        
        The data is simulated for demonstration purposes, but uses actual ML models trained on 
        real energy consumption data.
        """)

# Run the app
if __name__ == "__main__":
    main()
        
  