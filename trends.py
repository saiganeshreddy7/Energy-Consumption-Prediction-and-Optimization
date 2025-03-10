import streamlit as st
import pandas as pd
import plotly.express as px
import time

# Load historical data
csv_file = "historical_energy_data.csv"

st.title("📊 Energy Trends & Insights")

# Auto-refresh every 10 seconds
while True:
    df = pd.read_csv(csv_file)

    st.subheader("🏢 Floor-Wise Energy Consumption")
    floor_trend = df.groupby("floor")["predicted_appliances"].mean().reset_index()
    fig1 = px.bar(floor_trend, x="floor", y="predicted_appliances", title="Average Energy Consumption per Floor")
    st.plotly_chart(fig1)

    st.subheader("🛋 Room-Wise Energy Consumption")
    room_trend = df.groupby("room")["predicted_appliances"].mean().reset_index()
    fig2 = px.bar(room_trend, x="room", y="predicted_appliances", title="Average Energy Consumption per Room")
    st.plotly_chart(fig2)

    st.subheader("📈 Energy Consumption Over Time")
    if len(df) > 20:  # Only show if there is enough data
        time_series = df.tail(50)  # Last 50 readings
        fig3 = px.line(time_series, x="timestamp", y="predicted_appliances", title="Energy Usage Over Time")
        st.plotly_chart(fig3)

    st.subheader("📉 Energy Optimization Trends")
    opt_trend = df.groupby("occupancy")["predicted_appliances"].mean().reset_index()
    fig4 = px.line(opt_trend, x="occupancy", y="predicted_appliances", title="Energy Consumption vs Occupancy")
    st.plotly_chart(fig4)

    time.sleep(10)  # Refresh every 10 seconds
    st.experimental_rerun()  # Auto-refresh the page
