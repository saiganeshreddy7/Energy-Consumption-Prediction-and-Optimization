import numpy as np
import pandas as pd
import time
import random

# Define number of floors & rooms per floor
NUM_FLOORS = 10
ROOMS_PER_FLOOR = 5

# Function to simulate real-life energy consumption in a building
def generate_synthetic_data():
    floor = random.randint(1, NUM_FLOORS)  # Select random floor (1-10)
    room = random.randint(1, ROOMS_PER_FLOOR)  # Select random room (1-5)
    
    occupancy = np.random.randint(0, 5)  # Number of people in the room
    device_usage = np.random.randint(max(occupancy, 1), occupancy + 3)  # Devices must be >= occupancy
    
    # Older floors consume more energy
    age_factor = 1 + ((NUM_FLOORS - floor) * 0.05)  # Older floors consume 5% more energy per floor
    
    # Generate environmental conditions
    temperature = round(np.random.uniform(18, 30), 2)
    humidity = round(np.random.uniform(30, 70), 2)
    windspeed = round(np.random.uniform(0, 10), 2)
    visibility = round(np.random.uniform(10, 100), 2)
    
    # Base power consumption (W) for lights, fans, ACs
    base_power = 50 * device_usage  # Each device uses ~50W
    energy_consumption = base_power * age_factor  # Adjust based on floor age
    
    data = {
        "timestamp": pd.Timestamp.now(),
        "floor": floor,
        "room": room,
        "occupancy": occupancy,
        "device_usage": device_usage,
        "temperature": temperature,
        "humidity": humidity,
        "windspeed": windspeed,
        "visibility": visibility,
        "predicted_appliances": round(energy_consumption, 2)
    }
    return data

# Simulating real-time data feed & saving to CSV
csv_file = "historical_energy_data.csv"

if __name__ == "__main__":
    while True:
        synthetic_data = generate_synthetic_data()
        
        # Save data to CSV for historical tracking
        df = pd.DataFrame([synthetic_data])
        df.to_csv(csv_file, mode='a', header=not pd.io.common.file_exists(csv_file), index=False)
        
        print(f"Generated Data: {synthetic_data}")
        time.sleep(2)  # Generate new data every 2 seconds
