import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore

def load_data(filepath):
    """Load the preprocessed data."""
    return pd.read_csv(filepath)

def build_model(input_shape):
    """Build a simple neural network model."""
    model = Sequential([
        Input(shape=(input_shape,)),  # Use Input layer
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    df = load_data("/Users/saiganeshreddykodekandla/Documents/Projects/7th-Semester_project/Energy-Consumption-Prediction/data/processed/processed_data.csv")
    X = df[['temperature', 'humidity', 'hour', 'day', 'month']]
    y = df['energy_consumption']
    
    model = build_model(X.shape[1])
    model.fit(X, y, epochs=10, validation_split=0.2)
    
    model.save("/Users/saiganeshreddykodekandla/Documents/Projects/7th-Semester_project/Energy-Consumption-Prediction/models/model_v0.1.keras")  # Save using recommended format
    print("Model training complete and saved.")
