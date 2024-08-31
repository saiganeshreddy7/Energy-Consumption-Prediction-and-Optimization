import pandas as pd

def load_data(filepath):
    """Load the raw data from a CSV file."""
    return pd.read_csv(filepath, parse_dates=['timestamp'])

def preprocess_data(df):
    """Preprocess the raw data."""
    # Convert timestamp to datetime and extract features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    
    # Handle missing values (if any)
    df.fillna(method='ffill', inplace=True)
    
    # Normalize the features
    df['temperature'] = (df['temperature'] - df['temperature'].mean()) / df['temperature'].std()
    df['humidity'] = (df['humidity'] - df['humidity'].mean()) / df['humidity'].std()

    return df

if __name__ == "__main__":
    df = load_data("/Users/saiganeshreddykodekandla/Documents/Projects/7th-Semester_project/Energy-Consumption-Prediction/data/raw/sample_data.csv")
    df = preprocess_data(df)
    df.to_csv("/Users/saiganeshreddykodekandla/Documents/Projects/7th-Semester_project/Energy-Consumption-Prediction/data/processed/processed_data.csv", index=False)
    print("Data preprocessing complete.")
