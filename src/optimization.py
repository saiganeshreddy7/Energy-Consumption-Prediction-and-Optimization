import pandas as pd
from sklearn.linear_model import LinearRegression

def load_data(filepath):
    """Load the preprocessed data."""
    return pd.read_csv(filepath)

def optimize_energy_consumption(df):
    """Simple optimization using Linear Regression."""
    X = df[['temperature', 'humidity', 'hour', 'day', 'month']]
    y = df['energy_consumption']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future consumption (example)
    future_consumption = model.predict(X)
    
    return future_consumption

if __name__ == "__main__":
    df = load_data("/Users/saiganeshreddykodekandla/Documents/Projects/7th-Semester_project/Energy-Consumption-Prediction/data/processed/processed_data.csv")
    future_consumption = optimize_energy_consumption(df)
    df['predicted_consumption'] = future_consumption
    
    df.to_csv("/Users/saiganeshreddykodekandla/Documents/Projects/7th-Semester_project/Energy-Consumption-Prediction/results/optimization.csv", index=False)
    print("Optimization complete and results saved.")
