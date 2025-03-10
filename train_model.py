import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("historical_energy_data.csv")

# Feature Selection
features = ["floor", "room", "occupancy", "device_usage", "temperature", "humidity", "windspeed", "visibility"]
target = "predicted_appliances"

X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Multiple Models
models = {
    "RidgeRegression": Ridge(alpha=1.0),
    "LassoRegression": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100)
}

# Train & Save Models
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}.pkl")
    print(f"Model {name} trained and saved!")
