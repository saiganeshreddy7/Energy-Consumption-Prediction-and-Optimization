import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
df = pd.read_csv("energydata_complete.csv")

# Feature Engineering
df["hour"] = pd.to_datetime(df["date"]).dt.hour
df["weekday"] = pd.to_datetime(df["date"]).dt.weekday
df["month"] = pd.to_datetime(df["date"]).dt.month

# Define features & target
features = ["T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5",
            "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out", "Press_mm_hg",
            "RH_out", "Windspeed", "Visibility", "Tdewpoint", "rv1", "rv2",
            "hour", "weekday", "month"]

target = "Appliances"

X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Multiple Models
models = {
    "LinearRegression": LinearRegression(),
    "SGDRegressor": SGDRegressor(max_iter=1000, tol=1e-3),
    "RidgeRegression": Ridge(alpha=1.0),
    "LassoRegression": Lasso(alpha=0.1),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Train & Save Models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"{name} MSE: {mse}")
    joblib.dump(model, f"{name}.pkl")
    print(f"Model saved as {name}.pkl")
