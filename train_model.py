import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt 
import os
import json

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv("energydata_complete.csv")

# Basic EDA
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nSample data:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data statistics
print("\nData statistics:")
print(df.describe())

# Feature Engineering
print("Performing feature engineering...")
# Convert date to datetime first
df["date"] = pd.to_datetime(df["date"])
df["hour"] = df["date"].dt.hour
df["weekday"] = df["date"].dt.weekday
df["month"] = df["date"].dt.month
df["season"] = df["date"].dt.month % 12 // 3  # 0: winter, 1: spring, 2: summer, 3: fall
df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
df["day_period"] = df["hour"].apply(lambda x: 0 if 0 <= x < 6 else (1 if 6 <= x < 12 else (2 if 12 <= x < 18 else 3)))

# Temperature differentials (indoor vs outdoor)
df["temp_diff_avg"] = df["T_out"] - df[["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]].mean(axis=1)

# Average temperature and humidity
df["T_avg"] = df[["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]].mean(axis=1)
df["RH_avg"] = df[["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9"]].mean(axis=1)

# Correlation with target - exclude date column
df_numeric = df.drop(columns=['date'])
correlations = df_numeric.corr()["Appliances"].sort_values(ascending=False)
print("\nTop 10 features correlated with Appliances:")
print(correlations.head(10))

# Visualize correlations
plt.figure(figsize=(12, 8))
plt.barh(correlations.index[:15], correlations.values[:15])
plt.title("Top 15 Features Correlated with Appliances Energy Consumption")
plt.xlabel("Correlation Coefficient")
plt.savefig("visualizations/feature_correlations.png")

# Define features & target
features = [
    "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5",
    "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out", "Press_mm_hg",
    "RH_out", "Windspeed", "Visibility", "Tdewpoint", "rv1", "rv2",
    "hour", "weekday", "month", "season", "is_weekend", "day_period", 
    "temp_diff_avg", "T_avg", "RH_avg"
]

target = "Appliances"

X = df[features]
y = df[target]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Train Multiple Models with hyperparameter tuning
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "SGDRegressor": SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),  # Added random_state
    "KNN": KNeighborsRegressor(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

# Hyperparameter grids
param_grids = {
    "LinearRegression": {},
    "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1.0]},
    "SGDRegressor": {"alpha": [0.0001, 0.001, 0.01], "penalty": ["l1", "l2", "elasticnet"]},
    "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    "RandomForest": {"n_estimators": [50, 100], "max_depth": [10, 20, None]},
    "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
}

# Performance metrics
results = {}

# Train models with hyperparameter tuning
print("\nTraining models with hyperparameter tuning...")
for name, model in models.items():
    print(f"Training {name}...")
    
    # Perform grid search if there are hyperparameters to tune
    if param_grids[name]:
        grid_search = GridSearchCV(model, param_grids[name], scoring="neg_mean_squared_error", cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    
    # Handle negative predictions
    y_pred = np.maximum(y_pred, 0)  # Make sure predictions are not negative
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    
    # Save model
    joblib.dump(best_model, f"models/{name}.pkl")
    print(f"Model saved as models/{name}.pkl")
    
    # Feature importance (for models that support it)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances - {name}')
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f"visualizations/feature_importance_{name}.png")

# Save results
with open("models/performance_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

# Visualize model comparison
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), [results[model]["rmse"] for model in results.keys()])
plt.title("Model Comparison - RMSE")
plt.ylabel("Root Mean Squared Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/model_comparison_rmse.png")

plt.figure(figsize=(12, 6))
plt.bar(results.keys(), [results[model]["r2"] for model in results.keys()])
plt.title("Model Comparison - R²")
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/model_comparison_r2.png")

print("\nModel training and evaluation complete!")
print(f"Performance metrics saved to models/performance_metrics.json")
print(f"Visualizations saved to the visualizations directory")