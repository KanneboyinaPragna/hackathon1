# ============================
# IMPORT REQUIRED LIBRARIES
# ============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ==================================================
# STEP 1: LOAD DATASET
# ==================================================
df = pd.read_csv("data/zomato.csv")
print("Dataset loaded successfully")


# ==================================================
# STEP 2: BASIC EXPLORATORY DATA ANALYSIS (EDA)
# ==================================================
print("\n--- EDA START ---")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nStatistical Summary:")
print(df.describe())

eda_df = df[
    ["Delivery_person_Ratings", "Vehicle_condition", "Time_taken (min)"]
]
print("\nCorrelation Matrix:")
print(eda_df.corr())
print("--- EDA END ---\n")


# ==================================================
# STEP 3: SELECT REQUIRED FEATURES
# ==================================================
df = df[
    [
        "Road_traffic_density",
        "Delivery_person_Ratings",
        "Vehicle_condition",
        "Type_of_order",
        "Type_of_vehicle",
        "Time_taken (min)"
    ]
]


# ==================================================
# STEP 4: HANDLE MISSING VALUES
# ==================================================
df.dropna(inplace=True)


# ==================================================
# STEP 5: ENCODE CATEGORICAL FEATURES
# ==================================================
encoder = LabelEncoder()

categorical_cols = [
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle"
]

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])


# ==================================================
# STEP 6: FEATURES & TARGET
# ==================================================
X = df.drop("Time_taken (min)", axis=1)
y = df["Time_taken (min)"]


# ==================================================
# STEP 7: TRAIN-TEST SPLIT
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================================================
# MODEL 1: LINEAR REGRESSION (WITH TUNING)
# ==================================================
lr_params = {
    "fit_intercept": [True, False],
    "positive": [True, False]
}

lr_grid = GridSearchCV(
    LinearRegression(),
    lr_params,
    cv=3,
    scoring="neg_mean_absolute_error"
)

lr_grid.fit(X_train, y_train)
lr_best = lr_grid.best_estimator_

lr_preds = lr_best.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2 = r2_score(y_test, lr_preds)

print("\nLinear Regression (Tuned)")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R2:", lr_r2)


# ==================================================
# MODEL 2: RANDOM FOREST (WITH TUNING)
# ==================================================
rf_params = {
    "n_estimators": [100],
    "max_depth": [None, 10, 20]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

rf_preds = rf_best.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)

print("\nRandom Forest (Tuned)")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)


# ==================================================
# MODEL 3: XGBOOST (WITH TUNING)
# ==================================================
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1]
}

xgb_grid = GridSearchCV(
    XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    ),
    xgb_params,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_

xgb_preds = xgb_best.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_r2 = r2_score(y_test, xgb_preds)

print("\nXGBoost (Tuned)")
print("MAE:", xgb_mae)
print("RMSE:", xgb_rmse)
print("R2:", xgb_r2)


# ==================================================
# STEP 8: SELECT BEST MODEL (LOWEST MAE)
# ==================================================
models = {
    "LinearRegression": (lr_best, lr_mae),
    "RandomForest": (rf_best, rf_mae),
    "XGBoost": (xgb_best, xgb_mae)
}

best_model_name = min(models, key=lambda x: models[x][1])
best_model = models[best_model_name][0]

print(f"\n✅ Best Model Selected: {best_model_name}")


# ==================================================
# STEP 9: SAVE FINAL MODEL
# ==================================================
joblib.dump(best_model, "models/delivery_time_model.pkl")
print("✅ Final model saved to models/delivery_time_model.pkl")
