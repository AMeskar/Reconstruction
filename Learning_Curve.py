#!/usr/bin/env python
import time
import h5py
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    max_error,
    median_absolute_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sys

warnings.filterwarnings("ignore")

# -------- Define directories --------
SAVE_DIR_DATA = "./"
CAT = "model"
DATA_FILE_PATH = "/Desktop/Energy_Reconstruction/Data_for_ML.h5"

# -------- Define features --------
FEATURES = ["channel_id", "dom_id", "t", "tdc", "pos_x", "pos_y", "pos_z", "dir_x", "dir_y", "dir_z", "tot", "a", "trig"]
TARGET = "E"

# -------- Define models --------
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeCV,
    SGDRegressor,
    ARDRegression,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    OrthogonalMatchingPursuit,
)
import xgboost as xgb
import lightgbm as lgb

MADLS = [
    ["MLP", MLPRegressor(random_state=0, hidden_layer_sizes=(100, 50, 25), max_iter=100)],
    ["KNeighbors", KNeighborsRegressor(n_jobs=-1)],
    ["DecisionTree", DecisionTreeRegressor(random_state=0)],
    ["AdaBoost", AdaBoostRegressor(random_state=0)],
    ["GradientBoosting", GradientBoostingRegressor(random_state=0, learning_rate=0.5, n_estimators=100)],
    ["RandomForest", RandomForestRegressor(random_state=0, n_jobs=-1, n_estimators=5)],
    ["ExtraTrees", ExtraTreesRegressor(random_state=0, n_jobs=-1, n_estimators=25)],
    ["Bagging", BaggingRegressor(estimator=ExtraTreesRegressor(random_state=0, n_jobs=-1, n_estimators=2), random_state=0, n_jobs=-1, n_estimators=2)],
    ["Linear", LinearRegression(n_jobs=-1)],
    ["Ridge", Ridge(random_state=0)],
    ["RidgeCV", RidgeCV()],
    ["SGD", SGDRegressor(random_state=0)],
    ["ARD", ARDRegression()],
    ["Huber", HuberRegressor(max_iter=1000)],
    ["Lars", Lars(random_state=0)],
    ["Lasso", Lasso(random_state=0)],
    ["LassoLars", LassoLars(random_state=0)],
    ["OrthogonalMatchingPursuit", OrthogonalMatchingPursuit()],
    ["HistGradientBoosting", HistGradientBoostingRegressor(learning_rate=0.1, max_iter=500)],
    ["LightGBM", lgb.LGBMRegressor(random_state=0, n_jobs=-1, learning_rate=0.1, num_iterations=100)],
    ["XGBoost", xgb.XGBRegressor(n_jobs=-1, random_state=0, learning_rate=0.1)],
]

MODELS = np.array(MADLS)[:, 0]
REGRS = np.array(MADLS)[:, 1]

# -------- Parse Command-Line Arguments --------
if len(sys.argv) < 2:
    print("Usage: python script.py <MODEL_ID>")
    sys.exit(1)

MODEL_ID = int(sys.argv[1])
MADL = MODELS[MODEL_ID]
regressor = REGRS[MODEL_ID]
print(f"Model Selected: {MADL}")

# -------- Load dataset --------
data = h5py.File(DATA_FILE_PATH, "r")
X = np.column_stack([data[F][:] for F in FEATURES])
y = np.log10(data[TARGET])  # Log-transform target variable for stability

# -------- Preprocessing --------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Training set and validation set as DataFrames
train_data = pd.DataFrame(X_train, columns=FEATURES)
train_data[TARGET] = y_train
valid_data = pd.DataFrame(X_valid, columns=FEATURES)
valid_data[TARGET] = y_valid

# -------- Learning Curve Function --------
def compute_learning_curves(train_set, x_valid, y_valid, features, target, train_sizes, regressor, n_seeds):
    metrics = {
        "r2": r2_score,
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "explained_variance": explained_variance_score,
        "max_error": max_error,
        "mape": lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "median_absolute_error": median_absolute_error,
    }
    results = {key: {"train": [], "valid": []} for key in metrics.keys()}

    for i, ts in enumerate(train_sizes):
        print(f"[{i}] Train size: {ts}")
        temp_scores = {key: {"train": [], "valid": []} for key in metrics.keys()}

        for seed in range(n_seeds):
            # Randomly sample training data
            sample = train_set.sample(n=ts, random_state=seed)
            regr = regressor
            regr.random_state = seed

            # Train the model
            regr.fit(sample[features], sample[target])
            y_pred_train = regr.predict(sample[features])
            y_pred_valid = regr.predict(x_valid)

            # Compute metrics
            for key, metric_fn in metrics.items():
                temp_scores[key]["train"].append(metric_fn(sample[target], y_pred_train))
                temp_scores[key]["valid"].append(metric_fn(y_valid[target], y_pred_valid))

        # Store mean and std for all metrics
        for key in metrics.keys():
            results[key]["train"].append((np.mean(temp_scores[key]["train"]), np.std(temp_scores[key]["train"])))
            results[key]["valid"].append((np.mean(temp_scores[key]["valid"]), np.std(temp_scores[key]["valid"])))

    return results


# -------- Compute Learning Curves --------
train_sizes = np.linspace(0.1, 1.0, 10) * len(train_data)
train_sizes = train_sizes.astype(int)
results = compute_learning_curves(
    train_data,
    valid_data[FEATURES],
    valid_data[[TARGET]],
    FEATURES,
    TARGET,
    train_sizes,
    regressor,
    n_seeds=5,
)

# -------- Plot Results --------
plt.figure(figsize=(16, 12))
for metric, values in results.items():
    train_mean = [v[0] for v in values["train"]]
    train_std = [v[1] for v in values["train"]]
    valid_mean = [v[0] for v in values["valid"]]
    valid_std = [v[1] for v in values["valid"]]

    plt.plot(train_sizes, train_mean, label=f"{metric.upper()} Train", marker="o")
    plt.fill_between(train_sizes, np.array(train_mean) - np.array(train_std), np.array(train_mean) + np.array(train_std), alpha=0.2)
    plt.plot(train_sizes, valid_mean, label=f"{metric.upper()} Validation", marker="o")
    plt.fill_between(train_sizes, np.array(valid_mean) - np.array(valid_std), np.array(valid_mean) + np.array(valid_std), alpha=0.2)

plt.xlabel("Training Size")
plt.ylabel("Score")
plt.title(f"Learning Curves: {MADL}")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid()
plt.tight_layout()
plt.savefig(f'Learning_curve_Plot_{MADL}.png')
plt.show()
