# SVR model training and prediction
"""
model.py
Implements Support Vector Regression (SVR) model with RBF kernel,
GridSearchCV, and Leave-One-Out Cross-Validation (LOOCV).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from .config import SVR_CONFIG, CROSS_VALIDATION, MODELS_DIR

def load_data(features_csv):
    """Load feature dataset with labels."""
    df = pd.read_csv(features_csv)
    X = df.drop(columns=["plasticity_index", "sample_id"], errors="ignore")
    y = df["plasticity_index"]
    return X, y

def train_svr(X, y, tune=True):
    """Train SVR model with LOOCV or KFold."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svr = SVR(kernel=SVR_CONFIG["kernel"],
              C=SVR_CONFIG["C"],
              gamma=SVR_CONFIG["gamma"],
              epsilon=SVR_CONFIG["epsilon"])

    if tune:
        param_grid = {
            "C": [1, 10, 50, 100],
            "gamma": ["scale", 0.1, 0.01],
            "epsilon": [0.1, 0.2, 0.5]
        }
        grid = GridSearchCV(
            svr, param_grid,
            scoring="r2",
            cv=min(5, len(X)), verbose=0
        )
        grid.fit(X_scaled, y)
        svr = grid.best_estimator_
        print(f"Best parameters: {grid.best_params_}")

    method = CROSS_VALIDATION.get("method", "loocv")
    if method == "loocv":
        cv = LeaveOneOut()
    else:
        n_splits = CROSS_VALIDATION.get("n_splits", 5)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    preds, actuals = [], []
    for train_idx, test_idx in cv.split(X_scaled):
        svr.fit(X_scaled[train_idx], y.iloc[train_idx])
        y_pred = svr.predict(X_scaled[test_idx])
        preds.extend(y_pred)
        actuals.extend(y.iloc[test_idx])

    r2 = r2_score(actuals, preds)
    rmse = mean_squared_error(actuals, preds, squared=False)
    mae = mean_absolute_error(actuals, preds)
    acc = np.mean(np.abs(np.array(actuals) - np.array(preds)) <= 5) * 100

    print(f"\nModel Performance:\nR²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, ±5PI Accuracy: {acc:.2f}%")

    # Save model
    joblib.dump({"model": svr, "scaler": scaler}, f"{MODELS_DIR}/svr_model.joblib")
    return svr, scaler, {"r2": r2, "rmse": rmse, "mae": mae, "accuracy": acc}

def predict_single(model_path, features):
    """Predict PI for a single feature vector."""
    bundle = joblib.load(model_path)
    model, scaler = bundle["model"], bundle["scaler"]
    X_scaled = scaler.transform([features])
    return model.predict(X_scaled)[0]
