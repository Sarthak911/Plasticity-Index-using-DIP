# Evaluation metrics and plotting
"""
evaluation.py
Handles model performance visualization and metric export.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from .config import RESULTS_DIR

def compute_metrics(y_true, y_pred):
    """Compute R², RMSE, MAE, ±5PI accuracy."""
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    acc = np.mean(np.abs(np.array(y_true) - np.array(y_pred)) <= 5) * 100
    return {"r2": r2, "rmse": rmse, "mae": mae, "accuracy": acc}

def save_metrics(metrics, filename="metrics.json"):
    """Save computed metrics to a JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = os.path.join(RESULTS_DIR, filename)
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics saved to {file_path}")

def plot_results(y_true, y_pred, title="SVR Model Performance"):
    """Generate evaluation plots and save to results folder."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.8)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual PI")
    plt.ylabel("Predicted PI")
    plt.title(f"{title} - Predicted vs Actual")
    plt.savefig(os.path.join(RESULTS_DIR, "pred_vs_actual.png"))
    plt.close()

    # 2. Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title(f"{title} - Residual Distribution")
    plt.savefig(os.path.join(RESULTS_DIR, "residuals.png"))
    plt.close()

    # 3. Error Distribution
    abs_err = np.abs(residuals)
    plt.figure(figsize=(6, 4))
    plt.hist(abs_err, bins=10, color='orange', edgecolor='black')
    plt.xlabel("Absolute Error (|Actual - Pred|)")
    plt.ylabel("Frequency")
    plt.title(f"{title} - Error Distribution")
    plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"))
    plt.close()

    print("✅ Plots saved: pred_vs_actual.png, residuals.png, error_distribution.png")

def evaluate_and_save(y_true, y_pred, title="SVR Model Performance"):
    """Compute metrics, save JSON, and generate plots."""
    metrics = compute_metrics(y_true, y_pred)
    save_metrics(metrics)
    plot_results(y_true, y_pred, title)
    return metrics
