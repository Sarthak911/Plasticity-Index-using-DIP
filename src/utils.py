# General utility functions
"""
utils.py
General helper functions for data loading, logging, and preprocessing utilities.
"""

import os
import pandas as pd
import json
import datetime

def load_labels(csv_path):
    """Load the plasticity index labels CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "sample_id" not in df.columns or "plasticity_index" not in df.columns:
        raise ValueError("Label CSV must contain 'sample_id' and 'plasticity_index' columns.")
    return df

def merge_features_labels(features_df, labels_df):
    """Merge extracted features with labels based on sample_id."""
    merged = pd.merge(features_df, labels_df, on="sample_id", how="inner")
    if merged.empty:
        raise ValueError("No matching sample IDs found between features and labels.")
    return merged

def save_dataframe(df, path):
    """Save DataFrame to CSV safely."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ DataFrame saved to {path}")

def load_json(file_path):
    """Load JSON configuration or metrics."""
    with open(file_path, "r") as f:
        return json.load(f)

def timestamp():
    """Generate a timestamp string for logging or filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def log_message(msg, logfile="logs.txt"):
    """Append a message to a log file with timestamp."""
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", logfile), "a") as f:
        f.write(f"[{timestamp()}] {msg}\n")

def check_missing_values(df):
    """Print and return columns with missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("⚠ Missing values detected:")
        print(missing)
    else:
        print("✅ No missing values detected.")
    return missing
