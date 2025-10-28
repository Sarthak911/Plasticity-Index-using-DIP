# Configuration parameters
"""
config.py
Central configuration for the Soil Plasticity Index Estimation project.
"""

import os

# === Directory Configuration ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
LABELS_FILE = os.path.join(DATA_DIR, "labels", "plasticity_index.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure directories exist
for path in [DATA_DIR, RAW_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)

# === Image Preprocessing Parameters ===
IMAGE_PREPROCESSING = {
    "target_size": (256, 256),   # configurable image resize
    "denoise_strength": 10,
    "normalize": True,
    "remove_background": False
}

# === Data Augmentation Parameters ===
DATA_AUGMENTATION = {
    "augmentation_factor": 5,     # how many augmented images per original
    "rotation_range": 10,         # degrees
    "brightness_variation": 0.1,
    "contrast_variation": 0.1,
    "saturation_variation": 0.05,
    "apply_gaussian_noise": True
}

# === Model Configuration (SVR) ===
SVR_CONFIG = {
    "C": 50,
    "gamma": "scale",
    "epsilon": 0.1,
    "kernel": "rbf"
}

# === Cross-Validation Settings ===
CROSS_VALIDATION = {
    "method": "loocv",   # or 'kfold'
    "n_splits": 5        # used only if method='kfold'
}

# === Evaluation Metrics ===
EVALUATION = {
    "metrics": ["r2", "rmse", "mae"],
    "plot_results": True
}
