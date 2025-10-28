# Soil Plasticity Index Estimation

This project implements an end-to-end machine learning pipeline for estimating the **Plasticity Index (PI)** of soil samples using Support Vector Regression (SVR).

## 📂 Project Structure

```
soil_pi_estimation/
│
├── src/
│   ├── __init__.py              # Package initializer
│   ├── config.py                # Configurable parameters
│   ├── preprocessing.py         # Image preprocessing logic
│   ├── feature_extraction.py    # Feature extraction modules
│   ├── data_augmentation.py     # Geometric + photometric augmentations
│   ├── model.py                 # SVR model training and prediction
│   ├── evaluation.py            # Model performance metrics and visualization
│   └── utils.py                 # Helper and logging utilities
│
├── scripts/
│   ├── train_model.py           # Main training entry point
│   └── predict.py               # Run inference on new soil samples
│
├── data/
│   ├── raw/                     # Raw input soil images
│   ├── processed/               # Preprocessed images
│   └── labels.csv               # Actual PI values for samples

└── requirements.txt             # Dependencies list
```

## Setup Instructions

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Windows: venv\Scripts\activate)
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run training**
   ```bash
   python scripts/train_model.py --augment --tune
   ```

4. **Predict PI for new images**
   ```bash
   python scripts/predict.py --input data/raw --output predictions.csv
   ```

## Notes
- The SVR model uses an RBF kernel with configurable parameters in `config.py`.
- Dataset augmentation helps improve generalization with small datasets.
- Evaluation metrics and plots are saved in the `results/` folder.

## Synthetic Data
A sample augmented dataset `pi_data_augmented.csv` is provided, containing 5 synthetic variations per sample.
