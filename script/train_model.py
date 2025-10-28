# Main training pipeline script
"""
train_model.py
Main script to train the SVR model for Soil Plasticity Index estimation.
"""

import os
import pandas as pd
import argparse
from src import (
    config,
    preprocessing,
    feature_extraction,
    data_augmentation,
    model,
    evaluation,
    utils,
)

def main(args):
    print("🚀 Starting Soil Plasticity Index Training Pipeline")

    # Step 1: Data Augmentation
    if args.augment:
        print("🔄 Performing data augmentation...")
        augmented = data_augmentation.augment_dataset(config.RAW_DIR)
        print(f"✅ Augmented {len(augmented)} new images")

    # Step 2: Preprocess all images
    print("🧹 Preprocessing images...")
    images, names = preprocessing.preprocess_dataset(config.RAW_DIR)
    print(f"✅ Processed {len(images)} images")

    # Step 3: Extract features
    print("🔍 Extracting features...")
    features_data = feature_extraction.extract_features_batch(images, names)
    features_df = pd.DataFrame(features_data)
    print(f"✅ Extracted {features_df.shape[1]} features for {features_df.shape[0]} samples")

    # Step 4: Load labels and merge
    labels_df = utils.load_labels(config.LABELS_FILE)
    full_df = utils.merge_features_labels(features_df, labels_df)
    features_csv = os.path.join(config.RESULTS_DIR, "features.csv")
    utils.save_dataframe(full_df, features_csv)

    # Step 5: Train SVR model
    print("🤖 Training SVR model...")
    X, y = full_df.drop(columns=["sample_id", "plasticity_index"]), full_df["plasticity_index"]
    svr_model, scaler, metrics = model.train_svr(X, y, tune=args.tune)

    # Step 6: Save metrics
    evaluation.save_metrics(metrics)

    print("\n✅ Training complete!")
    print(f"📈 Results: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}, "
          f"MAE={metrics['mae']:.3f}, ±5PI Acc={metrics['accuracy']:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVR model for Soil PI estimation")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    args = parser.parse_args()
    main(args)
