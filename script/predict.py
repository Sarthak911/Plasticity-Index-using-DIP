# Prediction script
"""
predict.py
Predicts Soil Plasticity Index (PI) for new soil sample images using the trained SVR model.
"""

import os
import argparse
import pandas as pd
from src import config, preprocessing, feature_extraction, model

def predict_for_images(input_dir, output_csv="predictions.csv"):
    print(f"🔮 Predicting PI for images in: {input_dir}")

    # Step 1: Preprocess images
    images, names = preprocessing.preprocess_dataset(input_dir)
    if not images:
        print("⚠ No valid images found!")
        return None

    # Step 2: Extract features
    features_data = feature_extraction.extract_features_batch(images, names)
    features_df = pd.DataFrame(features_data)

    # Step 3: Load trained model
    model_path = os.path.join(config.MODELS_DIR, "svr_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Trained model not found. Run train_model.py first.")

    # Step 4: Predict PI values
    preds = []
    for _, row in features_df.iterrows():
        X = row.drop("sample_id").values
        y_pred = model.predict_single(model_path, X)
        preds.append(round(float(y_pred), 2))

    # Step 5: Save predictions
    features_df["predicted_PI"] = preds
    out_path = os.path.join(config.RESULTS_DIR, output_csv)
    features_df.to_csv(out_path, index=False)
    print(f"✅ Predictions saved to {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict PI using trained SVR model")
    parser.add_argument("--input", type=str, default=config.RAW_DIR,
                        help="Directory containing new soil images")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV file name for predictions")
    args = parser.parse_args()

    predict_for_images(args.input, args.output)
