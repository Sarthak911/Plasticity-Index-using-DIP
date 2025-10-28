# Feature extraction logic
"""
feature_extraction.py
Extracts color, texture, and morphological features from soil images.
"""

import cv2
import numpy as np
from skimage.filters import gabor
from skimage.color import rgb2hsv, rgb2gray
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu

def extract_color_features(img):
    """Extract mean RGB and HSV components."""
    hsv = rgb2hsv(img)
    features = {
        "R_mean": np.mean(img[:, :, 0]),
        "G_mean": np.mean(img[:, :, 1]),
        "B_mean": np.mean(img[:, :, 2]),
        "H_mean": np.mean(hsv[:, :, 0]),
        "S_mean": np.mean(hsv[:, :, 1]),
        "V_mean": np.mean(hsv[:, :, 2])
    }
    return features

def extract_gabor_features(img):
    """Extract texture features using Gabor filters at 4 orientations."""
    gray = rgb2gray(img)
    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    freq = 0.3
    features = {}
    responses = []
    for i, theta in enumerate(orientations):
        filt_real, _ = gabor(gray, frequency=freq, theta=theta)
        features[f"Gabor_mean_{i}"] = np.mean(filt_real)
        features[f"Gabor_var_{i}"] = np.var(filt_real)
        responses.append(filt_real)
    all_responses = np.array(responses)
    features["Gabor_energy"] = np.mean(all_responses ** 2)
    features["Gabor_std"] = np.std(all_responses)
    return features

def extract_morphological_features(img):
    """Compute morphology features from binary mask."""
    gray = rgb2gray(img)
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    labeled = label(binary)
    props = regionprops(labeled)
    if not props:
        return {"Area_ratio": 0, "Perimeter_ratio": 0}
    p = props[0]
    area_ratio = p.area / (img.shape[0] * img.shape[1])
    perimeter_ratio = p.perimeter / (2 * (img.shape[0] + img.shape[1]))
    return {
        "Area_ratio": area_ratio,
        "Perimeter_ratio": perimeter_ratio
    }

def extract_all_features(img):
    """Combine all features into a single flat dictionary."""
    features = {}
    features.update(extract_color_features(img))
    features.update(extract_gabor_features(img))
    features.update(extract_morphological_features(img))
    return features

def extract_features_batch(images, names):
    """Extract features for a batch of preprocessed images."""
    data = []
    for img, name in zip(images, names):
        feats = extract_all_features(img)
        feats["sample_id"] = name
        data.append(feats)
    return data
