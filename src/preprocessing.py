# Image preprocessing functions
"""
preprocessing.py
Handles image loading, resizing, denoising, normalization, and optional background removal.
"""

import cv2
import numpy as np
import os
from skimage import exposure
from skimage.restoration import denoise_nl_means, estimate_sigma
from .config import IMAGE_PREPROCESSING, RAW_DIR

def load_image(image_path):
    """Load image in RGB format."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_image(img, target_size=None):
    """Resize to target size."""
    if target_size is None:
        target_size = IMAGE_PREPROCESSING['target_size']
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def denoise_image(img, strength=None):
    """Apply non-local means denoising."""
    if strength is None:
        strength = IMAGE_PREPROCESSING['denoise_strength']
    sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))
    return denoise_nl_means(
        img,
        h=1.15 * sigma_est * strength / 10.0,
        fast_mode=True,
        channel_axis=-1
    )

def normalize_image(img):
    """Normalize pixel values to 0–1."""
    img = img.astype(np.float32)
    img /= 255.0
    img = np.clip(img, 0, 1)
    return img

def remove_background(img):
    """Optional simple background removal using thresholding."""
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask)
    background = np.ones_like(img) * np.mean(img)
    fg = cv2.bitwise_and(img, img, mask=mask)
    bg = cv2.bitwise_and(background, background, mask=mask_inv)
    return cv2.add(fg, bg)

def preprocess_image(image_path):
    """Complete preprocessing pipeline."""
    img = load_image(image_path)
    img = resize_image(img)
    img = denoise_image(img)
    if IMAGE_PREPROCESSING.get("normalize", True):
        img = normalize_image(img)
    if IMAGE_PREPROCESSING.get("remove_background", False):
        img = remove_background(img)
    return img

def preprocess_dataset(directory=RAW_DIR):
    """Batch process all images in the dataset."""
    processed_images = []
    image_names = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(directory, file)
            processed_images.append(preprocess_image(path))
            image_names.append(file)
    return processed_images, image_names
