# Data augmentation implementation
"""
data_augmentation.py
Performs geometric and color augmentations for small soil-image datasets.
"""

import cv2
import numpy as np
import os
import random
from .config import DATA_AUGMENTATION, RAW_DIR

def random_rotate(img, max_angle):
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_flip(img):
    flip_type = random.choice([-1, 0, 1])  # horizontal / vertical / both
    return cv2.flip(img, flip_type)

def random_brightness_contrast(img, brightness_var, contrast_var):
    alpha = 1 + random.uniform(-contrast_var, contrast_var)
    beta = 255 * random.uniform(-brightness_var, brightness_var)
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return np.clip(new_img, 0, 255)

def random_saturation(img, sat_var):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1 + random.uniform(-sat_var, sat_var)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def add_gaussian_noise(img):
    row, col, ch = img.shape
    mean = 0
    sigma = random.uniform(5, 15)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = img + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def augment_image(img):
    """Apply a random sequence of augmentations."""
    cfg = DATA_AUGMENTATION
    img_aug = img.copy()
    if random.random() < 0.7:
        img_aug = random_rotate(img_aug, cfg["rotation_range"])
    if random.random() < 0.5:
        img_aug = random_flip(img_aug)
    if random.random() < 0.6:
        img_aug = random_brightness_contrast(img_aug,
                                             cfg["brightness_variation"],
                                             cfg["contrast_variation"])
    if random.random() < 0.5:
        img_aug = random_saturation(img_aug, cfg["saturation_variation"])
    if cfg.get("apply_gaussian_noise", True) and random.random() < 0.4:
        img_aug = add_gaussian_noise(img_aug)
    return img_aug

def augment_dataset(input_dir=RAW_DIR, output_dir=None):
    """
    Creates augmented images for each original image in RAW_DIR.
    Returns list of new file paths.
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "augmented")
    os.makedirs(output_dir, exist_ok=True)

    aug_factor = DATA_AUGMENTATION["augmentation_factor"]
    generated = []

    for file in os.listdir(input_dir):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(input_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(aug_factor):
            aug = augment_image(img)
            new_name = f"{os.path.splitext(file)[0]}_aug{i+1}.png"
            save_path = os.path.join(output_dir, new_name)
            cv2.imwrite(save_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
            generated.append(save_path)

    return generated
