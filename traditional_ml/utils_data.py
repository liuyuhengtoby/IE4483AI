"""
utils_data.py
Utility functions for loading Dogs vs Cats dataset and extracting HOG features.
"""

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.feature import hog

# --- Configuration ---
IMG_SIZE = (128, 128)  # resize target (height, width)
HOG_PARAMS = dict(
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    orientations=9,
    transform_sqrt=True,
    block_norm="L2-Hys",
    feature_vector=True,
)

# --- Image loading ---
def load_image(path, size=IMG_SIZE):
    """Open image, convert to grayscale, resize to target size."""
    img = Image.open(path).convert("L")  # convert to grayscale
    img = img.resize(size, Image.BILINEAR)  # resize for uniformity
    return np.array(img)  # convert to numpy array

# --- HOG feature extraction ---
def extract_hog(img_array):
    """Compute HOG feature vector for a given grayscale image."""
    return hog(img_array, **HOG_PARAMS)

# --- Dataset helpers ---
def collect_image_paths(data_dir):
    """
    Given a directory with subfolders 'cat' and 'dog',
    return image paths and labels (0=cat, 1=dog).
    """
    data_dir = Path(data_dir)
    paths, labels = [], []
    for label_name, label in [("cat", 0), ("dog", 1)]:
        subdir = data_dir / label_name
        for img_path in subdir.glob("*.jpg"):
            paths.append(img_path)
            labels.append(label)
    return np.array(paths), np.array(labels)

def build_feature_matrix(image_paths):
    """Extract HOG features for all images in the list."""
    features = []
    for path in tqdm(image_paths, desc="Extracting HOG features"):
        img = load_image(path)
        hog_feat = extract_hog(img)
        features.append(hog_feat)
    return np.vstack(features)
