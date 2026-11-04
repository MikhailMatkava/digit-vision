"""Dataset splitting and the shared train/inference input contract."""
import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def split_indices(labels, seed=42):
    indices = np.arange(len(labels))
    train_val, test = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=0.25, stratify=labels[train_val], random_state=seed
    )
    return train, val, test


def dataset():
    data = load_digits()
    return data.data.astype(np.float32) / 16.0, data.target


def image_features(path, invert=False):
    """Expect one centered light digit on dark background; invert explicitly."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('L')
        if invert:
            image = ImageOps.invert(image)
        image = ImageOps.pad(image, (8, 8), method=Image.Resampling.LANCZOS, color=0)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    if float(pixels.max() - pixels.min()) < 0.05:
        raise ValueError('Image is blank or has too little contrast.')
    return pixels.reshape(1, 64)
