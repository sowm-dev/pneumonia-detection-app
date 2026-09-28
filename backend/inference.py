"""Backend for the Pneumonia Detection app: image reading, preprocessing and prediction.

The preprocessing here mirrors the training pipeline exactly:
DICOM rescale (if present) -> 1st-99th percentile normalisation -> 224x224 (INTER_AREA)
-> uint8 -> divide by 255. Any RGB image is converted to single-channel grayscale first.
"""
import io
import json
import os

import cv2
import numpy as np
import pydicom
from PIL import Image

IMG_SIZE = 224
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "pneumonia_best_model.keras")
CONFIG_PATH = os.path.join(BASE_DIR, "model", "model_config.json")

_MODEL = None


def load_config():
    """Model metadata: name, decision threshold and headline validation/test metrics."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)



def _ensure_model_file():
    """Make sure MODEL_PATH is a real Keras file, not a Git-LFS pointer."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    url = os.environ.get("MODEL_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("MODEL_URL")
        except Exception:
            url = None
    if not url:
        raise RuntimeError("Model file is missing or is a Git-LFS pointer, and MODEL_URL is not set.")
    import urllib.request
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(url, MODEL_PATH)


def load_model():
    """Load the serialized Keras model once and reuse it."""
    global _MODEL
    if _MODEL is None:
        import tensorflow as tf
        _ensure_model_file()
        _MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _MODEL


def _normalise_to_uint8(img, size=IMG_SIZE):
    """Percentile-normalise a 2-D array and resize to size x size (uint8)."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99))
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    img = (img * 255.0).astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def read_upload(file_bytes, filename):
    """Read a DICOM / PNG / JPG upload and return a 2-D grayscale float array."""
    if filename.lower().endswith(".dcm"):
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr  # make sure bones/air polarity matches MONOCHROME2 training data
        return arr
    return np.array(Image.open(io.BytesIO(file_bytes)).convert("L"), dtype=np.float32)


def predict(file_bytes, filename):
    """Return the pneumonia probability, the class label and the preprocessed image."""
    cfg = load_config()
    model = load_model()
    img_u8 = _normalise_to_uint8(read_upload(file_bytes, filename), int(cfg.get("img_size", IMG_SIZE)))
    x = (img_u8.astype(np.float32) / 255.0)[None, :, :, None]
    prob = float(model.predict(x, verbose=0).ravel()[0])
    threshold = float(cfg.get("threshold", 0.5))
    label = "Pneumonia (lung opacity) likely" if prob >= threshold else "No pneumonia opacity detected"
    return {"probability": prob, "threshold": threshold, "label": label,
            "is_positive": prob >= threshold, "image": img_u8}
