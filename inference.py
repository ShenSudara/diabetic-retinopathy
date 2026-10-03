"""Model loading, prediction and Grad-CAM for the DR classifiers."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Output order of both models (do not reorder).
CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

DISPLAY_NAMES = {
    "No_DR": "No DR",
    "Mild": "Mild",
    "Moderate": "Moderate",
    "Severe": "Severe",
    "Proliferate_DR": "Proliferative DR",
}
# Order used when listing probabilities in the UI (clinical severity).
SEVERITY_ORDER = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

MODEL_REGISTRY = {
    "EfficientNetB0": MODELS_DIR / "EfficientNetB0_best_model_fine_tuned_1.keras",
    "DenseNet121": MODELS_DIR / "DenseNet121_best_model_fine_tuned_1.keras",
}


# --------------------------------------------------------------------------
# Loading + prediction
# --------------------------------------------------------------------------
def load_keras_model(model_key: str) -> keras.Model:
    path = MODEL_REGISTRY[model_key]
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: models/{path.name}. "
            "Make sure it is committed to the repository."
        )
    # compile=False: we only need inference, and it avoids needing the
    # training-time loss/metric objects.
    return keras.models.load_model(str(path), compile=False)


def predict_probabilities(model: keras.Model, batch: np.ndarray) -> np.ndarray:
    """Return a (5,) probability vector in CLASS_NAMES order."""
    out = model(batch, training=False)
    if isinstance(out, (list, tuple)):
        out = out[0]
    probs = np.asarray(out)[0].astype(np.float64)
    # Safety net in case a model was saved without its softmax.
    if probs.min() < 0 or abs(probs.sum() - 1.0) > 1e-3:
        e = np.exp(probs - probs.max())
        probs = e / e.sum()
    return probs


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------
def _last_4d_layer(layers):
    """Last layer whose output is a feature map (batch, H, W, C)."""
    for layer in reversed(layers):
        try:
            shape = layer.output.shape
        except (AttributeError, RuntimeError, ValueError):
            continue
        if len(shape) == 4:
            return layer
    return None


def build_gradcam_model(model: keras.Model) -> keras.Model:
    """Return a model mapping input -> [last conv feature map, predictions].

    Both models use the head: base network -> [GlobalAveragePooling2D,
    GlobalMaxPooling2D] -> Concatenate -> ... so the feature map is taken
    as the input of the GlobalAveragePooling2D layer (same as the Colab
    notebooks). Falls back to the last 4D layer for flat models.
    """
    pool_layer = next(
        (l for l in model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D)),
        None,
    )
    if pool_layer is not None:
        return keras.Model(model.inputs, [pool_layer.input, model.output])

    target = _last_4d_layer(model.layers)
    if target is None:
        raise ValueError("No convolutional feature map found in the model.")
    return keras.Model(model.input, [target.output, model.output])


def compute_gradcam(grad_model: keras.Model, batch: np.ndarray, class_index: int) -> np.ndarray:
    """Grad-CAM heatmap in [0, 1], shape (h, w) of the last feature map."""
    x = tf.convert_to_tensor(batch)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.nn.relu(cam).numpy()
    peak = cam.max()
    if peak > 0:
        cam = cam / peak
    return cam.astype(np.float32)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Blend a JET-coloured heatmap onto an RGB uint8 image."""
    h, w = image_rgb.shape[:2]
    cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_u8 = np.uint8(255 * np.clip(cam, 0.0, 1.0))
    colored = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)  # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, colored, alpha, 0)
