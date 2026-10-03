"""Upload validation and image preprocessing for the DR classifiers.

The OpenCV steps below are copied from the training notebook. In training the
array reached them straight from the tf pipeline (RGB order) and OpenCV
treated it as if it were BGR. To reproduce training exactly we do the same:
there is NO colour-order conversion before or inside these steps.

Order of operations (same as training):
    decode JPEG (tf.io.decode_jpeg) -> RGB -> tf.image.resize 224x224 -> brightness -> contrast
    -> denoise -> sharpen -> float32 (0-255, no extra scaling)
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile, UnidentifiedImageError

IMG_SIZE = 224
ALLOWED_EXTENSIONS = (".jpg", ".jpeg")
MAX_FILE_BYTES = 20 * 1024 * 1024  # keep in sync with .streamlit/config.toml
MIN_SIDE = 64  # pixels; anything smaller is not a usable fundus photo

JPEG_SIGNATURE = b"\xff\xd8\xff"


class InvalidImageError(ValueError):
    """Raised when an upload is not a usable, intact JPEG."""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_and_load(file_bytes: bytes, filename: str) -> Image.Image:
    """Check the upload is a real, uncorrupted JPEG and return it as RGB.

    Checks, in order: extension, non-empty, size, JPEG signature, Pillow
    reports the format as JPEG, structure verifies, and the full image
    decodes (this is what catches truncated files).
    """
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise InvalidImageError(
            "Unsupported file type. Please upload a .jpg or .jpeg image."
        )
    if not file_bytes:
        raise InvalidImageError("The uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_BYTES:
        raise InvalidImageError(
            f"The file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB."
        )
    if not file_bytes.startswith(JPEG_SIGNATURE):
        raise InvalidImageError(
            "This is not a real JPEG file: the file signature is wrong, even "
            "though the name ends in .jpg/.jpeg (it may be a renamed PNG or "
            "another format)."
        )

    ImageFile.LOAD_TRUNCATED_IMAGES = False  # default, stated explicitly

    # Pass 1: format + structure check
    try:
        with Image.open(io.BytesIO(file_bytes)) as probe:
            if probe.format != "JPEG":
                raise InvalidImageError(
                    f"The file is a {probe.format} image, not a JPEG."
                )
            probe.verify()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(
            "The image could not be read. The file looks corrupted."
        ) from exc

    # Pass 2: full decode. verify() does not catch truncated data, load() does.
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.load()
            rgb = img.convert("RGB")
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(
            "The image is corrupted or incomplete (it could not be fully decoded)."
        ) from exc

    if min(rgb.size) < MIN_SIDE:
        raise InvalidImageError(
            f"The image is too small ({rgb.size[0]}x{rgb.size[1]} px). "
            f"Both sides must be at least {MIN_SIDE} px."
        )
    return rgb


# --------------------------------------------------------------------------
# Training-time preprocessing (copied from the notebook)
# --------------------------------------------------------------------------
def adjust_brightness(image: np.ndarray, target_brightness: int = 75) -> np.ndarray:
    """Shift brightness so the mean grey level becomes target_brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_brightness = gray.mean()
    beta = target_brightness - current_brightness
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def reduce_noise(image: np.ndarray) -> np.ndarray:
    """Bilateral filter."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """Unsharp mask."""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def preprocess_single_image_opencv(image: np.ndarray) -> np.ndarray:
    """Same as the training function, but takes a NumPy array (no .numpy())."""
    image = np.ascontiguousarray(image).astype(np.uint8)
    image = adjust_brightness(image)
    image = adjust_contrast(image)
    image = reduce_noise(image)
    image = sharpen_image(image)
    return image.astype(np.float32)


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
@dataclass
class PreparedImage:
    resized: np.ndarray       # uint8  (224, 224, 3) - what enters preprocessing
    preprocessed: np.ndarray  # float32 (224, 224, 3), values 0-255
    batch: np.ndarray         # float32 (1, 224, 224, 3) - model input


def _decode_like_training(rgb_image: Image.Image, file_bytes: bytes | None) -> np.ndarray:
    """Decode with tf.io.decode_jpeg (as the tf.data training pipeline does).

    Pillow and TensorFlow use slightly different JPEG decoders, so the same file
    can differ by a few intensity levels. Using TensorFlow's decoder keeps the
    pixels identical to training. If TensorFlow cannot decode the file (rare
    colour spaces such as CMYK), fall back to the already-validated Pillow RGB.
    """
    if file_bytes is not None:
        try:
            return tf.io.decode_jpeg(file_bytes, channels=3).numpy()
        except Exception:  # noqa: BLE001
            pass
    return np.asarray(rgb_image, dtype=np.uint8)


def prepare_image(rgb_image: Image.Image, file_bytes: bytes | None = None) -> PreparedImage:
    """Decode, resize FIRST, run the training preprocessing, add a batch axis.

    Resize is `tf.image.resize` (bilinear, float32) followed by truncation to
    uint8, exactly what `image.numpy().astype(np.uint8)` did in training.
    """
    arr = _decode_like_training(rgb_image, file_bytes)
    resized_f = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE)).numpy()
    resized = resized_f.astype(np.uint8)

    preprocessed = preprocess_single_image_opencv(resized)
    batch = np.expand_dims(preprocessed, axis=0)  # no extra scaling
    return PreparedImage(resized=resized, preprocessed=preprocessed, batch=batch)
