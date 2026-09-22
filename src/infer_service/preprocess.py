"""Image decoding and ImageNet-style preprocessing to NCHW float32."""

import io

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def image_to_tensor(data: bytes, size: int = 224) -> np.ndarray:
    """Decode bytes, centre-crop to square, resize, normalise. Returns shape (3, size, size)."""
    with Image.open(io.BytesIO(data)) as img:
        # JPEG draft mode lets libjpeg decode at 1/2, 1/4 or 1/8 scale (never below the target),
        # which removes most of the decode cost for large photos before the resize.
        img.draft("RGB", (size, size))
        img = img.convert("RGB")
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
