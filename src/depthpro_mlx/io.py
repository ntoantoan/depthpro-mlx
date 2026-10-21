"""Image loading (with EXIF focal length) and visualization helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(path: str | Path) -> tuple[np.ndarray, float | None]:
    """RGB uint8 image plus focal length in pixels from EXIF (35mm-equivalent), if present."""
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        import pillow_heif

        pillow_heif.register_heif_opener()
    img = Image.open(path)
    exif = img.getexif().get_ifd(0x8769)  # Exif IFD
    f_35mm = exif.get(0xA405)  # FocalLengthIn35mmFilm
    img = np.array(img.convert("RGB"))
    f_px = None
    if f_35mm:
        h, w = img.shape[:2]
        f_px = float(f_35mm) * np.sqrt(w**2 + h**2) / np.sqrt(36**2 + 24**2)
    return img, f_px


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Inverse depth, robustly normalized, turbo colormap. Returns RGB uint8."""
    inv = 1.0 / np.clip(depth, 1e-3, None)
    lo, hi = np.percentile(inv, [1, 99])
    norm = np.clip((inv - lo) / (hi - lo + 1e-8), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def fit_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (width, round(h * width / w) // 2 * 2), interpolation=cv2.INTER_AREA)
