"""Depth Pro (Apple): sharp monocular metric depth, ported to MLX for Apple Silicon.

from depthpro_mlx import load_model, predict
model = load_model()                      # converted weights from the Hugging Face Hub
out = predict(model, image)               # image: RGB uint8 (H, W, 3)
out["depth"], out["focallength_px"]       # meters, pixels
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .hub import HF_REPO, WEIGHTS_FILE, resolve_weights
from .model import DepthPro

__version__ = "0.2.0"
__all__ = [
    "HF_REPO",
    "WEIGHTS_FILE",
    "DepthPro",
    "load_model",
    "predict",
    "preprocess",
    "resolve_weights",
    "__version__",
]


def load_model(weights: str | Path | None = None, dtype: mx.Dtype = mx.float16) -> DepthPro:
    """Load MLX weights: a converted local file, or the published ones from the Hub."""
    model = DepthPro()
    model.load_weights(str(resolve_weights(weights)), strict=True)
    model.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())
    return model


def preprocess(image: np.ndarray, dtype: mx.Dtype = mx.float16) -> mx.array:
    """uint8 RGB (H, W, 3) -> MLX array normalized to [-1, 1]."""
    x = mx.array(image).astype(mx.float32) / 255.0
    return ((x - 0.5) / 0.5).astype(dtype)


def predict(model: DepthPro, image: np.ndarray, f_px: float | None = None) -> dict:
    """Run Depth Pro on an RGB uint8 image.

    Returns ``{"depth": np.ndarray (H, W) in meters, "focallength_px": float}``.
    """
    dtype = model.encoder.patch_encoder.pos_embed.dtype
    out = model.infer(preprocess(image, dtype), f_px=f_px)
    mx.eval(out)
    return {"depth": np.array(out["depth"]), "focallength_px": float(out["focallength_px"])}
