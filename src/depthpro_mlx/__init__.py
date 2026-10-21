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

__version__ = "0.3.0"
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


def default_dtype() -> mx.Dtype:
    """float16 on a GPU (Metal or CUDA); float32 on CPU, where BLAS only accelerates float32."""
    return mx.float16 if mx.default_device().type == mx.DeviceType.gpu else mx.float32


def load_model(
    weights: str | Path | None = None, dtype: mx.Dtype | None = None, patch_batch: int | None = None
) -> DepthPro:
    """Load MLX weights: a converted local file, or the published ones from the Hub.

    ``dtype`` defaults to :func:`default_dtype` (float16 on GPU, float32 on CPU).
    ``patch_batch`` limits how many of the 37 image tiles go through the ViT at
    once (lower = less memory, same result); defaults to all on GPU and 8 on CPU. Setting it,
    or running on CPU, also decodes in strips, so float32 peaks around 7.6 GB instead of 15 GB.
    """
    dtype = dtype or default_dtype()
    model = DepthPro()
    on_gpu = mx.default_device().type == mx.DeviceType.gpu
    model.encoder.patch_batch = patch_batch if patch_batch is not None else (None if on_gpu else 8)
    model.tiles = None if on_gpu and patch_batch is None else 12  # strip-wise decoding when saving memory
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
