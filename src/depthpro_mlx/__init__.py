"""Depth Pro (Apple) — sharp monocular metric depth, ported to MLX for Apple Silicon."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from .model import DepthPro

__all__ = ["DepthPro", "load_model", "preprocess", "predict", "HF_REPO"]

HF_REPO = "ntoan199/DepthPro-mlx"
WEIGHTS_FILE = "depth_pro.safetensors"


def resolve_weights(weights: Optional[Union[str, Path]] = None) -> Path:
    """A local file if given (or present in ./weights), otherwise download from the Hub."""
    if weights is not None:
        return Path(weights)
    local = Path("weights") / WEIGHTS_FILE
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, WEIGHTS_FILE))


def load_model(weights: Optional[Union[str, Path]] = None, dtype: mx.Dtype = mx.float16) -> DepthPro:
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


def predict(model: DepthPro, image: np.ndarray, f_px: Optional[float] = None) -> dict:
    """Run Depth Pro on an RGB uint8 image.

    Returns ``{"depth": np.ndarray (H, W) in meters, "focallength_px": float}``.
    """
    dtype = model.encoder.patch_encoder.pos_embed.dtype
    out = model.infer(preprocess(image, dtype), f_px=f_px)
    mx.eval(out)
    return {"depth": np.array(out["depth"]), "focallength_px": float(out["focallength_px"])}
