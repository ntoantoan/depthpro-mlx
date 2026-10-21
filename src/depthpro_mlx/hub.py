"""Locating the converted weights: a local file, ./weights, or the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

HF_REPO = "ntoan199/DepthPro-mlx"
WEIGHTS_FILE = "depth_pro.safetensors"


def resolve_weights(weights: str | Path | None = None) -> Path:
    """A local file if given (or present in ./weights), otherwise download from the Hub."""
    if weights is not None:
        return Path(weights)
    local = Path("weights") / WEIGHTS_FILE
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, WEIGHTS_FILE))
