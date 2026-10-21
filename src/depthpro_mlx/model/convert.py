"""Convert the official PyTorch checkpoint (depth_pro.pt) to MLX safetensors.

Usage:
    python -m depthpro_mlx.model.convert --checkpoint checkpoints/depth_pro.pt --out weights/depth_pro.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .depth_pro import DepthPro


def _conv_transpose_paths(model: nn.Module) -> set[str]:
    return {f"{name}.weight" for name, m in model.named_modules() if isinstance(m, nn.ConvTranspose2d)}


def convert_state_dict(state_dict: dict[str, np.ndarray], model: DepthPro) -> dict[str, mx.array]:
    """Map torch tensors onto the MLX parameter tree, fixing conv layouts."""
    expected = dict(tree_flatten(model.parameters()))
    transposed_convs = _conv_transpose_paths(model)

    weights = {}
    for key, value in state_dict.items():
        if "fc_norm" in key:
            continue
        if key not in expected:
            raise KeyError(f"Unexpected key in checkpoint: {key}")
        if value.ndim == 4:
            # torch Conv2d: (out, in, kh, kw); ConvTranspose2d: (in, out, kh, kw)
            # MLX (both):   (out, kh, kw, in)
            axes = (1, 2, 3, 0) if key in transposed_convs else (0, 2, 3, 1)
            value = value.transpose(axes)
        if value.shape != tuple(expected[key].shape):
            raise ValueError(f"Shape mismatch for {key}: {value.shape} vs {expected[key].shape}")
        weights[key] = mx.array(value)

    missing = set(expected) - set(weights)
    if missing:
        raise KeyError(f"Missing keys: {sorted(missing)[:10]}")
    return weights


def load_torch_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    import torch  # only needed for conversion

    state_dict = torch.load(path, map_location="cpu")
    return {k: v.float().numpy() for k, v in state_dict.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/depth_pro.pt")
    parser.add_argument("--out", default="weights/depth_pro.safetensors")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    model = DepthPro()
    weights = convert_state_dict(load_torch_checkpoint(args.checkpoint), model)
    dtype = getattr(mx, args.dtype)
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), weights, metadata={"format": "mlx", "source": "apple/ml-depth-pro"})
    n = sum(v.size for v in weights.values())
    print(f"saved {len(weights)} tensors ({n / 1e6:.1f}M params, {args.dtype}) -> {out}")


if __name__ == "__main__":
    main()
