"""End-to-end parity with Apple's PyTorch reference. Needs local weights and the reference
outputs written by tools/bench/bench_torch.py --save outputs/ref_depth_fp32.npy."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "depth_pro.safetensors"
REF = ROOT / "outputs" / "ref_depth_fp32.npy"
IMAGE = ROOT / "third_party" / "ml-depth-pro" / "data" / "example.jpg"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (WEIGHTS.exists() and REF.exists() and IMAGE.exists()), reason="weights/reference not present"
    ),
]


def test_fp32_matches_pytorch_reference():
    import mlx.core as mx

    from depthpro_mlx import load_model, predict
    from depthpro_mlx.io import load_image

    image, _ = load_image(IMAGE)
    depth = predict(load_model(WEIGHTS, dtype=mx.float32), image)["depth"]
    ref = np.load(REF)
    assert np.mean(np.abs(depth - ref) / ref) < 1e-4
