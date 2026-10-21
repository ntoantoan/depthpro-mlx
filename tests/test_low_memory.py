import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from depthpro_mlx import default_dtype
from depthpro_mlx.model.depth_pro import DepthProEncoder, FeatureFusionBlock2d, _rows_tiled


@pytest.mark.parametrize("deconv", [False, True])
def test_strip_decoding_matches_a_single_pass(deconv):
    """Halo strips must give exactly the full-image result, including the 2x upsampling level."""
    mx.random.seed(0)
    block = FeatureFusionBlock2d(16, deconv=deconv)
    x0 = mx.random.normal((1, 40, 24, 16))
    x1 = mx.random.normal((1, 40, 24, 16))
    full = block(x0, x1)
    tiled = _rows_tiled(lambda a, b: block(a, b), [x0, x1], tiles=5, halo=6, scale=2 if deconv else 1)
    assert tiled.shape == full.shape
    np.testing.assert_allclose(np.array(tiled), np.array(full), atol=1e-5)


def test_chunked_patch_encoding_matches_one_batch():
    mx.random.seed(1)
    enc = DepthProEncoder.__new__(DepthProEncoder)
    nn.Module.__init__(enc)

    class TinyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 4)

        def __call__(self, x, return_blocks=()):
            t = self.proj(x.reshape(x.shape[0], -1, 3))
            return t, [t * 2, t * 3]

    enc.patch_encoder = TinyViT()
    pyramid = mx.random.normal((7, 4, 4, 3))
    enc.patch_batch = None
    whole = enc._encode_patches(pyramid)
    enc.patch_batch = 3
    chunked = enc._encode_patches(pyramid)
    for a, b in zip(whole, chunked, strict=True):
        np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-6)


def test_default_dtype_follows_the_device():
    expected = mx.float16 if mx.default_device().type == mx.DeviceType.gpu else mx.float32
    assert default_dtype() == expected
