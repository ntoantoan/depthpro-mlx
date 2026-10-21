import mlx.core as mx
import numpy as np
import pytest

from depthpro_mlx.model.depth_pro import DepthProEncoder
from depthpro_mlx.model.layers import resize_bilinear, tokens_to_grid


def test_resize_identity_is_noop():
    x = mx.array(np.random.default_rng(0).standard_normal((1, 8, 8, 3)).astype(np.float32))
    assert np.array_equal(np.array(resize_bilinear(x, 8, 8)), np.array(x))


@pytest.mark.parametrize("size", [(4, 4), (16, 16), (6, 10)])
def test_resize_matches_torch(size):
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(1).standard_normal((1, 3, 8, 8)).astype(np.float32)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x), size=size, mode="bilinear", align_corners=False).numpy()
    out = np.array(resize_bilinear(mx.array(x.transpose(0, 2, 3, 1)), *size)).transpose(0, 3, 1, 2)
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_tokens_to_grid_drops_class_token():
    tokens = mx.arange(1 * 5 * 2).reshape(1, 5, 2)  # class token + 2x2 grid
    grid = np.array(tokens_to_grid(tokens, 2))
    assert grid.shape == (1, 2, 2, 2)
    assert grid[0, 0, 0, 0] == 2  # first patch token, not the class token


def test_split_merge_round_trip():
    """5x5 tiles at 25% overlap over 1536 px, merged back with the model's padding."""
    x = mx.array(np.random.default_rng(2).standard_normal((1, 1536, 1536, 2)).astype(np.float32))
    tiles = DepthProEncoder.split(x, overlap_ratio=0.25)
    assert tiles.shape == (25, 384, 384, 2)
    # Downsample each tile 16x (as the ViT does) and merge: the 24x24 grids overlap by 6 cells.
    feats = tiles[:, ::16, ::16, :]
    merged = DepthProEncoder.merge(feats, batch_size=1, padding=3)
    assert merged.shape == (1, 96, 96, 2)
