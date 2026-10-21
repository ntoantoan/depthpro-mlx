import numpy as np
import pytest

from depthpro_mlx.geometry import depth_to_points, layered_points
from depthpro_mlx.inpaint import occlusion_mask


def test_occlusion_mask_covers_foreground_near_edges_only(scene):
    _, depth, _ = scene
    mask, bg_depth = occlusion_mask(depth, radius_frac=0.05, jump=0.06)
    assert mask[60, 62]  # inside the square, next to its edge
    assert not mask[10, 10]  # the wall itself has nothing behind it
    assert bg_depth[60, 62] > 2.5  # the hidden layer sits at the wall's depth
    assert np.array_equal(bg_depth[~mask], depth[~mask])


def test_gentle_slopes_are_not_occluders():
    """A floor receding smoothly from 1 m to 10 m has no discontinuity to look behind."""
    depth = np.repeat(np.geomspace(10.0, 1.0, 120)[:, None], 160, axis=1).astype(np.float32)
    mask, _ = occlusion_mask(depth)
    assert not mask.any()


def test_layered_points_add_hidden_background(scene):
    image, depth, f = scene
    mask, bg_depth = occlusion_mask(depth, radius_frac=0.05)
    bg_color = np.zeros_like(image)
    points, colors = layered_points(depth, image, f, (bg_color, bg_depth, mask))
    base, _ = depth_to_points(depth, image, f)
    assert len(points) == len(base) + mask.sum()
    assert points[len(base) :, 2].min() > 2.5  # all background points are behind the square


@pytest.mark.slow
def test_lama_fills_the_mask():
    pytest.importorskip("onnxruntime")
    from depthpro_mlx.inpaint import lama_inpaint

    rng = np.random.default_rng(0)
    image = np.full((300, 400, 3), (40, 120, 200), np.uint8) + rng.integers(0, 10, (300, 400, 3), dtype=np.uint8)
    mask = np.zeros((300, 400), bool)
    mask[100:200, 150:250] = True
    image[mask] = (255, 0, 0)  # an object to remove
    out = lama_inpaint(image, mask)
    assert np.array_equal(out[~mask], image[~mask])  # untouched outside
    filled = out[mask].astype(int).mean(0)
    assert abs(filled - np.array([45, 125, 205])).max() < 30  # the blue background, not the red object
