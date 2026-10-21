import numpy as np
import pytest


@pytest.fixture
def scene():
    """A textured 160x120 photo: a square 1 m away in front of a wall 3 m away."""
    h, w = 120, 160
    rng = np.random.default_rng(0)
    image = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
    image[40:80, 60:100] = (220, 30, 30)  # the red square
    depth = np.full((h, w), 3.0, np.float32)
    depth[40:80, 60:100] = 1.0
    f_px = 150.0
    return image, depth, f_px
