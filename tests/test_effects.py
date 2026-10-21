import numpy as np

from depthpro_mlx.effects import Refocus, dolly_zoom_frames
from depthpro_mlx.geometry import depth_to_points


def _sharpness(img: np.ndarray, box) -> float:
    y0, y1, x0, x1 = box
    return float(np.abs(np.diff(img[y0:y1, x0:x1].astype(float), axis=1)).mean())


def test_refocus_keeps_the_focused_plane_sharp(scene):
    image, depth, _ = scene
    r = Refocus(image, depth, max_radius=6, levels=7)
    wall = (5, 35, 5, 55)
    on_wall = r.render(focus_depth=3.0, strength=12.0)
    on_square = r.render(focus_depth=1.0, strength=12.0)
    # Focusing on the wall keeps it as sharp as the photo; focusing on the square blurs the wall.
    assert abs(_sharpness(on_wall, wall) - _sharpness(image, wall)) < 0.15 * _sharpness(image, wall)
    assert _sharpness(on_square, wall) < 0.7 * _sharpness(image, wall)


def test_refocus_sweep_spans_near_to_far(scene):
    image, depth, _ = scene
    focuses = [
        f for _, f in Refocus(image, depth, max_radius=4, levels=5).sweep(depth, n_frames=8, near_q=0.0, far_q=1.0)
    ]
    assert min(focuses) == np.float32(1.0) or abs(min(focuses) - 1.0) < 1e-3
    assert 1.0 < max(focuses) <= 3.0 + 1e-3


def test_dolly_zoom_keeps_frame_size(scene):
    image, depth, f = scene
    points, colors = depth_to_points(depth, image, f)
    frames = list(dolly_zoom_frames(points, colors, f, 80, 60, n_frames=3))
    assert [fr.shape for fr in frames] == [(60, 80, 3)] * 3
