"""Camera helpers: look-at rotation, a seamless easing curve and the looping orbit path."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from .splat import render_view

_Y_AXIS = np.array([0.0, 1.0, 0.0])


def look_at(eye: np.ndarray, target: np.ndarray, y_axis: np.ndarray = _Y_AXIS) -> np.ndarray:
    """World->camera rotation for a camera at ``eye`` looking at ``target``.

    Camera convention matches the unprojection: x right, y down, z forward.
    """
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(y_axis, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward]).astype(np.float32)


def ease_loop(t: float) -> float:
    """Smooth 0 -> 1 -> 0 over t in [0, 1) for seamless loops."""
    return 0.5 - 0.5 * math.cos(2 * math.pi * t)


def orbit_frames(
    points: np.ndarray,
    colors: np.ndarray,
    f_px: float,
    width: int,
    height: int,
    n_frames: int = 90,
    amplitude: float = 0.06,
    max_parallax: float = 0.035,
    zoom: float = 0.04,
    pivot_quantile: float = 0.35,
    crop: float | None = None,
) -> Iterator[np.ndarray]:
    """Yield frames of a looping "3D Ken Burns" camera path.

    The camera circles around its original position while gently dollying in,
    always looking at a pivot placed at the ``pivot_quantile`` depth so the
    subject stays anchored. The radius is ``amplitude`` x pivot depth, reduced
    when needed so near and far content never shift apart by more than
    ``max_parallax`` x frame width (large shifts open holes that cannot be
    filled convincingly). ``crop`` zooms in so regions outside the photo never
    enter the frame; by default it is derived from the largest shift.
    """
    pivot_depth = float(np.quantile(points[:, 2], pivot_quantile))
    target = np.array([0.0, 0.0, pivot_depth], dtype=np.float32)
    near, far = (float(q) for q in np.quantile(points[:, 2], [0.01, 0.99]))
    parallax_per_meter = f_px * (1 / near - 1 / far) / width
    radius = min(amplitude * pivot_depth, max_parallax / max(parallax_per_meter, 1e-9))
    if crop is None:
        shift = f_px * radius * max(1 / near - 1 / pivot_depth, 1 / pivot_depth - 1 / far)
        crop = float(np.clip(1 + 2.2 * shift / width + zoom, 1.06, 1.35))
    f_px = f_px * crop
    for i in range(n_frames):
        t = 2 * math.pi * i / n_frames
        eye = np.array(
            [radius * math.sin(t), 0.5 * radius * math.sin(2 * t), zoom * pivot_depth * (1 - math.cos(t)) / 2],
            dtype=np.float32,
        )
        yield render_view(points, colors, look_at(eye, target), eye, f_px, width, height)
