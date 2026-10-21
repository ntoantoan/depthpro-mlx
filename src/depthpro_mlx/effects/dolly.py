"""Hitchcock dolly zoom ("vertigo") from a single photo and its metric depth."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..render import ease_loop, look_at, render_view


def dolly_zoom_frames(
    points: np.ndarray,
    colors: np.ndarray,
    f_px: float,
    width: int,
    height: int,
    n_frames: int = 96,
    subject_quantile: float = 0.3,
    min_magnification: float = 0.9,
    max_dolly: float = 0.5,
) -> Iterator[np.ndarray]:
    """Hitchcock "vertigo" shot from a single photo.

    The camera pulls back by ``b`` while the focal length grows as
    ``f * (z_s + b) / z_s``, so the subject at depth ``z_s`` keeps its size and
    the background swells around it. Near content shrinks by
    ``(z_s + b) * z / (z_s * (z + b))``; ``b`` is capped so that factor never
    drops below ``min_magnification`` and the frame is cropped to match, which
    keeps every pixel inside the original photo.
    """
    z = points[:, 2]
    z_s = float(np.quantile(z, subject_quantile))
    z_n = float(np.quantile(z, 0.02))

    def near_mag(b: float) -> float:
        return (z_s + b) * z_n / (z_s * (z_n + b))

    b_max = max_dolly * z_s
    if near_mag(b_max) < min_magnification:
        lo, hi = 0.0, b_max
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if near_mag(mid) >= min_magnification else (lo, mid)
        b_max = lo
    crop = 1.0 / near_mag(b_max) * 1.02

    target = np.array([0.0, 0.0, z_s], dtype=np.float32)
    for i in range(n_frames):
        b = b_max * ease_loop(i / n_frames)
        eye = np.array([0.0, 0.0, -b], dtype=np.float32)
        f = f_px * crop * (z_s + b) / z_s
        yield render_view(points, colors, look_at(eye, target), eye, f, width, height)
