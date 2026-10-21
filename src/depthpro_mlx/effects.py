"""Cinematic effects that rely on *metric* depth: dolly zoom and post-capture refocus."""

from __future__ import annotations

import math
from typing import Iterator

import cv2
import numpy as np

from .render import look_at, render_view


def _ease(t: float) -> float:
    """Smooth 0 -> 1 -> 0 over t in [0, 1) for seamless loops."""
    return 0.5 - 0.5 * math.cos(2 * math.pi * t)


def dolly_zoom_frames(
    points: np.ndarray,
    colors: np.ndarray,
    f_px: float,
    width: int,
    height: int,
    n_frames: int = 96,
    subject_quantile: float = 0.3,
    min_magnification: float = 0.85,
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
        b = b_max * _ease(i / n_frames)
        eye = np.array([0.0, 0.0, -b], dtype=np.float32)
        f = f_px * crop * (z_s + b) / z_s
        yield render_view(points, colors, look_at(eye, target), eye, f, width, height)


def _disk(radius: float) -> np.ndarray:
    r = max(radius, 0.5)
    n = int(math.ceil(r))
    y, x = np.mgrid[-n : n + 1, -n : n + 1]
    k = np.clip(r + 0.5 - np.sqrt(x * x + y * y), 0, 1).astype(np.float32)
    return k / k.sum()


class Refocus:
    """Thin-lens depth of field with a precomputed bokeh stack.

    The blur radius of a point at depth ``z`` when focused at ``z_f`` is
    ``strength * |1/z - 1/z_f|`` pixels (circle of confusion of a thin lens).
    Blurs use disk kernels in linear light with a highlight boost, so bright
    lights bloom into round bokeh like a real fast lens.
    """

    def __init__(self, image: np.ndarray, depth: np.ndarray, max_radius: float | None = None, levels: int = 14):
        h, w = image.shape[:2]
        self.max_radius = max_radius or 0.018 * w
        self.radii = np.linspace(0, self.max_radius, levels)
        self.inv_depth = 1.0 / np.clip(depth, 1e-3, None).astype(np.float32)

        lin = (image.astype(np.float32) / 255.0) ** 2.2
        luma = lin.mean(axis=-1, keepdims=True)
        lin = lin * (1 + 4.0 * np.clip(luma - 0.6, 0, None) / 0.4)  # bright lights -> stronger bokeh
        self.stack = np.stack([lin if r < 0.5 else cv2.filter2D(lin, -1, _disk(r)) for r in self.radii])

    def render(self, focus_depth: float, strength: float) -> np.ndarray:
        coc = np.clip(strength * np.abs(self.inv_depth - 1.0 / focus_depth), 0, self.max_radius)
        pos = coc / (self.radii[1] - self.radii[0])
        lo = np.floor(pos).astype(np.int32)
        hi = np.minimum(lo + 1, len(self.radii) - 1)
        t = (pos - lo)[..., None]
        rows, cols = np.indices(coc.shape)
        out = self.stack[lo, rows, cols] * (1 - t) + self.stack[hi, rows, cols] * t
        return (np.clip(out, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)

    def sweep(self, depth: np.ndarray, n_frames: int = 96, near_q: float = 0.03, far_q: float = 0.75):
        """Focus pull near -> far -> near (uniform in diopters). Yields (frame, focus_m)."""
        near, far = (float(q) for q in np.quantile(depth, [near_q, far_q]))
        # Strength chosen so content at the far end is maximally blurred when focused near.
        strength = self.max_radius / max(1 / near - 1 / far, 1e-6)
        for i in range(n_frames):
            s = _ease(i / n_frames)
            focus = 1.0 / ((1 - s) / near + s / far)
            yield self.render(focus, strength), focus
