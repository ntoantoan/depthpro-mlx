"""Novel-view rendering of a colored point cloud (z-buffered splatting + hole filling).

Used to turn a single photo into a short parallax "3D photo" video.
"""

from __future__ import annotations

import math
from typing import Iterator

import cv2
import numpy as np


def look_at(eye: np.ndarray, target: np.ndarray, y_axis: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """World->camera rotation for a camera at ``eye`` looking at ``target``.

    Camera convention matches the unprojection: x right, y down, z forward.
    """
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(y_axis, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward]).astype(np.float32)


def _max3x3(x: np.ndarray) -> np.ndarray:
    """3x3 max filter in exact float64 (cv2.dilate would round the packed keys)."""
    p = np.pad(x, 1)
    h, w = x.shape
    out = x.copy()
    for dy in range(3):
        for dx in range(3):
            np.maximum(out, p[dy : dy + h, dx : dx + w], out=out)
    return out


def _fill_farthest(frame: np.ndarray, zbuf: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Fill holes with the farthest surface on their border (depth-aware push-pull).

    Disocclusions open up next to foreground edges and reveal what lies behind,
    so the right fill is whatever is *farther* on the hole's border, never the
    foreground. Colors are packed with depth into one sortable key
    (depth in the high bits), a max-pyramid is built, and holes are filled
    coarse-to-fine from the max over each parent's 3x3 neighbourhood, which
    spans both sides of the hole at the matching scale.
    """
    # Integer depth code: a fractional part would spill into the packed color bits.
    zq = np.round(np.clip(zbuf / max(zbuf.max(), 1e-6) * 65535, 1, 65535)).astype(np.float64)
    rgb = frame.astype(np.float64)
    key = zq * 2.0**24 + rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]
    key[holes] = 0

    pyramid = [key]
    while min(pyramid[-1].shape) > 1 and (pyramid[-1] == 0).any():
        k = pyramid[-1]
        h, w = k.shape
        k = np.pad(k, ((0, h % 2), (0, w % 2)))
        pyramid.append(k.reshape(k.shape[0] // 2, 2, k.shape[1] // 2, 2).max(axis=(1, 3)))
    for level in range(len(pyramid) - 2, -1, -1):
        k = pyramid[level]
        parent = _max3x3(pyramid[level + 1])
        up = np.repeat(np.repeat(parent, 2, axis=0), 2, axis=1)[: k.shape[0], : k.shape[1]]
        pyramid[level] = np.where(k == 0, up, k)

    packed = (pyramid[0] % 2.0**24).astype(np.int64)
    filled = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=-1).astype(np.uint8)
    # The pull step yields flat patches; blur them wide and feather the seam so a
    # disocclusion reads as soft out-of-focus background rather than a cut-out.
    sigma = max(2.0, 0.006 * frame.shape[1])
    smooth = cv2.GaussianBlur(filled, (0, 0), sigma)
    pinholes = cv2.morphologyEx(holes.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) == 0
    smooth = np.where((holes & pinholes)[..., None], cv2.GaussianBlur(filled, (3, 3), 0), smooth)
    alpha = cv2.GaussianBlur(holes.astype(np.float32), (0, 0), 1.5)[..., None]
    alpha = np.maximum(alpha, holes[..., None])
    return (smooth * alpha + frame * (1 - alpha)).astype(np.uint8)


def render_view(
    points: np.ndarray,
    colors: np.ndarray,
    rotation: np.ndarray,
    eye: np.ndarray,
    f_px: float,
    width: int,
    height: int,
    supersample: int = 2,
    fill_holes: bool = True,
) -> np.ndarray:
    """Project points into a pinhole camera and z-buffer them. Returns uint8 (H, W, 3)."""
    s = supersample
    W, H, f = width * s, height * s, f_px * s
    cam = (points - eye) @ rotation.T
    z = cam[:, 2]
    front = z > 1e-3
    cam, z, cols = cam[front], z[front], colors[front]

    u = np.round(f * cam[:, 0] / z + (W - 1) / 2).astype(np.int64)
    v = np.round(f * cam[:, 1] / z + (H - 1) / 2).astype(np.int64)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, cols = u[inside], v[inside], z[inside], cols[inside]

    # Nearest point wins: sort by depth, keep the first hit per pixel.
    order = np.argsort(z, kind="stable")
    idx = (v * W + u)[order]
    _, first = np.unique(idx, return_index=True)
    hit = idx[first]

    frame = np.zeros((H * W, 3), dtype=np.uint8)
    frame[hit] = cols[order][first]
    frame = frame.reshape(H, W, 3)

    if fill_holes:
        zbuf = np.zeros(H * W, dtype=np.float32)
        zbuf[hit] = z[order][first]
        holes = (zbuf == 0).reshape(H, W)
        if holes.any():
            frame = _fill_farthest(frame, zbuf.reshape(H, W), holes)

    if s > 1:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


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
