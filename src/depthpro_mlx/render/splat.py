"""Novel-view rendering of a colored point cloud: z-buffered splatting with depth-aware hole filling."""

from __future__ import annotations

import cv2
import numpy as np


def _max3x3(x: np.ndarray) -> np.ndarray:
    """3x3 max filter in exact float64 (cv2.dilate would round the packed keys)."""
    p = np.pad(x, 1)
    h, w = x.shape
    out = x.copy()
    for dy in range(3):
        for dx in range(3):
            np.maximum(out, p[dy : dy + h, dx : dx + w], out=out)
    return out


def fill_holes_farthest(frame: np.ndarray, zbuf: np.ndarray, holes: np.ndarray) -> np.ndarray:
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
            frame = fill_holes_farthest(frame, zbuf.reshape(H, W), holes)

    if s > 1:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame
