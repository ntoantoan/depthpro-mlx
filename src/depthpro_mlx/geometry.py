"""Depth map -> colored 3D point cloud, plus PLY export."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def edge_mask(depth: np.ndarray, threshold: float = 0.04, band: int | None = None) -> np.ndarray:
    """True where depth is locally smooth.

    Depth discontinuities produce "flying pixels" that smear between foreground
    and background when the scene is viewed from a new angle. Points whose
    log-depth jumps by more than ``threshold`` to a 4-neighbour are dropped,
    together with a ``band``-pixel margin (default ~0.1% of the width): pixels
    right at an edge often carry the foreground's color at the background's
    depth and would otherwise show up as ghosts behind the subject.
    """
    import cv2

    if band is None:
        band = max(1, round(depth.shape[1] / 1000))
    log_d = np.log(np.clip(depth, 1e-3, None))
    jump = np.zeros_like(log_d)
    dy = np.abs(np.diff(log_d, axis=0))
    dx = np.abs(np.diff(log_d, axis=1))
    jump[1:, :] = np.maximum(jump[1:, :], dy)
    jump[:-1, :] = np.maximum(jump[:-1, :], dy)
    jump[:, 1:] = np.maximum(jump[:, 1:], dx)
    jump[:, :-1] = np.maximum(jump[:, :-1], dx)
    edges = (jump >= threshold).astype(np.uint8)
    if band > 1:
        edges = cv2.dilate(edges, np.ones((2 * band - 1, 2 * band - 1), np.uint8))
    return edges == 0


def depth_to_points(
    depth: np.ndarray,
    image: np.ndarray,
    f_px: float,
    stride: int = 1,
    edge_threshold: float | None = 0.04,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a metric depth map to camera-space points (x right, y down, z forward).

    Returns ``points`` (N, 3) float32 in meters and ``colors`` (N, 3) uint8.
    """
    h, w = depth.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    v, u = np.mgrid[0:h:stride, 0:w:stride].astype(np.float32)
    z = depth[::stride, ::stride].astype(np.float32)
    colors = image[::stride, ::stride]

    keep = np.isfinite(z) & (z > 0)
    if edge_threshold is not None:
        keep &= edge_mask(depth, edge_threshold)[::stride, ::stride]
    if max_depth is not None:
        keep &= z < max_depth

    x = (u - cx) * z / f_px
    y = (v - cy) * z / f_px
    points = np.stack([x, y, z], axis=-1)[keep]
    return points.astype(np.float32), colors[keep].astype(np.uint8)


def save_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary little-endian PLY point cloud (viewable in MeshLab, Blender, Preview)."""
    n = len(points)
    vertex = np.empty(
        n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    # Flip to a y-up, z-toward-viewer convention used by most 3D viewers.
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], -points[:, 1], -points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())
