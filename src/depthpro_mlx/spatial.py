"""Stereo rendering at a true human baseline -> Apple spatial photo (HEIC), SBS and wiggle.

Metric depth is what makes this physically right: the two eyes are placed
``baseline`` meters apart (63 mm by default), so the reconstructed scene has
the same stereo depth you would have seen standing where the photo was taken.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .render import render_view

HUMAN_BASELINE_M = 0.063
_SWIFT_SOURCE = Path(__file__).with_name("spatial_writer.swift")


def render_stereo(
    points: np.ndarray,
    colors: np.ndarray,
    f_px: float,
    width: int,
    height: int,
    baseline: float = HUMAN_BASELINE_M,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Render parallel left/right views. Returns (left, right, focal_px of the outputs).

    Eyes sit at -/+ baseline/2 around the original camera, looking straight
    ahead. A small zoom-in (from the nearest content's disparity) keeps both
    frames inside the photo.
    """
    z_near = float(np.quantile(points[:, 2], 0.01))
    shift = f_px * (baseline / 2) / z_near  # px, largest per-eye displacement
    f_out = f_px * (1 + 2 * shift / width + 0.01)
    views = []
    for sign in (-1, 1):
        eye = np.array([sign * baseline / 2, 0.0, 0.0], dtype=np.float32)
        views.append(render_view(points, colors, np.eye(3, dtype=np.float32), eye, f_out, width, height))
    return views[0], views[1], f_out


def _writer_binary() -> Path:
    """Compile the Swift helper once and cache it."""
    if shutil.which("swiftc") is None:
        raise RuntimeError("Writing spatial HEIC needs the Swift toolchain (xcode-select --install).")
    cache = Path.home() / ".cache" / "depthpro-mlx"
    binary = cache / "spatial_writer"
    if not binary.exists() or binary.stat().st_mtime < _SWIFT_SOURCE.stat().st_mtime:
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(["swiftc", "-O", str(_SWIFT_SOURCE), "-o", str(binary)], check=True)
    return binary


def write_spatial_heic(
    path: str | Path,
    left: np.ndarray,
    right: np.ndarray,
    f_px: float,
    baseline: float = HUMAN_BASELINE_M,
    disparity_adjustment: float = 0.0,
) -> None:
    """Write a stereo-pair HEIC that Photos shows as a spatial photo (Vision Pro, iPhone)."""
    with tempfile.TemporaryDirectory() as tmp:
        lp, rp = Path(tmp) / "left.png", Path(tmp) / "right.png"
        Image.fromarray(left).save(lp)
        Image.fromarray(right).save(rp)
        subprocess.run(
            [str(_writer_binary()), str(lp), str(rp), str(path), f"{f_px:.3f}", f"{baseline:.4f}",
             f"{disparity_adjustment:.5f}"],
            check=True,
        )


def side_by_side(left: np.ndarray, right: np.ndarray, cross_eyed: bool = False) -> np.ndarray:
    """Parallel (default) or cross-eyed side-by-side stereo image."""
    return np.concatenate([right, left] if cross_eyed else [left, right], axis=1)


def wiggle_frames(left: np.ndarray, right: np.ndarray, steps: int = 6) -> list[np.ndarray]:
    """A smooth left<->right "wiggle" loop: 3D without glasses."""
    frames = []
    for i in range(2 * steps):
        t = i / steps if i < steps else 2 - i / steps
        s = t * t * (3 - 2 * t)
        frames.append((left * (1 - s) + right * s).astype(np.uint8))
    return frames
