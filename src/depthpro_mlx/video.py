"""Video depth with temporal stabilization, and "cinematic mode" refocus for any video.

Depth Pro is a single-image model: run frame by frame its depth flickers, and
its focal-length estimate wobbles, which changes the metric scale. Here the
focal length is fixed for the whole clip (median over the first frames) and
each new depth map is blended with the previous one warped by optical flow,
only where the flow is forward-backward consistent, so moving objects don't
leave ghosts.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from .effects import Refocus

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def read_frames(
    path: str | Path, start: float = 0.0, duration: float | None = None, max_side: int | None = 1280
) -> tuple[list[np.ndarray], float]:
    """Decode ``[start, start + duration)`` seconds as RGB uint8 frames, optionally downsized. Returns (frames, fps)."""
    meta = iio.immeta(path)
    fps = float(meta.get("fps", 30.0))
    first = int(round(start * fps))
    last = None if duration is None else first + int(round(duration * fps))
    frames = []
    for i, frame in enumerate(iio.imiter(path)):
        if i < first:
            continue
        if last is not None and i >= last:
            break
        if max_side and max(frame.shape[:2]) > max_side:
            s = max_side / max(frame.shape[:2])
            frame = cv2.resize(
                frame,
                (round(frame.shape[1] * s) // 2 * 2, round(frame.shape[0] * s) // 2 * 2),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(np.ascontiguousarray(frame[..., :3]))
    return frames, fps


def _flow(a: np.ndarray, b: np.ndarray, dis) -> np.ndarray:
    return dis.calc(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None)


def _warp(x: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Sample ``x`` (previous frame) at ``p + flow(p)``: brings it into the current frame."""
    h, w = flow.shape[:2]
    grid = np.dstack(np.meshgrid(np.arange(w), np.arange(h))).astype(np.float32)
    return cv2.remap(x, grid + flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def stabilize(frames: list[np.ndarray], depths: list[np.ndarray], keep: float = 0.6) -> list[np.ndarray]:
    """Temporally smooth per-frame depth in log space along optical flow.

    ``keep`` is the weight of the warped history where the flow is reliable
    (forward-backward error < 1 px); it falls to 0 where the flow fails.
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = [depths[0]]
    prev_log = np.log(np.clip(depths[0], 1e-3, None))
    for t in range(1, len(frames)):
        cur_log = np.log(np.clip(depths[t], 1e-3, None))
        back = _flow(frames[t], frames[t - 1], dis)  # current -> previous
        fwd = _flow(frames[t - 1], frames[t], dis)
        err = np.linalg.norm(back + _warp(fwd, back), axis=-1)  # cycle consistency
        w = keep * np.clip(1.5 - err, 0, 1)
        warped = _warp(prev_log, back)
        smoothed = w * warped + (1 - w) * cur_log
        out.append(np.exp(smoothed).astype(np.float32))
        prev_log = smoothed
    return out


def video_depth(model, frames: list[np.ndarray], f_px: float | None = None, calib_frames: int = 5, progress=None):
    """Depth for every frame with one focal length for the whole clip. Returns (depths, f_px)."""
    from . import predict

    if f_px is None:
        picks = np.linspace(0, len(frames) - 1, min(calib_frames, len(frames))).astype(int)
        f_px = float(np.median([predict(model, frames[i])["focallength_px"] for i in picks]))
    depths = []
    for i, frame in enumerate(frames):
        depths.append(predict(model, frame, f_px=f_px)["depth"])
        if progress:
            progress(i + 1, len(frames))
    return depths, f_px


def cinematic(
    frames: list[np.ndarray],
    depths: list[np.ndarray],
    mode: str = "track",
    focus_xy: tuple[float, float] = (0.5, 0.5),
    strength_frac: float = 0.012,
) -> Iterator[tuple[np.ndarray, float]]:
    """Refocus every frame; yields (frame, focus distance in meters).

    ``track`` keeps focus on the subject around ``focus_xy`` (relative coords),
    smoothed over time like a focus puller; ``pull`` racks focus from near to
    far and back over the clip.
    """
    n = len(frames)
    h, w = depths[0].shape
    near = float(np.median([np.quantile(d, 0.03) for d in depths]))
    far = float(np.median([np.quantile(d, 0.75) for d in depths]))
    max_radius = strength_frac * w
    strength = max_radius / max(1 / near - 1 / far, 1e-6)
    cx, cy = int(focus_xy[0] * w), int(focus_xy[1] * h)
    r = max(4, int(0.06 * min(h, w)))
    focus = None
    for t in range(n):
        if mode == "pull":
            s = 0.5 - 0.5 * np.cos(2 * np.pi * t / n)
            target = 1.0 / ((1 - s) / near + s / far)
            focus = target
        else:
            target = float(np.median(depths[t][max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]))
            focus = target if focus is None else 0.85 * focus + 0.15 * target  # focus puller lag
        yield Refocus(frames[t], depths[t], max_radius=max_radius, levels=10).render(focus, strength), focus
