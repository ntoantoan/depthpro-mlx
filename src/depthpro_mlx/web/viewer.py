"""Self-contained interactive 3D viewer: one HTML file with the photo and depth embedded.

The photo is drawn as a single mesh unprojected from metric depth. Seen from a
new angle, triangles that straddle a depth edge stretch; the fragment shader
detects that stretch (UV barely changes across many pixels) and shows an
inpainted *background* layer there instead of smearing the subject. The
background layer comes from :func:`background_layer`: depth max-filtered so
farther surfaces extend under foreground edges, colors filled from the
farthest neighbouring surface.

The viewer also measures real distances (two clicks) thanks to metric depth.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_TEMPLATE = Path(__file__).with_name("viewer_template.html")


def _encode_jpeg(image: np.ndarray, quality: int = 86) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_inverse_depth(depth: np.ndarray, inv_min: float, inv_max: float) -> str:
    """16-bit inverse depth packed into the R (high) and G (low) bytes of a PNG."""
    inv = 1.0 / np.clip(depth, 1e-3, None)
    q = np.round(np.clip((inv - inv_min) / (inv_max - inv_min), 0, 1) * 65535).astype(np.uint16)
    rgb = np.stack([(q >> 8).astype(np.uint8), (q & 255).astype(np.uint8), np.zeros_like(q, np.uint8)], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def background_layer(
    image: np.ndarray, depth: np.ndarray, radius_frac: float = 0.006, jump: float = 0.06
) -> tuple[np.ndarray, np.ndarray]:
    """Color and depth of what is plausibly *behind* foreground edges.

    ``depth`` must match ``image`` in size. A stretched triangle in the viewer
    only samples texture within about one mesh cell of the edge, so the band
    that needs a background color is narrow: foreground pixels within
    ``radius_frac`` x width of a surface that is more than ``jump`` farther.
    It is filled with a smooth, far-weighted average of nearby background,
    so a stretch reads as a soft gradient of the right tones.
    """
    h, w = depth.shape
    r = max(2, round(radius_frac * w))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    bg_depth = cv2.dilate(depth.astype(np.float32), kernel)
    bg_depth = cv2.GaussianBlur(bg_depth, (0, 0), r / 3)
    band = bg_depth > depth * (1 + jump)
    bg_depth = np.maximum(bg_depth, depth)
    if not band.any():
        return image, bg_depth

    # Soft "farthest" fill: normalized convolution at growing scales, with weights that
    # strongly favour far pixels (z^8). Unlike a hard max this keeps the background's
    # own tones and texture scale instead of flat patches of its brightest/farthest pixel.
    # Normalize by the *local* farthest depth so a distant sky doesn't zero out every weight.
    z = depth.astype(np.float32)
    weight = np.where(band, 0.0, np.clip(z / bg_depth, 0, 1) ** 8).astype(np.float32)
    weighted = image.astype(np.float32) * weight[..., None]
    fill = image.astype(np.float32)  # fallback: keep the photo where no background is in reach
    done = np.zeros(band.shape, bool)
    sigma = max(1.5, 0.004 * w)
    for _ in range(4):
        num = cv2.GaussianBlur(weighted, (0, 0), sigma)
        den = cv2.GaussianBlur(weight, (0, 0), sigma)
        # Accept the smallest scale at which ~10% of the neighbourhood is local background.
        ok = band & ~done & (den > 0.1)
        fill[ok] = num[ok] / den[ok, None]
        done |= ok
        if done[band].all():
            break
        sigma *= 2
    bg_color = image.copy()
    bg_color[band] = np.clip(fill[band], 0, 255).astype(np.uint8)
    # Stretched triangles also reach a little past the band; soften that zone too so a
    # stretch shows a smooth gradient rather than streaks of sharp detail.
    zone = cv2.dilate(band.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    soft = cv2.GaussianBlur(bg_color, (0, 0), max(1.5, 0.003 * w))
    bg_color[zone] = soft[zone]
    return bg_color, bg_depth


def _has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def export_viewer(
    path: str | Path,
    image: np.ndarray,
    depth: np.ndarray,
    f_px: float,
    title: str = "depthpro-mlx",
    max_size: int = 1920,
    depth_size: int = 960,
    grid: int = 960,
    max_parallax: float | None = None,
    depth_ratio: float | None = None,
    background: str | tuple[np.ndarray, np.ndarray, np.ndarray] | None = "auto",
) -> Path:
    """Write ``path``: an .html file that works offline, from file://, and on phones.

    Rendering compresses the depth range around the subject with
    ``z' = pivot * (z / pivot) ** gamma``, choosing ``gamma`` so the near/far
    ratio is at most ``depth_ratio``. Extreme foreground (a leaf at 0.4 m in
    front of a sky) then no longer eats the whole parallax budget, so the edge
    stretch can be capped low (``max_parallax`` x width) while the middle of
    the scene still moves. Measurements always use the true metric depth.

    ``background`` controls what shows where the photo is uncovered:
    ``"lama"`` inpaints a real background layer (needs the ``inpaint`` extra),
    ``"fast"`` blends a blurred fill into stretched areas, ``"auto"`` picks LaMa
    when available, or pass ``(color, depth, mask)`` from
    :func:`depthpro_mlx.inpaint.layered_background`. With a real layer the
    camera may move further (``max_parallax`` 0.03 vs 0.018).
    """
    if isinstance(background, str):
        if background == "auto":
            background = "lama" if _has_onnxruntime() else "fast"
        if background == "lama":
            from ..inpaint import layered_background

            background = layered_background(image, depth)
        elif background != "fast":
            raise ValueError(f"unknown background mode: {background}")
    layered = isinstance(background, tuple)
    max_parallax = max_parallax or (0.03 if layered else 0.018)
    depth_ratio = depth_ratio or (10.0 if layered else 6.0)
    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    color = (
        cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    )
    ds = min(1.0, depth_size / max(h, w))
    small_depth = cv2.resize(depth, (round(w * ds), round(h * ds)), interpolation=cv2.INTER_AREA)

    if layered:
        bg_rgb, bg_z, _ = background
        bg_color = cv2.resize(bg_rgb, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_AREA)
        bg_depth = cv2.resize(bg_z, (small_depth.shape[1], small_depth.shape[0]), interpolation=cv2.INTER_AREA)
    else:  # blurred fill blended into stretched areas of a single mesh
        depth_c = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_LINEAR)
        bg_color, _ = background_layer(color, depth_c)
        bg_depth = None

    inv = 1.0 / np.clip(small_depth, 1e-3, None)
    inv_min, inv_max = float(np.percentile(inv, 0.2)), float(np.percentile(inv, 99.8))
    near, pivot, far = (float(q) for q in np.quantile(depth, [0.01, 0.35, 0.99]))
    gamma = float(min(1.0, np.log(depth_ratio) / np.log(max(far / near, 1.0001))))
    near_r, far_r = (pivot * (z / pivot) ** gamma for z in (near, far))  # compressed range
    radius = min(0.06 * pivot, max_parallax * w / max(f_px * (1 / near_r - 1 / far_r), 1e-9))
    shift = f_px * radius * max(1 / near_r - 1 / pivot, 1 / pivot - 1 / far_r)
    params = {
        "width": w,
        "height": h,
        "f": f_px,
        "grid": grid,
        "invMin": inv_min,
        "invMax": inv_max,
        "near": near,
        "far": far,
        "pivot": pivot,
        "radius": radius,
        "gamma": gamma,
        "crop": float(np.clip(1 + 2.2 * shift / w + 0.02, 1.04, 1.35)),
    }

    html = (
        _TEMPLATE.read_text()
        .replace("__TITLE__", title.replace("<", "").replace(">", ""))
        .replace("__PARAMS__", json.dumps(params))
        .replace("__IMAGE__", _encode_jpeg(color))
        .replace("__DEPTH__", _encode_inverse_depth(small_depth, inv_min, inv_max))
        .replace("__BG_IMAGE__", _encode_jpeg(bg_color, quality=80))
        .replace(
            "__BG_DEPTH_URI__",
            "data:image/png;base64," + _encode_inverse_depth(bg_depth, inv_min, inv_max) if layered else "",
        )
    )
    path = Path(path)
    path.write_text(html)
    return path
