"""Layered background via LaMa inpainting: what lies *behind* the foreground near depth edges.

Moving the camera around a single photo reveals surfaces the photo never saw.
Instead of stretching or blurring, the foreground is removed near every depth
edge and LaMa (big-lama, Apache-2.0, ONNX port ``Carve/LaMa-ONNX``) paints the
background in. The result is a second layer, with its own color *and* depth,
that renderers draw behind the photo.

Needs the ``inpaint`` extra: ``pip install "depthpro-mlx[inpaint]"``.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

LAMA_REPO = "Carve/LaMa-ONNX"
LAMA_FILE = "lama_fp32.onnx"
TILE = 512  # LaMa's fixed input size


@lru_cache(maxsize=1)
def _session():
    try:
        import onnxruntime as ort
    except ImportError as err:  # pragma: no cover - depends on the extra
        raise RuntimeError('LaMa inpainting needs onnxruntime: pip install "depthpro-mlx[inpaint]"') from err
    from huggingface_hub import hf_hub_download

    # CPU beats the CoreML provider here: CoreML only takes ~half of LaMa's FFT-heavy graph.
    return ort.InferenceSession(hf_hub_download(LAMA_REPO, LAMA_FILE), providers=["CPUExecutionProvider"])


def lama_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint ``mask`` (True = fill) in an RGB uint8 image of any size.

    Works on overlapping 512 px tiles (only those touching the mask) and
    feathers them together, so large images keep their native detail.
    """
    h, w = mask.shape
    if not mask.any():
        return image.copy()
    sess = _session()
    stride = TILE * 3 // 4
    pad_h, pad_w = max(0, TILE - h), max(0, TILE - w)
    img = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    msk = np.pad(mask, ((0, pad_h), (0, pad_w)))
    H, W = msk.shape
    ys = sorted({*range(0, max(H - TILE, 0) + 1, stride), max(H - TILE, 0)})
    xs = sorted({*range(0, max(W - TILE, 0) + 1, stride), max(W - TILE, 0)})

    ramp = np.minimum(np.arange(TILE) + 1, TILE - np.arange(TILE)).astype(np.float32)
    weight_tile = np.minimum.outer(ramp, ramp)  # feather toward tile borders
    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for y in ys:
        for x in xs:
            m = msk[y : y + TILE, x : x + TILE]
            if not m.any():
                continue
            tile = img[y : y + TILE, x : x + TILE].astype(np.float32) / 255.0
            feeds = {
                "image": tile.transpose(2, 0, 1)[None],
                "mask": m.astype(np.float32)[None, None],
            }
            out = sess.run(None, feeds)[0][0].transpose(1, 2, 0)
            acc[y : y + TILE, x : x + TILE] += out * weight_tile[..., None]
            wsum[y : y + TILE, x : x + TILE] += weight_tile

    result = img.astype(np.float32)
    covered = wsum > 0
    result[covered] = acc[covered] / wsum[covered, None]
    result = np.where(msk[..., None], result, img.astype(np.float32))
    return np.clip(result[:h, :w], 0, 255).astype(np.uint8)


def occlusion_mask(
    depth: np.ndarray, radius_frac: float = 0.025, jump: float = 0.06, step: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Foreground pixels within ``radius_frac`` x width of a depth *discontinuity*.

    Only real steps count: a pixel whose neighbour ``step`` px away is more than
    ``jump`` (relative) farther. Gently sloping ground or hills never qualify,
    however much farther they get over a distance. The far side's depth is
    spread over the radius, and pixels sufficiently nearer than it form the mask.

    Returns ``(mask, background_depth)``; ``background_depth`` is where the
    inpainted layer sits (the far side, smoothed), equal to ``depth`` elsewhere.
    """
    h, w = depth.shape
    # Beyond ~20x the median depth everything is "infinitely far" for parallax; clamping
    # stops noisy sky depth (900 m vs 10 km) from registering as discontinuities.
    z = np.minimum(depth.astype(np.float32), 20 * float(np.median(depth)))
    far_neighbour = np.zeros_like(z)
    for dy, dx in ((0, step), (0, -step), (step, 0), (-step, 0)):
        # Shift with edge padding (np.roll would wrap the bottom row onto the sky at the top).
        padded = np.pad(z, ((max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0))), mode="edge")
        shifted = padded[max(-dy, 0) : max(-dy, 0) + h, max(-dx, 0) : max(-dx, 0) + w]
        is_step = shifted > z * (1 + jump)
        far_neighbour = np.where(is_step, np.maximum(far_neighbour, shifted), far_neighbour)

    r = max(2, round(radius_frac * w))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    far = cv2.dilate(far_neighbour, kernel)  # far-side depth, spread around each discontinuity
    mask = far > z * (1 + jump)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # cover anti-aliased edges
    # Smooth the far depth among masked pixels only, so foreground depth never leaks in
    # (that would leave background points floating between the two layers).
    m = (mask & (far > 0)).astype(np.float32)  # pixels the dilation added have no far value yet
    far_smooth = cv2.GaussianBlur(far * m, (0, 0), r / 3) / np.maximum(cv2.GaussianBlur(m, (0, 0), r / 3), 1e-6)
    mask &= far_smooth > z * (1 + jump / 2)  # keep only pixels with something genuinely behind them
    background_depth = np.where(mask, far_smooth, depth).astype(np.float32)
    return mask, background_depth


def layered_background(
    image: np.ndarray, depth: np.ndarray, work_size: int = 1024, radius_frac: float = 0.025, jump: float = 0.06
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Background layer for a photo: ``(color, depth, mask)`` at the photo's resolution.

    The mask and inpainting run at ``work_size`` on the long side (LaMa's sweet
    spot), then the fill is upsampled and used only inside the mask.
    """
    h, w = depth.shape
    s = min(1.0, work_size / max(h, w))
    small = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else image
    small_depth = cv2.resize(depth, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) if s < 1 else depth
    mask_s, _ = occlusion_mask(small_depth, radius_frac, jump)
    filled_s = lama_inpaint(small, mask_s)

    mask, bg_depth = occlusion_mask(depth, radius_frac, jump)
    filled = cv2.resize(filled_s, (w, h), interpolation=cv2.INTER_CUBIC) if s < 1 else filled_s
    color = np.where(mask[..., None], filled, image)
    return color, bg_depth, mask
