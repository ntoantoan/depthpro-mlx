"""Build README media: photo -> depth wipe -> 3D parallax, as animated WebP (and MP4).

Uses cached depth/focal from outputs/ (written by the sample runs).
"""

import argparse
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from _scene import load_scene
from PIL import Image

from depthpro_mlx.geometry import depth_to_points, layered_points
from depthpro_mlx.io import colorize_depth
from depthpro_mlx.render import orbit_frames

ROOT = Path(__file__).resolve().parents[2]


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    h = out.shape[0]
    scale = h / 600
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
    x, y = int(16 * scale), int(h - 18 * scale)
    overlay = out.copy()
    cv2.rectangle(overlay, (x - 8, y - th - 10), (x + tw + 8, y + 10), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def wipe(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    """Left-to-right wipe from a to b with a soft bright edge."""
    w = a.shape[1]
    xs = np.arange(w)[None, :, None]
    frames = []
    for i in range(1, n + 1):
        t = i / n
        t = t * t * (3 - 2 * t)  # smoothstep
        edge = t * (w + 40) - 20
        alpha = np.clip((edge - xs) / 20 + 0.5, 0, 1)
        f = (b * alpha + a * (1 - alpha)).astype(np.uint8)
        line = np.abs(xs - edge) < 2
        frames.append(np.where(line, 255, f).astype(np.uint8))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--out", default=None)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=60, help="WebP quality")
    parser.add_argument("--story", action="store_true", help="photo -> depth -> 3D sequence (hero)")
    args = parser.parse_args()

    image, depth, f_px, background = load_scene(args.name)
    h, w = image.shape[:2]
    width = args.width
    height = round(h * width / w) // 2 * 2
    stride = max(1, int(w // (2.5 * width)))
    f_out = f_px * width / w

    points, colors = layered_points(depth, image, f_px, background, stride=stride)
    video = list(orbit_frames(points, colors, f_out, width, height, n_frames=args.frames))

    frames = video
    if args.story:
        depth_rgb = colorize_depth(depth)
        depth_points, depth_colors = depth_to_points(depth, depth_rgb, f_px, stride=stride)
        depth_view = next(orbit_frames(depth_points, depth_colors, f_out, width, height, n_frames=args.frames))
        photo = video[0]
        frames = (
            [label(photo, "1 photo")] * 12
            + [label(f, "Depth Pro on MLX") for f in wipe(photo, depth_view, 14)]
            + [label(depth_view, "metric depth, 3 s on M4 Pro")] * 14
            + wipe(depth_view, photo, 12)
            + [label(f, "3D, rendered locally") for f in video]
        )

    # Animated WebP: GitHub renders it inline, ~5x smaller than GIF on textured photos.
    out = Path(args.out or ROOT / "assets" / f"{args.name}.webp")
    out.parent.mkdir(parents=True, exist_ok=True)
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(out, save_all=True, append_images=pil[1:], duration=1000 // args.fps, loop=0, quality=args.quality, method=6)
    iio.imwrite(out.with_suffix(".mp4"), frames, fps=args.fps * 2 if not args.story else 24, codec="libx264", quality=8)
    print(f"{out.name}: {len(frames)} frames {width}x{height}, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
