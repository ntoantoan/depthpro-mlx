"""Render dolly-zoom and refocus clips for samples (cached depth in outputs/)."""

import argparse
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image

from make_hero import label
from depthpro_mlx.cli import load_image
from depthpro_mlx.effects import Refocus, dolly_zoom_frames
from depthpro_mlx.geometry import depth_to_points

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    image, _ = load_image(ROOT / "samples" / f"{name}.jpg")
    depth = np.load(ROOT / "outputs" / f"{name}_depth.npy")
    f_px = float(np.load(ROOT / "outputs" / f"{name}_f.npy"))
    return image, depth, f_px


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("effect", choices=["dolly", "refocus"])
    parser.add_argument("names", nargs="+")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--label", action="store_true", help="overlay effect name / focus distance")
    parser.add_argument("--webp", default=None, help="also write assets/<name> as animated WebP")
    parser.add_argument("--webp-step", type=int, default=2, help="keep every n-th frame in the WebP")
    args = parser.parse_args()

    for name in args.names:
        image, depth, f_px = load(name)
        h, w = image.shape[:2]
        width, height = args.width, round(h * args.width / w) // 2 * 2
        t0 = time.perf_counter()
        if args.effect == "dolly":
            points, colors = depth_to_points(depth, image, f_px, stride=max(1, int(w // (2.5 * width))))
            frames = list(dolly_zoom_frames(points, colors, f_px * width / w, width, height, n_frames=args.frames))
            if args.label:
                frames = [label(f, "dolly zoom from 1 photo") for f in frames]
        else:
            small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            d = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
            frames = []
            for f, focus in Refocus(small, d).sweep(d, n_frames=args.frames):
                frames.append(label(f, f"focus {focus:4.1f} m") if args.label else f)
        out = ROOT / "outputs" / f"{name}_{args.effect}.mp4"
        iio.imwrite(out, frames, fps=30, codec="libx264", quality=8, macro_block_size=1)
        if args.webp:
            webp = ROOT / "assets" / args.webp
            pil = [Image.fromarray(f) for f in frames[:: args.webp_step]]
            pil[0].save(webp, save_all=True, append_images=pil[1:], duration=1000 * args.webp_step // 30,
                        loop=0, quality=72, method=6)
            print(f"  -> {webp.name} {webp.stat().st_size / 1e6:.1f} MB")
        print(f"{name} {args.effect}: {len(frames)} frames in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
