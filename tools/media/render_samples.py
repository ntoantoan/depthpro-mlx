"""Render 3D parallax videos for sample images, reusing cached depth from outputs/."""

import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from depthpro_mlx.geometry import depth_to_points
from depthpro_mlx.io import load_image
from depthpro_mlx.render import orbit_frames

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--amplitude", type=float, default=0.06)
    args = parser.parse_args()

    for name in args.names:
        image, _ = load_image(ROOT / "samples" / f"{name}.jpg")
        depth = np.load(ROOT / "outputs" / f"{name}_depth.npy")
        f_px = float(np.load(ROOT / "outputs" / f"{name}_f.npy"))
        h, w = image.shape[:2]
        # Keep >= ~2.5 source points per output pixel (2x supersampling + zoom margin).
        stride = max(1, int(w // (2.5 * args.width)))
        points, colors = depth_to_points(depth, image, f_px, stride=stride)
        width, height = args.width, round(h * args.width / w) // 2 * 2
        t0 = time.perf_counter()
        frames = list(
            orbit_frames(
                points, colors, f_px * width / w, width, height, n_frames=args.frames, amplitude=args.amplitude
            )
        )
        out = ROOT / "outputs" / f"{name}_3d.mp4"
        iio.imwrite(out, frames, fps=30, codec="libx264", quality=8)
        print(
            f"{name}: {len(points):,} pts, {len(frames)} frames in {time.perf_counter() - t0:.0f}s -> {out.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
