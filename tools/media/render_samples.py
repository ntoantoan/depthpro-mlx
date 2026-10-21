"""Render 3D parallax videos for sample images, reusing cached depth from outputs/."""

import argparse
import time
from pathlib import Path

import imageio.v3 as iio
from _scene import load_scene

from depthpro_mlx.geometry import layered_points
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
        image, depth, f_px, background = load_scene(name)
        h, w = image.shape[:2]
        # Keep >= ~2.5 source points per output pixel (2x supersampling + zoom margin).
        stride = max(1, int(w // (2.5 * args.width)))
        points, colors = layered_points(depth, image, f_px, background, stride=stride)
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
