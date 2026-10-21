"""README wiggle previews: real views along the 63 mm eye baseline (with the LaMa layer).

uv run python tools/media/render_wiggles.py incense_village hue_citadel
"""

import argparse

from _scene import ROOT, load_scene
from PIL import Image

from depthpro_mlx.geometry import layered_points
from depthpro_mlx.spatial import wiggle_views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()

    for name in args.names:
        image, depth, f_px, background = load_scene(name)
        h, w = image.shape[:2]
        width, height = args.width, round(h * args.width / w) // 2 * 2
        points, colors = layered_points(depth, image, f_px, background, stride=max(1, int(w // (2.5 * width))))
        frames = wiggle_views(points, colors, f_px * width / w, width, height, n_frames=args.frames)
        out = ROOT / "assets" / f"wiggle_{name.split('_')[0]}.webp"
        pil = [Image.fromarray(f) for f in frames]
        pil[0].save(out, save_all=True, append_images=pil[1:], duration=40, loop=0, quality=70, method=6)
        print(f"{out.name}: {len(frames)} frames {width}x{height}, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
