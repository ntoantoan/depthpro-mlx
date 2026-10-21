"""depthpro-mlx: single photo -> metric depth, 3D point cloud and cinematic effects.

Examples:
    depthpro-mlx photo.jpg                          # depth.png + 3D parallax video
    depthpro-mlx photo.jpg --effects all --ply      # + dolly zoom, refocus, point cloud
    depthpro-mlx photo.jpg --effects refocus --gif  # just the focus pull, also as GIF
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import mlx.core as mx
import numpy as np
from PIL import Image

from . import load_model, predict
from .effects import Refocus, dolly_zoom_frames
from .geometry import depth_to_points, save_ply
from .render import orbit_frames
from .spatial import render_stereo, side_by_side, wiggle_frames, write_spatial_heic

EFFECTS = ("orbit", "dolly", "refocus")


def load_image(path: str | Path) -> tuple[np.ndarray, float | None]:
    """RGB uint8 image plus focal length in pixels from EXIF (35mm-equivalent), if present."""
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        import pillow_heif

        pillow_heif.register_heif_opener()
    img = Image.open(path)
    exif = img.getexif().get_ifd(0x8769)  # Exif IFD
    f_35mm = exif.get(0xA405)  # FocalLengthIn35mmFilm
    img = np.array(img.convert("RGB"))
    f_px = None
    if f_35mm:
        h, w = img.shape[:2]
        f_px = float(f_35mm) * np.sqrt(w**2 + h**2) / np.sqrt(36**2 + 24**2)
    return img, f_px


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Inverse depth, robustly normalized, turbo colormap. Returns RGB uint8."""
    inv = 1.0 / np.clip(depth, 1e-3, None)
    lo, hi = np.percentile(inv, [1, 99])
    norm = np.clip((inv - lo) / (hi - lo + 1e-8), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def fit_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (width, round(h * width / w) // 2 * 2), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(prog="depthpro-mlx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=None, help="default: <image>_depthpro/")
    parser.add_argument("--weights", type=Path, default=None, help="local .safetensors (default: download from the Hub)")
    parser.add_argument("--f-px", type=float, default=None, help="focal length in pixels (overrides EXIF / estimate)")
    parser.add_argument("--ply", action="store_true", help="export a colored point cloud")
    parser.add_argument("--effects", default="orbit",
                        help="comma list of orbit,dolly,refocus, or 'all' / 'none' (default: orbit)")
    parser.add_argument("--gif", action="store_true", help="also export each video as a GIF")
    parser.add_argument("--spatial", action="store_true",
                        help="Apple spatial photo (stereo HEIC at a 63 mm baseline) + side-by-side + wiggle")
    parser.add_argument("--web", action="store_true", help="self-contained interactive 3D viewer (viewer.html)")
    parser.add_argument("--width", type=int, default=1280, help="video width")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--amplitude", type=float, default=0.06, help="camera motion, fraction of subject depth")
    args = parser.parse_args()
    effects = EFFECTS if args.effects == "all" else () if args.effects == "none" else tuple(args.effects.split(","))
    unknown = set(effects) - set(EFFECTS)
    if unknown:
        parser.error(f"unknown effect(s): {', '.join(sorted(unknown))}")


    out_dir = args.out_dir or args.image.with_name(args.image.stem + "_depthpro")
    out_dir.mkdir(parents=True, exist_ok=True)

    image, f_exif = load_image(args.image)
    f_px = args.f_px or f_exif

    t0 = time.perf_counter()
    model = load_model(args.weights)
    print(f"loaded model in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    out = predict(model, image, f_px=f_px)
    depth, f_px = out["depth"], out["focallength_px"]
    print(f"depth in {time.perf_counter() - t0:.2f}s  range {depth.min():.2f}-{depth.max():.2f} m  f={f_px:.0f}px"
          f" ({'EXIF' if f_exif and not args.f_px else 'given' if args.f_px else 'estimated'})")
    del model
    mx.clear_cache()

    np.save(out_dir / "depth.npy", depth)
    Image.fromarray(colorize_depth(depth)).save(out_dir / "depth.png")

    if args.ply:
        stride = max(1, round(max(image.shape[:2]) / 3000))
        points, colors = depth_to_points(depth, image, f_px, stride=stride)
        save_ply(out_dir / "pointcloud.ply", points, colors)
        print(f"point cloud: {len(points):,} points")

    h, w = image.shape[:2]
    width = min(args.width, w)
    height = round(h * width / w) // 2 * 2
    f_out = f_px * width / w
    if {"orbit", "dolly"} & set(effects):
        # ~2.5 source points per output pixel: enough for 2x supersampling plus zoom.
        points, colors = depth_to_points(depth, image, f_px, stride=max(1, int(w // (2.5 * width))))

    for effect in effects:
        t0 = time.perf_counter()
        if effect == "orbit":
            frames = list(orbit_frames(points, colors, f_out, width, height, n_frames=args.frames, amplitude=args.amplitude))
            name = "3d"
        elif effect == "dolly":
            frames = list(dolly_zoom_frames(points, colors, f_out, width, height, n_frames=args.frames))
            name = "dolly_zoom"
        else:
            small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            d = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
            frames = [f for f, _ in Refocus(small, d).sweep(d, n_frames=args.frames)]
            name = "refocus"
        iio.imwrite(out_dir / f"{name}.mp4", frames, fps=30, codec="libx264", quality=8, macro_block_size=1)
        if args.gif:
            gif = [fit_width(f, 640) for f in frames[::2]]
            iio.imwrite(out_dir / f"{name}.gif", gif, duration=1000 / 15, loop=0)
        print(f"{name}.mp4: {len(frames)} frames in {time.perf_counter() - t0:.0f}s")

    if args.spatial:
        t0 = time.perf_counter()
        sw = min(w, 2560)
        sh = round(h * sw / w) // 2 * 2
        sp, sc = depth_to_points(depth, image, f_px, stride=max(1, int(w // (2.5 * sw))))
        left, right, f_stereo = render_stereo(sp, sc, f_px * sw / w, sw, sh)
        Image.fromarray(side_by_side(left, right)).save(out_dir / "stereo_sbs.jpg", quality=92)
        wiggle = [Image.fromarray(fit_width(f, 640)) for f in wiggle_frames(left, right)]
        wiggle[0].save(out_dir / "wiggle.webp", save_all=True, append_images=wiggle[1:], duration=60, loop=0, quality=80)
        try:
            write_spatial_heic(out_dir / "spatial.heic", left, right, f_stereo)
            print(f"spatial.heic + stereo_sbs.jpg + wiggle.webp in {time.perf_counter() - t0:.0f}s")
        except (RuntimeError, OSError, subprocess.CalledProcessError) as err:
            print(f"stereo_sbs.jpg + wiggle.webp written; spatial.heic skipped: {err}")

    if args.web:
        from .web import export_viewer

        export_viewer(out_dir / "viewer.html", image, depth, f_px, title=args.image.stem)
        print("viewer.html: open it in a browser, drag to look around")

    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
