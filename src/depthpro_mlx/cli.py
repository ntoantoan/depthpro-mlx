"""depthpro-mlx: metric depth for a photo or a video, plus 3D and cinematic effects.

Examples:
    depthpro-mlx photo.jpg                          # depth.png + 3D parallax video
    depthpro-mlx photo.jpg --effects all --ply      # + dolly zoom, refocus, point cloud
    depthpro-mlx photo.jpg --effects refocus --gif  # just the focus pull, also as GIF
    depthpro-mlx clip.mp4 --duration 6              # stabilized video depth + cinematic refocus
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
from .geometry import depth_to_points, layered_points, save_ply
from .io import colorize_depth, fit_width, load_image
from .render import orbit_frames
from .spatial import render_stereo, side_by_side, wiggle_views, write_spatial_heic
from .video import VIDEO_SUFFIXES, cinematic, read_frames, stabilize, video_depth

EFFECTS = ("orbit", "dolly", "refocus")


def run_video(args: argparse.Namespace) -> None:
    """Video input: stabilized per-frame depth and cinematic refocus."""
    out_dir = args.out_dir or args.image.with_name(args.image.stem + "_depthpro")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = read_frames(args.image, start=args.start, duration=args.duration, max_side=args.max_side)
    print(f"{len(frames)} frames at {fps:.1f} fps")

    model = load_model(args.weights)
    t0 = time.perf_counter()
    depths, f_px = video_depth(model, frames, f_px=args.f_px, progress=lambda i, n: print(f"  depth {i}/{n}", end="\r"))
    depths = stabilize(frames, depths)
    print(f"depth in {time.perf_counter() - t0:.0f}s (f={f_px:.0f}px, fixed for the clip)")
    del model
    mx.clear_cache()

    np.savez_compressed(out_dir / "depth.npz", depth=np.stack(depths).astype(np.float16), f_px=f_px)
    inv_range = tuple(np.percentile(np.concatenate([1 / d[::8, ::8].ravel() for d in depths]), [1, 99]))
    depth_video = [colorize_depth(d, inv_range) for d in depths]  # one colour scale for the whole clip
    iio.imwrite(out_dir / "depth.mp4", depth_video, fps=fps, codec="libx264", quality=8, macro_block_size=2)
    if args.cinematic != "none":
        focus_xy = tuple(float(v) for v in args.focus.split(","))
        cine = [f for f, _ in cinematic(frames, depths, mode=args.cinematic, focus_xy=focus_xy)]
        iio.imwrite(out_dir / "cinematic.mp4", cine, fps=fps, codec="libx264", quality=8, macro_block_size=2)
    print(f"outputs -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="depthpro-mlx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=None, help="default: <image>_depthpro/")
    parser.add_argument(
        "--weights", type=Path, default=None, help="local .safetensors (default: download from the Hub)"
    )
    parser.add_argument("--f-px", type=float, default=None, help="focal length in pixels (overrides EXIF / estimate)")
    parser.add_argument("--ply", action="store_true", help="export a colored point cloud")
    parser.add_argument(
        "--effects", default="orbit", help="comma list of orbit,dolly,refocus, or 'all' / 'none' (default: orbit)"
    )
    parser.add_argument("--gif", action="store_true", help="also export each video as a GIF")
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="Apple spatial photo (stereo HEIC at a 63 mm baseline) + side-by-side + wiggle",
    )
    parser.add_argument("--web", action="store_true", help="self-contained interactive 3D viewer (viewer.html)")
    parser.add_argument(
        "--inpaint",
        choices=["auto", "on", "off"],
        default="auto",
        help="LaMa background layer for uncovered regions (needs the 'inpaint' extra; auto = when installed)",
    )
    parser.add_argument("--width", type=int, default=1280, help="video width")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--amplitude", type=float, default=0.06, help="camera motion, fraction of subject depth")
    video = parser.add_argument_group("video input")
    video.add_argument("--start", type=float, default=0.0, help="seconds into the clip")
    video.add_argument("--duration", type=float, default=None, help="seconds to process (default: all)")
    video.add_argument(
        "--cinematic",
        choices=["track", "pull", "none"],
        default="track",
        help="track: keep the subject at --focus sharp; pull: rack focus near to far",
    )
    video.add_argument("--focus", default="0.5,0.5", help="subject position for --cinematic track, as x,y in 0..1")
    video.add_argument("--max-side", type=int, default=960, help="process frames at this size (long side)")
    args = parser.parse_args()
    if args.image.suffix.lower() in VIDEO_SUFFIXES:
        return run_video(args)
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
    print(
        f"depth in {time.perf_counter() - t0:.2f}s  range {depth.min():.2f}-{depth.max():.2f} m  f={f_px:.0f}px"
        f" ({'EXIF' if f_exif and not args.f_px else 'given' if args.f_px else 'estimated'})"
    )
    del model
    mx.clear_cache()

    np.save(out_dir / "depth.npy", depth)
    Image.fromarray(colorize_depth(depth)).save(out_dir / "depth.png")

    if args.ply:
        stride = max(1, round(max(image.shape[:2]) / 3000))
        points, colors = depth_to_points(depth, image, f_px, stride=stride)
        save_ply(out_dir / "pointcloud.ply", points, colors)
        print(f"point cloud: {len(points):,} points")

    background = None
    needs_layer = bool({"orbit", "dolly"} & set(effects)) or args.spatial or args.web
    if needs_layer and args.inpaint != "off":
        from .web.viewer import _has_onnxruntime

        if _has_onnxruntime():
            from .inpaint import layered_background

            t0 = time.perf_counter()
            background = layered_background(image, depth)
            print(
                f"background layer (LaMa) in {time.perf_counter() - t0:.0f}s, {background[2].mean() * 100:.0f}% of pixels"
            )
        elif args.inpaint == "on":
            parser.error('--inpaint on needs onnxruntime: pip install "depthpro-mlx[inpaint]"')

    h, w = image.shape[:2]
    width = min(args.width, w)
    height = round(h * width / w) // 2 * 2
    f_out = f_px * width / w
    if {"orbit", "dolly"} & set(effects):
        # ~2.5 source points per output pixel: enough for 2x supersampling plus zoom.
        points, colors = layered_points(depth, image, f_px, background, stride=max(1, int(w // (2.5 * width))))

    for effect in effects:
        t0 = time.perf_counter()
        if effect == "orbit":
            frames = list(
                orbit_frames(points, colors, f_out, width, height, n_frames=args.frames, amplitude=args.amplitude)
            )
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
        sp, sc = layered_points(depth, image, f_px, background, stride=max(1, int(w // (2.5 * sw))))
        left, right, f_stereo = render_stereo(sp, sc, f_px * sw / w, sw, sh)
        Image.fromarray(side_by_side(left, right)).save(out_dir / "stereo_sbs.jpg", quality=92)
        ww = min(w, 640)
        wh = round(h * ww / w) // 2 * 2
        wp, wc = layered_points(depth, image, f_px, background, stride=max(1, int(w // (2.5 * ww))))
        wiggle = [Image.fromarray(f) for f in wiggle_views(wp, wc, f_px * ww / w, ww, wh)]
        wiggle[0].save(
            out_dir / "wiggle.webp", save_all=True, append_images=wiggle[1:], duration=40, loop=0, quality=80
        )
        try:
            write_spatial_heic(out_dir / "spatial.heic", left, right, f_stereo)
            print(f"spatial.heic + stereo_sbs.jpg + wiggle.webp in {time.perf_counter() - t0:.0f}s")
        except (RuntimeError, OSError, subprocess.CalledProcessError) as err:
            print(f"stereo_sbs.jpg + wiggle.webp written; spatial.heic skipped: {err}")

    if args.web:
        from .web import export_viewer

        export_viewer(
            out_dir / "viewer.html", image, depth, f_px, title=args.image.stem, background=background or "fast"
        )
        print("viewer.html: open it in a browser, drag to look around")

    print(f"outputs -> {out_dir}")


if __name__ == "__main__":
    main()
