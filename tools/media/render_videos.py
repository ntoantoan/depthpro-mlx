"""Video demos: stabilized depth (photo | depth side by side) and cinematic refocus.

    uv run python tools/media/render_videos.py      # needs samples/video/*_1080p.webm

Clips and their credits are listed in CLIPS / samples/video/sources.json.
"""

import time

import cv2
import imageio.v3 as iio
import numpy as np
from _scene import ROOT
from make_hero import label
from PIL import Image

from depthpro_mlx import load_model
from depthpro_mlx.io import colorize_depth
from depthpro_mlx.video import cinematic, read_frames, stabilize, video_depth

# name, source file, start (s), duration (s), cinematic mode, focus point (x, y) for "track"
CLIPS = [
    ("train_street", "train_street_b_1080p.webm", 20.0, 6.0, "pull", (0.5, 0.5)),
    ("train_departure", "train_street_short_1080p.webm", 0.0, 7.0, "pull", (0.5, 0.5)),
    ("basket_maker", "basket_maker_1080p.webm", 15.0, 6.0, "track", (0.45, 0.55)),
]


def save(frames: list[np.ndarray], name: str, fps: float) -> None:
    iio.imwrite(ROOT / "outputs" / f"{name}.mp4", frames, fps=fps, codec="libx264", quality=8, macro_block_size=2)
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(
        ROOT / "assets" / f"{name}.webp",
        save_all=True,
        append_images=pil[1:],
        duration=int(1000 / fps),
        loop=0,
        quality=58,
        method=6,
    )
    print(f"  {name}.webp {(ROOT / 'assets' / f'{name}.webp').stat().st_size / 1e6:.1f} MB", flush=True)


def main() -> None:
    model = load_model()
    for name, src, start, duration, mode, focus_xy in CLIPS:
        t0 = time.perf_counter()
        frames, fps = read_frames(ROOT / "samples" / "video" / src, start=start, duration=duration, max_side=960)
        frames, fps = frames[::2], fps / 2  # 15 fps keeps processing time and file size down
        depths, f_px = video_depth(model, frames)
        depths = stabilize(frames, depths)
        print(f"{name}: {len(frames)} frames, f={f_px:.0f}px, depth in {time.perf_counter() - t0:.0f}s", flush=True)

        # Photo | depth with a shared colour scale across the clip (so colours mean the same distance).
        inv_range = tuple(np.percentile(np.concatenate([1 / d[::8, ::8].ravel() for d in depths]), [1, 99]))
        sbs = [np.concatenate([f, colorize_depth(d, inv_range)], axis=1) for f, d in zip(frames, depths, strict=True)]
        sbs = [cv2.resize(s, (s.shape[1] * 360 // s.shape[0] // 2 * 2, 360), interpolation=cv2.INTER_AREA) for s in sbs]
        save([label(s, "video depth, Depth Pro on MLX") for s in sbs], f"{name}_depth", fps)

        cine = [label(f, f"focus {z:4.1f} m") for f, z in cinematic(frames, depths, mode=mode, focus_xy=focus_xy)]
        cine = [
            cv2.resize(c, (c.shape[1] * 480 // c.shape[0] // 2 * 2, 480), interpolation=cv2.INTER_AREA) for c in cine
        ]
        save(cine, f"{name}_cinematic", fps)
        print(f"{name}: done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
