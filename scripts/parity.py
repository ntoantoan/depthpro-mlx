"""Compare MLX output against the PyTorch reference depth (saved by bench_torch.py --save)."""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import depth_pro  # reference package, used only for identical image + EXIF loading
from depthpro_mlx import load_model, predict

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default=str(ROOT / "third_party/ml-depth-pro/data/example.jpg"))
    parser.add_argument("--ref", default=str(ROOT / "outputs/ref_depth_fp32.npy"))
    parser.add_argument("--weights", default=str(ROOT / "weights/depth_pro.safetensors"))
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    model = load_model(args.weights, dtype=getattr(mx, args.dtype))
    image, _, f_px = depth_pro.load_rgb(args.image)

    times = []
    for i in range(args.runs + 1):
        t0 = time.perf_counter()
        out = predict(model, image, f_px=f_px)
        if i > 0:
            times.append(time.perf_counter() - t0)

    depth = out["depth"]
    ref = np.load(args.ref)
    abs_rel = np.mean(np.abs(depth - ref) / ref)
    delta1 = np.mean(np.maximum(depth / ref, ref / depth) < 1.01)
    print(f"dtype={args.dtype}  image {image.shape[1]}x{image.shape[0]}")
    print(f"infer: median {np.median(times):.3f}s  min {np.min(times):.3f}s  peak mem {mx.get_peak_memory() / 2**30:.2f} GiB")
    print(f"depth range: {depth.min():.3f}..{depth.max():.3f} m   ref: {ref.min():.3f}..{ref.max():.3f} m")
    print(f"vs torch fp32:  AbsRel {abs_rel:.2e}   max |diff| {np.abs(depth - ref).max():.2e} m   within 1%: {delta1 * 100:.2f}%")


if __name__ == "__main__":
    main()
