"""Benchmark MLX variants: dtype, compile, quantization, FOV head skipping."""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import depth_pro
from depthpro_mlx import load_model, preprocess
from depthpro_mlx.layers import resize_bilinear

ROOT = Path(__file__).resolve().parents[1]


def run(fn, x, runs):
    times = []
    for i in range(runs + 1):
        t0 = time.perf_counter()
        mx.eval(fn(x))
        if i > 0:
            times.append(time.perf_counter() - t0)
    return np.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default=str(ROOT / "third_party/ml-depth-pro/data/example.jpg"))
    parser.add_argument("--weights", default=str(ROOT / "weights/depth_pro.safetensors"))
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    image, _, _ = depth_pro.load_rgb(args.image)
    ref = None

    for name, dtype, compile_, quant_bits, skip_fov in [
        ("fp16", mx.float16, False, None, False),
        ("fp16 + no-fov", mx.float16, False, None, True),
        ("fp16 + compile + no-fov", mx.float16, True, None, True),
        ("8-bit + no-fov", mx.float16, False, 8, True),
        ("4-bit + no-fov", mx.float16, False, 4, True),
    ]:
        model = load_model(args.weights, dtype=dtype)
        if skip_fov:
            del model.fov
        if quant_bits:
            nn.quantize(model, group_size=64, bits=quant_bits, class_predicate=lambda p, m: isinstance(m, nn.Linear))
            mx.eval(model.parameters())
        x = resize_bilinear(preprocess(image, dtype)[None], 1536, 1536)
        mx.eval(x)

        fwd = (lambda inp: model(inp)[0])
        if compile_:
            fwd = mx.compile(fwd)
        mx.reset_peak_memory()
        t = run(fwd, x, args.runs)
        out = np.array(fwd(x).astype(mx.float32))
        if ref is None:
            ref = out
        rel = np.mean(np.abs(out - ref) / (np.abs(ref) + 1e-6))
        print(f"{name:28s} {t:6.3f}s   peak {mx.get_peak_memory() / 2**30:5.2f} GiB   rel-diff vs fp16 {rel:.2e}")
        del model
        mx.clear_cache()


if __name__ == "__main__":
    main()
