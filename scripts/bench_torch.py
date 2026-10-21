"""Baseline: reference PyTorch Depth Pro on MPS / CPU."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import depth_pro
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

ROOT = Path(__file__).resolve().parents[1]


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default=str(ROOT / "third_party/ml-depth-pro/data/example.jpg"))
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--save", default=None, help="Save depth as .npy for parity checks")
    args = parser.parse_args()

    device = torch.device(args.device)
    precision = torch.half if args.fp16 else torch.float32
    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = str(ROOT / "checkpoints/depth_pro.pt")

    t0 = time.perf_counter()
    model, transform = depth_pro.create_model_and_transforms(config, device=device, precision=precision)
    model.eval()
    print(f"load: {time.perf_counter() - t0:.1f}s")

    image, _, f_px = depth_pro.load_rgb(args.image)
    x = transform(image)

    times = []
    with torch.no_grad():
        for i in range(args.runs + 1):
            sync(device)
            t0 = time.perf_counter()
            pred = model.infer(x, f_px=f_px)
            sync(device)
            dt = time.perf_counter() - t0
            if i > 0:  # first run is warm-up
                times.append(dt)

    depth = pred["depth"].float().cpu().numpy()
    print(f"image: {image.shape[1]}x{image.shape[0]}  device={args.device}  fp16={args.fp16}")
    print(f"infer: median {np.median(times):.3f}s  min {np.min(times):.3f}s  ({args.runs} runs)")
    print(f"depth: min {depth.min():.2f}m  max {depth.max():.2f}m  f_px {float(pred['focallength_px']):.1f}")
    if torch.backends.mps.is_available() and args.device == "mps":
        print(f"mps memory: {torch.mps.driver_allocated_memory() / 2**30:.2f} GiB")
    if args.save:
        np.save(args.save, depth)


if __name__ == "__main__":
    main()
