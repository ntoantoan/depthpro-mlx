"""Shared loader for the media tools: photo, cached depth and cached LaMa background layer."""

from pathlib import Path

import numpy as np

from depthpro_mlx.io import load_image

ROOT = Path(__file__).resolve().parents[2]


def load_scene(name: str, layered: bool = True):
    """Return ``image, depth, f_px, background`` for ``samples/<name>.jpg``.

    Depth comes from ``outputs/<name>_depth.npy`` (computed by the model if
    missing); the background layer is cached in ``outputs/<name>_layer.npz``.
    """
    image, f_exif = load_image(ROOT / "samples" / f"{name}.jpg")
    depth_path, f_path = ROOT / "outputs" / f"{name}_depth.npy", ROOT / "outputs" / f"{name}_f.npy"
    if depth_path.exists():
        depth, f_px = np.load(depth_path), float(np.load(f_path))
    else:
        from depthpro_mlx import load_model, predict

        out = predict(load_model(), image, f_px=f_exif)
        depth, f_px = out["depth"], out["focallength_px"]
        depth_path.parent.mkdir(exist_ok=True)
        np.save(depth_path, depth)
        np.save(f_path, f_px)

    background = None
    if layered:
        layer_path = ROOT / "outputs" / f"{name}_layer.npz"
        if layer_path.exists():
            z = np.load(layer_path)
            background = (z["color"], z["depth"], z["mask"])
        else:
            from depthpro_mlx.inpaint import layered_background

            background = layered_background(image, depth)
            np.savez_compressed(layer_path, color=background[0], depth=background[1], mask=background[2])
    return image, depth, f_px, background
