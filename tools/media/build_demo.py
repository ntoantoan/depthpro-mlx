"""Build the live demo site (demo/scenes/*.html, demo/thumbs/*.jpg) from the sample photos.

    uv run python tools/media/build_demo.py            # needs samples/*.jpg (see samples/CREDITS.md)

The generated files are not committed; the site is published to a Hugging Face
static Space (see README). Depth is computed with the MLX model, or reused from
outputs/<name>_depth.npy when present.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from depthpro_mlx import load_model, predict
from depthpro_mlx.io import load_image
from depthpro_mlx.web import export_viewer

ROOT = Path(__file__).resolve().parents[2]
SCENES = [
    ("incense_village", "Incense village · Quảng Phú Cầu"),
    ("hoi_an_lanterns", "Lantern street · Hội An"),
    ("hue_citadel", "Imperial City · Huế"),
    ("douc_langur", "Douc langurs · Sơn Trà"),
    ("ninh_binh_swan_boats", "Swan boats · Ninh Bình"),
    ("hai_van_pass", "Hải Vân Pass · Đà Nẵng"),
    ("hanoi_fruit_vendor", "Street vendor · Hà Nội"),
    ("vietnamese_coffee", "Cà phê phin"),
]


def depth_for(name: str, image: np.ndarray, f_exif, model_cache: dict) -> tuple[np.ndarray, float]:
    cached = ROOT / "outputs" / f"{name}_depth.npy"
    if cached.exists():
        return np.load(cached), float(np.load(ROOT / "outputs" / f"{name}_f.npy"))
    if "model" not in model_cache:
        model_cache["model"] = load_model()
    out = predict(model_cache["model"], image, f_px=f_exif)
    return out["depth"], out["focallength_px"]


def main() -> None:
    demo = ROOT / "demo"
    (demo / "scenes").mkdir(parents=True, exist_ok=True)
    (demo / "thumbs").mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    for name, title in SCENES:
        image, f_exif = load_image(ROOT / "samples" / f"{name}.jpg")
        depth, f_px = depth_for(name, image, f_exif, cache)
        page = export_viewer(demo / "scenes" / f"{name}.html", image, depth, f_px, title=title)
        h, w = image.shape[:2]
        thumb = cv2.resize(image, (480, round(480 * h / w)), interpolation=cv2.INTER_AREA)
        Image.fromarray(thumb).save(demo / "thumbs" / f"{name}.jpg", quality=82)
        print(f"{name:22s} {page.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
