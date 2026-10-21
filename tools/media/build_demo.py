"""Build the live demo site (demo/scenes/*.html, demo/thumbs/*.jpg) from the sample photos.

    uv run python tools/media/build_demo.py        # needs samples/*.jpg (see samples/CREDITS.md)

The generated files are not committed; the site is published to a Hugging Face
static Space (see README). Depth and the LaMa background layer are cached in
outputs/ by the shared scene loader.
"""

import cv2
from _scene import ROOT, load_scene
from PIL import Image

from depthpro_mlx.web import export_viewer

SCENES = [
    ("incense_village", "Incense village · Quảng Phú Cầu"),
    ("hoi_an_lanterns", "Lantern street · Hội An"),
    ("bamboo_bicycle", "Bamboo bicycle · Hội An"),
    ("hue_gate", "Imperial City gate · Huế"),
    ("douc_langur", "Douc langurs · Sơn Trà"),
    ("ninh_binh_swan_boats", "Swan boats · Ninh Bình"),
    ("hai_van_pass", "Hải Vân Pass · Đà Nẵng"),
    ("hanoi_fruit_vendor", "Street vendor · Hà Nội"),
    ("vietnamese_coffee", "Cà phê phin"),
]


def main() -> None:
    demo = ROOT / "demo"
    (demo / "scenes").mkdir(parents=True, exist_ok=True)
    (demo / "thumbs").mkdir(parents=True, exist_ok=True)
    for name, title in SCENES:
        image, depth, f_px, background = load_scene(name)
        page = export_viewer(demo / "scenes" / f"{name}.html", image, depth, f_px, title=title, background=background)
        h, w = image.shape[:2]
        thumb = cv2.resize(image, (480, round(480 * h / w)), interpolation=cv2.INTER_AREA)
        Image.fromarray(thumb).save(demo / "thumbs" / f"{name}.jpg", quality=82)
        print(f"{name:22s} {page.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
