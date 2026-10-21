import json
import re
import subprocess
import sys

import numpy as np

from depthpro_mlx.io import colorize_depth
from depthpro_mlx.web import background_layer, export_viewer


def test_background_layer_fills_only_near_edges_and_never_black(scene):
    image, depth, _ = scene
    bg, bg_depth = background_layer(image, depth)
    assert bg.shape == image.shape and (bg_depth >= depth).all()
    assert np.array_equal(bg[5, 5], image[5, 5])  # far from the square: untouched
    edge = bg[60, 62]  # just inside the square's edge
    assert not np.array_equal(edge, image[60, 62]) and edge.sum() > 0


def test_export_viewer_writes_a_self_contained_page(tmp_path, scene):
    image, depth, f = scene
    page = export_viewer(tmp_path / "v.html", image, depth, f, title="<test>")
    html = page.read_text()
    assert "data:image/jpeg;base64," in html and "data:image/png;base64," in html
    assert "__IMAGE__" not in html and "<test>" not in html  # placeholders replaced, title escaped
    params = json.loads(re.search(r"const P = (\{.*?\});", html).group(1))
    assert params["width"] == 160 and 0 < params["gamma"] <= 1 and params["radius"] > 0


def test_colorize_depth_is_rgb_uint8(scene):
    _, depth, _ = scene
    rgb = colorize_depth(depth)
    assert rgb.shape == (*depth.shape, 3) and rgb.dtype == np.uint8


def test_cli_rejects_unknown_effect_before_loading_the_model(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "depthpro_mlx.cli", str(tmp_path / "x.jpg"), "--effects", "bogus"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2 and "unknown effect" in result.stderr
