import numpy as np

from depthpro_mlx.geometry import depth_to_points, edge_mask, save_ply
from depthpro_mlx.render import look_at, orbit_frames, render_view


def test_unprojection_reprojects_to_the_same_pixel(scene):
    image, depth, f = scene
    points, colors = depth_to_points(depth, image, f, edge_threshold=None)
    h, w = depth.shape
    u = f * points[:, 0] / points[:, 2] + (w - 1) / 2
    v = f * points[:, 1] / points[:, 2] + (h - 1) / 2
    vv, uu = np.mgrid[0:h, 0:w]
    np.testing.assert_allclose(u, uu.ravel(), atol=1e-3)
    np.testing.assert_allclose(v, vv.ravel(), atol=1e-3)
    assert np.array_equal(colors, image.reshape(-1, 3))


def test_edge_mask_removes_a_band_around_depth_jumps(scene):
    _, depth, _ = scene
    keep = edge_mask(depth, band=2)
    assert not keep[60, 59] and not keep[60, 60]  # on both sides of the square's left edge
    assert keep[60, 80] and keep[10, 10]  # far from any edge


def test_identity_view_reproduces_the_photo(scene):
    """At the photo's own viewpoint the render matches the (downsized) photo.

    Uses a smooth image and ~2.5 source points per output pixel, as the CLI does;
    per-pixel noise would measure resampling, not geometry.
    """
    import cv2

    _, depth, f = scene
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    image = np.stack([xx * 255 / w, yy * 255 / h, 128 + 60 * np.sin(xx / 9.0)], -1).astype(np.uint8)
    points, colors = depth_to_points(depth, image, f, edge_threshold=None)
    out_w, out_h = 64, 48
    eye = np.zeros(3, np.float32)
    frame = render_view(
        points, colors, look_at(eye, np.array([0, 0, 1.0], np.float32)), eye, f * out_w / w, out_w, out_h
    )
    ref = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    assert np.abs(frame.astype(int) - ref.astype(int)).mean() < 4


def test_moving_camera_reveals_no_black_holes(scene):
    image, depth, f = scene
    points, colors = depth_to_points(depth, image, f)
    h, w = depth.shape
    frames = list(orbit_frames(points, colors, f, w, h, n_frames=4))
    assert len(frames) == 4
    assert all(fr.shape == (h, w, 3) and (fr.sum(-1) > 0).all() for fr in frames)


def test_ply_header(tmp_path, scene):
    image, depth, f = scene
    points, colors = depth_to_points(depth, image, f)
    save_ply(tmp_path / "x.ply", points, colors)
    data = (tmp_path / "x.ply").read_bytes()
    assert data.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert f"element vertex {len(points)}".encode() in data[:200]
    header_len = data.index(b"end_header\n") + len(b"end_header\n")
    assert len(data) - header_len == len(points) * 15  # 3 floats + 3 bytes
