import numpy as np

from depthpro_mlx.video import cinematic, stabilize


def _clip(n=8, h=96, w=128, seed=0):
    """A static textured scene with per-frame depth noise (what single-image models produce)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    frame = np.stack([xx * 2, yy * 2, 128 + 60 * np.sin(xx / 5.0) * np.cos(yy / 7.0)], -1).astype(np.uint8)
    true = np.where((xx > 40) & (xx < 90), 1.0, 3.0).astype(np.float32)
    depths = [true * np.exp(rng.normal(0, 0.05, true.shape)).astype(np.float32) for _ in range(n)]
    return [frame] * n, depths, true


def test_stabilize_reduces_flicker_without_bias():
    frames, depths, true = _clip()
    smooth = stabilize(frames, depths, keep=0.6)
    raw_flicker = np.std(np.log(np.stack(depths[1:])), axis=0).mean()
    smooth_flicker = np.std(np.log(np.stack(smooth[3:])), axis=0).mean()
    assert smooth_flicker < 0.6 * raw_flicker
    assert abs(np.log(smooth[-1] / true).mean()) < 0.02  # still centered on the true depth


def test_cinematic_pull_sweeps_between_near_and_far():
    frames, depths, _ = _clip(n=6)
    out = list(cinematic(frames, depths, mode="pull"))
    assert len(out) == 6 and out[0][0].shape == frames[0].shape
    focus = [f for _, f in out]
    assert min(focus) < 1.5 and max(focus) > 1.5
