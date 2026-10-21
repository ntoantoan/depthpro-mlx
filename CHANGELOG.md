# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Wiggle previews no longer stutter: `spatial.wiggle_views()` renders real views gliding along the 63 mm
  baseline (with the background layer) instead of cross-fading the two eye images, which doubled edges
  mid-way. Replaces `wiggle_frames()`.

## [0.2.0] - 2026-10-21

### Changed
- Package split into subpackages: `model`, `render`, `effects`, `spatial`, `web`, plus `hub` and `io`.
  Public API (`load_model`, `predict`, `preprocess`, `DepthPro`) is unchanged.
- Converter moved to `python -m depthpro_mlx.model.convert`.
- `scripts/` renamed to `tools/` (`tools/bench`, `tools/media`).
- Dependency groups: `test` (CI), `dev` (default), `reference` (Apple's PyTorch code for parity).

### Added
- **Inpainted background layer** (`depthpro_mlx.inpaint`, `inpaint` extra): LaMa removes the foreground near depth
  discontinuities and paints what is behind it; videos, dolly zoom, stereo and the web viewer render it behind the
  photo, so moving the camera no longer smears edges. CLI `--inpaint auto|on|off`.
- Web viewer draws two layers (photo fades out where stretched) and allows ~1.7x more camera motion with a layer.
- **Video input** (`depthpro_mlx.video`): per-frame depth with a clip-wide focal length, optical-flow stabilization,
  and cinematic-mode refocus (`track` a subject or `pull` near to far). The CLI detects video files:
  `depthpro-mlx clip.mp4 --duration 6 --cinematic track`.
- `colorize_depth(..., inv_range=...)` for a fixed colour scale across frames.
- Unit tests (pytest), GitHub Actions CI on Apple Silicon, ruff, pre-commit.

### Fixed
- Occlusion masks: shifts no longer wrap around the image (the bottom row was compared with the sky), and
  background depth is averaged over valid far-side pixels only (no points floating between layers).

## [0.1.0] - 2026-10-21

### Added
- MLX port of Apple Depth Pro with a weight converter; parity AbsRel 1.7e-6 (fp32), 19% faster than PyTorch MPS (fp16).
- Converted weights on the Hugging Face Hub, downloaded on first use.
- 3D photo rendering: point cloud (`.ply`), parallax orbit video with depth-aware hole filling.
- Dolly zoom and post-capture refocus with metric focus distance.
- Apple spatial photos (stereo HEIC at a 63 mm baseline), side-by-side and wiggle previews.
- Single-file WebGL viewer with depth-range compression, stretch-aware background fill and a metric measuring tool.
- `depthpro-mlx` CLI and a live demo on Hugging Face Spaces.

[Unreleased]: https://github.com/ntoantoan/depthpro-mlx/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ntoantoan/depthpro-mlx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ntoantoan/depthpro-mlx/releases/tag/v0.1.0
