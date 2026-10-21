# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Package split into subpackages: `model`, `render`, `effects`, `spatial`, `web`, plus `hub` and `io`.
  Public API (`load_model`, `predict`, `preprocess`, `DepthPro`) is unchanged.
- Converter moved to `python -m depthpro_mlx.model.convert`.
- `scripts/` renamed to `tools/` (`tools/bench`, `tools/media`).
- Dependency groups: `test` (CI), `dev` (default), `reference` (Apple's PyTorch code for parity).

### Added
- Unit tests (pytest), GitHub Actions CI on Apple Silicon, ruff, pre-commit.

## [0.1.0] - 2026-10-21

### Added
- MLX port of Apple Depth Pro with a weight converter; parity AbsRel 1.7e-6 (fp32), 19% faster than PyTorch MPS (fp16).
- Converted weights on the Hugging Face Hub, downloaded on first use.
- 3D photo rendering: point cloud (`.ply`), parallax orbit video with depth-aware hole filling.
- Dolly zoom and post-capture refocus with metric focus distance.
- Apple spatial photos (stereo HEIC at a 63 mm baseline), side-by-side and wiggle previews.
- Single-file WebGL viewer with depth-range compression, stretch-aware background fill and a metric measuring tool.
- `depthpro-mlx` CLI and a live demo on Hugging Face Spaces.

[Unreleased]: https://github.com/ntoantoan/depthpro-mlx/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ntoantoan/depthpro-mlx/releases/tag/v0.1.0
