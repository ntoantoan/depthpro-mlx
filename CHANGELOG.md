# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.1] - 2026-10-21

### Fixed
- Docker: images based on Ubuntu 24.04 (the CUDA flavor) failed to build because uid 1000 is taken by the
  default `ubuntu` user. The release workflow no longer cancels the CPU image when another flavor fails.

## [0.3.0] - 2026-10-21

### Added
- **Docker images** for Linux (`ghcr.io/ntoantoan/depthpro-mlx`): multi-stage build, non-root user,
  Hugging Face cache volume, amd64 + arm64 CPU images and an experimental CUDA image (`:latest-cuda`),
  published by a release workflow; CI builds the CPU image on every push.
- **Low-memory mode**: `load_model(patch_batch=...)` runs the ViT over the 37 tiles in chunks, and the
  768/384/192 px decoder levels and the head are decoded in halo strips. Identical output (AbsRel
  1.6e-6), float32 peak 15 GB → 7.6 GB. Enabled automatically on CPU.
- `default_dtype()`: float16 on a GPU, float32 on CPU (BLAS only accelerates float32).
- `depthpro-mlx --version`.

### Changed
- Docker CPU image swaps `mlx-cpu`'s bundled reference BLAS for OpenBLAS (~7 → ~400 GFLOP/s).
- Dolly zoom renders from a denser point set (the background is magnified up to ~1.5x).
- Demo media without identifiable people in the foreground: a bamboo bicycle (wheel measured at
  66 cm vs. 66–67 cm real), a gate and a corridor of the Imperial City replace the cyclist photo.

## [0.2.1] - 2026-10-21

### Added
- Published on PyPI: `pip install "depthpro-mlx[inpaint]"`.

### Changed
- README: per-component license table; Depth Pro weights are under Apple's terms (not OSI-approved,
  commercial use not explicitly addressed). Absolute image links so the README renders on PyPI.

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

[Unreleased]: https://github.com/ntoantoan/depthpro-mlx/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/ntoantoan/depthpro-mlx/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ntoantoan/depthpro-mlx/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/ntoantoan/depthpro-mlx/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ntoantoan/depthpro-mlx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ntoantoan/depthpro-mlx/releases/tag/v0.1.0
