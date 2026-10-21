# depthpro-mlx: notes for Claude

Apple's Depth Pro ported to MLX, plus effects built on its *metric* depth: 3D parallax, dolly zoom,
refocus, Vision Pro spatial photos, measuring, video depth with cinematic refocus, a single-file WebGL
viewer, and a LaMa-inpainted background layer so moving the camera never smears edges.

Owner: Nguyễn Đình Toàn (GitHub `ntoantoan`, HF/PyPI `ntoan199`). Current release: **v0.3.1** (2026-10-21).

## Where things live

| What | Where |
|---|---|
| Code | github.com/ntoantoan/depthpro-mlx (`main`) |
| Package | pypi.org/project/depthpro-mlx (`pip install "depthpro-mlx[inpaint]"`) |
| Docker | `ghcr.io/ntoantoan/depthpro-mlx:{X.Y.Z,latest,latest-cuda}`, built by `.github/workflows/docker.yml` on `v*` tags |
| MLX weights | huggingface.co/ntoan199/DepthPro-mlx (`depth_pro.safetensors`, model card from `hf/`) |
| Live demo | huggingface.co/spaces/ntoan199/depthpro-mlx-demo (static Space, uploaded from `demo/`) |
| Secrets | `.env` (gitignored): `HF_TOKEN`, `PYPI_TOKEN`, `HF_USER`, `GITHUB_USER`. `gh` is logged in via keychain. |

## Layout

- `src/depthpro_mlx/model/`: `layers.py` (DINOv2 ViT, bilinear resize matching torch), `depth_pro.py`
  (encoder/decoder/FOV, low-memory `patch_batch` + strip decoding `tiles`), `convert.py` (torch → safetensors).
- `render/` (point splatting, depth-aware push-pull hole fill, camera paths), `effects/` (dolly, refocus),
  `spatial/` (stereo, wiggle, Swift HEIC writer), `web/` (viewer + `viewer_template.html`),
  `inpaint.py` (LaMa ONNX background layer), `video.py` (stabilized video depth, cinematic), `geometry.py`,
  `io.py`, `hub.py`, `cli.py`.
- `tools/bench/` (parity, benchmarks), `tools/media/` (README media and demo builders; `_scene.py` caches
  depth and LaMa layers in `outputs/`).
- Not in git: `weights/`, `checkpoints/`, `third_party/ml-depth-pro`, `samples/`, `outputs/`, `demo/scenes|thumbs`, `dist/`.

## Commands

```bash
uv sync                                   # dev env (default group); `--group reference` for Apple's torch code
uv run pytest -m "not slow"               # 29 tests, no weights needed (what CI runs)
uv run pytest -m slow                     # parity vs PyTorch (needs weights + outputs/ref_depth_fp32.npy)
uv run ruff check src tests tools && uv run ruff format src tests tools
uv run python tools/media/build_demo.py   # rebuild demo/ (then upload to the Space with huggingface_hub)
docker build -t depthpro-mlx .            # CPU image
```

## Release checklist

1. Bump `__version__` in `src/depthpro_mlx/__init__.py` and `CITATION.cff`; add a CHANGELOG section + compare links.
2. Ruff + tests green; commit (Conventional Commits, author Nguyễn Đình Toàn <ntoan199@gmail.com>, Co-Authored-By trailer).
3. `git tag -a vX.Y.Z` → push `main` and the tag (the tag triggers the GHCR Docker build).
4. `rm -rf dist && uv build && uv publish --token "$PYPI_TOKEN" dist/*`.
5. `gh release create vX.Y.Z --latest --notes-file ... dist/*`.
6. If the model card or demo changed, upload `hf/README.md` / `demo/` with `huggingface_hub` (`HF_TOKEN` from `.env`).

## Conventions and gotchas

- Parity is sacred: changes under `model/` must keep AbsRel < 1e-4 vs. the PyTorch reference in fp32.
- Defaults: float16 + full batch on Metal; float32 + `patch_batch=8`, `tiles=12` on CPU (peak 7.6 GB vs 15 GB).
- Docker CPU image swaps `mlx-cpu`'s bundled reference BLAS for OpenBLAS (7 → 400 GFLOP/s). Needs ≥ 10 GB RAM
  for the container. On a Mac, use the native package instead (no Metal in Docker). HEIC needs macOS.
- README image links are absolute raw.githubusercontent URLs so PyPI renders them.
- Demo media: only openly licensed photos (Wikimedia Commons), credited in `assets/CREDITS.md`,
  `demo/CREDITS.md`, `samples/CREDITS.md`. The user prefers no identifiable people in the foreground
  (the cyclist photo was replaced by a bamboo bicycle; wheel measures 66 cm).
- Wikimedia rate-limits fast downloads (429): send a descriptive User-Agent and honor `retry-after`.
- Restarting Docker Desktop stops containers from other projects (e.g. RepoVault): ask first.

## Open items

- User should rotate the PyPI and HF tokens (they were pasted in chat) and upload `assets/social-preview.jpg`
  as the GitHub social preview.
- Optional: replace remaining media with visible faces (Hà Nội street vendor scene; train-street and
  basket-maker videos); write launch posts (Reddit, X, HN, Vietnamese AI groups); renderer on MLX/Metal
  (~10x faster than numpy); test the CUDA image on a real GPU; contribute weights to `mlx-community`.
