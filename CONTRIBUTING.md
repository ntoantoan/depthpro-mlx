# Contributing

Thanks for helping! Issues and pull requests are welcome.

## Setup

Requires an Apple Silicon Mac (MLX runs on Metal) and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ntoantoan/depthpro-mlx && cd depthpro-mlx
uv sync                  # package + dev tools (pytest, ruff, pre-commit, torch)
uv run pre-commit install
```

## Checks

```bash
uv run pytest -m "not slow"     # fast unit tests, no weights needed (what CI runs)
uv run ruff check src tests tools && uv run ruff format src tests tools
```

### Parity with Apple's implementation

Changes to `src/depthpro_mlx/model/` must keep numerical parity with the PyTorch reference:

```bash
bash third_party/get_reference.sh                         # apple/ml-depth-pro + checkpoint
uv sync --group reference
uv run python -m depthpro_mlx.model.convert               # -> weights/depth_pro.safetensors
uv run python tools/bench/bench_torch.py --save outputs/ref_depth_fp32.npy
uv run pytest -m slow                                     # AbsRel < 1e-4 in fp32
```

## Layout

| Path | What |
|---|---|
| `src/depthpro_mlx/model/` | Depth Pro network (`layers.py`, `depth_pro.py`) and the checkpoint converter |
| `src/depthpro_mlx/render/` | point splatting, hole filling, camera paths |
| `src/depthpro_mlx/effects/` | dolly zoom, refocus |
| `src/depthpro_mlx/spatial/` | stereo rendering and the Swift HEIC writer |
| `src/depthpro_mlx/web/` | single-file WebGL viewer |
| `src/depthpro_mlx/{hub,io,geometry,cli}.py` | weights lookup, image IO, unprojection, CLI |
| `tools/bench/` | benchmarks and parity scripts |
| `tools/media/` | README media and demo-site builders |
| `tests/` | pytest suite |

## Commits

[Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`), one logical change per commit. Update `CHANGELOG.md` under *Unreleased*.

Please don't commit weights, checkpoints, sample photos or generated outputs; `.gitignore` covers them.
