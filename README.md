<div align="center">

# depthpro-mlx

**One photo → metric depth → 3D, dolly zoom, refocus and Vision Pro spatial photos. Fully on your Mac.**

Apple's [Depth Pro](https://github.com/apple/ml-depth-pro) ported to [MLX](https://github.com/ml-explore/mlx), plus cinematic effects that only work because the depth is in *meters*, with no GPU server and no PyTorch.

<img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/hero.webp" width="100%" alt="Photo to Depth Pro depth map to 3D parallax video, incense village in Vietnam">

[![CI](https://github.com/ntoantoan/depthpro-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/ntoantoan/depthpro-mlx/actions/workflows/ci.yml)
[![HF weights](https://img.shields.io/badge/🤗%20weights-ntoan199%2FDepthPro--mlx-yellow)](https://huggingface.co/ntoan199/DepthPro-mlx)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX-black?logo=apple)
![Python](https://img.shields.io/badge/python-3.12+-blue)

</div>

## Effects

All of these come from **one ordinary photo**.

| Refocus after the shot | Dolly zoom ("vertigo") |
|---|---|
| <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/refocus_hoi_an.webp" width="100%" alt="Focus pull across a lantern street in Hoi An with round bokeh"> | <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/dolly_zoom_gate.webp" width="100%" alt="Dolly zoom: a gate of the Imperial City keeps its size while the gardens behind its arches swell"> |
| A thin-lens focus pull. Because depth is metric, the label is the **real focus distance**. Lights bloom into round bokeh like a fast lens. | Hitchcock's camera move: pull back while zooming in, so the subject keeps its size and the background swells. Normally this needs a dolly track. |
| <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/refocus_langur.webp" width="100%" alt="Rack focus from leaves to a red-shanked douc langur"> | <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/hai_van_pass.webp" width="100%" alt="3D parallax at Hai Van pass, leaves in front of the bay"> |
| Rack focus from foreground leaves to a red-shanked douc langur, 4.9 m → 18.9 m. | 3D parallax with depth-aware disocclusion filling. |

### Spatial photos for Apple Vision Pro and iPhone

`--spatial` renders a left and a right eye **63 mm apart, the average human interpupillary distance**, and writes a stereo-pair HEIC with per-eye camera intrinsics and extrinsics, the format Photos uses for spatial photos. Because the depth is metric, the stereo depth matches what you would have seen standing there. Relative-depth models can only guess this scale.

| Wiggle preview (left ↔ right eye) | |
|---|---|
| <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/wiggle_incense.webp" width="100%" alt="Left/right eye wiggle stereogram of the incense village"> | <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/wiggle_bamboo.webp" width="100%" alt="Left/right eye wiggle stereogram of a bamboo bicycle in Hoi An"> |

AirDrop `spatial.heic` to a Vision Pro or iPhone 15 Pro or later to view it in 3D. A parallel side-by-side `stereo_sbs.jpg` is written too.

### Measure real distances in a photo

Metric depth turns every photo into something you can measure. In the viewer, press **Measure** and click two points:

<img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/measure_bicycle.jpg" width="100%" alt="Measuring the rear wheel of a bamboo bicycle as 66 cm">

Sanity check: the rear wheel of this bamboo mountain bike measures **66 cm**, and a 26″ MTB wheel with its tire is 66–67 cm across. That comes from a single photo with no reference object and no LiDAR; the focal length was estimated by the model too. Measurements parallel to the image are the reliable kind; separations along the viewing direction pick up depth noise, which grows with distance. Points at nearly the same depth are treated as one plane for that reason. Measurements are shareable: `viewer.html?measure=u1,v1,u2,v2`.

### Video: stable metric depth and cinematic mode

Feed it a video. Depth Pro runs per frame with **one focal length for the whole clip** (so meters stay meters), and the depth is stabilized along optical flow, blended only where the flow is forward-backward consistent, so moving people don't leave ghosts. Then "cinematic mode" refocuses the footage like a focus puller: `track` keeps a subject sharp, `pull` racks from near to far.

| Cinematic mode · Hà Nội train street | Video depth · the train leaves | Tracking focus · basket maker |
|---|---|---|
| <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/train_street_cinematic.webp" width="100%" alt="Focus pulls from a tourist at 0.9 m to the passing train at 3.8 m"> | <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/train_departure_depth.webp" width="100%" alt="Stabilized depth of a crowd on motorbikes as the train leaves"> | <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/basket_maker_cinematic.webp" width="100%" alt="Focus follows a craftsman weaving a basket"> |

```bash
depthpro-mlx clip.mp4 --duration 6 --cinematic track --focus 0.45,0.55
```

### Interactive 3D in the browser

**[▶ Try the live demo](https://huggingface.co/spaces/ntoan199/depthpro-mlx-demo)**: nine scenes from Vietnam. Drag, move the mouse, or tilt your phone; press **Depth** or **Measure**.

`--web` writes a single self-contained `viewer.html` (WebGL2, no dependencies, ~3 MB) with the photo, depth and background layer embedded. It works offline from `file://` and is easy to share; `tools/media/build_demo.py` rebuilds the demo site.

### No smearing: an inpainted background layer

Moving the camera reveals things the photo never saw. Instead of stretching the subject over them, the foreground near every depth edge is removed and [LaMa](https://github.com/advimman/lama) paints in what is behind it. That gives a second layer with its own color and depth, drawn behind the photo by every renderer: video, dolly zoom, stereo and the web viewer. Only real depth *discontinuities* count, so gently sloping ground is left alone. With the layer the camera can also move about 1.7× further.

## Why

- **First MLX port of Depth Pro.** Numerically matches the official PyTorch model (AbsRel 1.7e-6 in fp32) and runs **19% faster** than PyTorch on MPS.
- **Metric depth, sharp edges.** Depth in meters, with hair, leaves and fur resolved. That is what makes the 3D look real and the bokeh fall off correctly.
- **Photo → effects in one command.** Point cloud (`.ply`), 3D parallax, dolly zoom and refocus videos, all rendered locally.
- **No smearing.** A LaMa-inpainted background layer fills what the camera uncovers.
- **Video too.** Stable metric depth per frame and a cinematic-mode refocus for any clip.

## Quickstart

```bash
pip install "depthpro-mlx[inpaint]"
depthpro-mlx photo.jpg --effects all --spatial --web --ply
```

The first run downloads the converted weights (1.9 GB) from the Hub, plus LaMa (208 MB) for the background layer. Without the `inpaint` extra everything still works, with a blurred fill instead of an inpainted layer (`--inpaint off` forces that). Outputs land in `photo_depthpro/`:

| File | What |
|---|---|
| `depth.png` / `depth.npy` | colorized depth / raw depth in meters |
| `pointcloud.ply` | colored point cloud (open in MeshLab, Blender, Preview) |
| `3d.mp4` | parallax 3D video (`--effects orbit`, the default) |
| `dolly_zoom.mp4` | vertigo shot (`--effects dolly`), best with a clear subject and a background that isn't too far away |
| `refocus.mp4` | focus pull near → far (`--effects refocus`) |
| `spatial.heic`, `stereo_sbs.jpg`, `wiggle.webp` | spatial photo, side-by-side stereo, wiggle preview (`--spatial`, needs the Swift toolchain for the HEIC) |
| `viewer.html` | interactive 3D viewer, single file (`--web`) |
| video input: `depth.mp4`, `cinematic.mp4`, `depth.npz` | stabilized depth video (one colour scale for the clip), refocused video, raw depth in meters |

Add `--gif` for GIF copies.

### Docker (Linux servers)

```bash
docker run --rm -v "$PWD:/data" -v depthpro-cache:/cache \
  ghcr.io/ntoantoan/depthpro-mlx photo.jpg --web --effects refocus
```

Images for `linux/amd64` and `linux/arm64` run MLX's CPU backend; `:latest-cuda` uses MLX's CUDA backend (amd64, NVIDIA GPU, `--gpus all`; experimental, built but not yet tested on a GPU). Build locally with `docker build -t depthpro-mlx .`.

Things to know:

- **On a Mac, don't use Docker.** Containers run in a Linux VM without Metal. Natively, Metal takes 3 s per photo; MLX's CPU backend needs ~2.5 min (measured on M4 Pro CPU cores, the same float32 low-memory path the image uses).
- **CPU inference runs in float32 and needs about 8 GB of RAM**, so give the container ≥ 10 GB (Docker Desktop: Settings → Resources). It peaks at 7.6 GB thanks to strip-wise decoding; a naive float32 run needs 15 GB.
- The image replaces the reference BLAS bundled with `mlx-cpu` by OpenBLAS: float32 matmul goes from ~7 to ~400 GFLOP/s.
- `spatial.heic` needs macOS (ImageIO). In the container `--spatial` writes the side-by-side and wiggle previews only.

### Python

```python
import numpy as np
from PIL import Image
from depthpro_mlx import load_model, predict

model = load_model()
image = np.array(Image.open("photo.jpg").convert("RGB"))
out = predict(model, image)          # pass f_px=... if you know the focal length
out["depth"]                         # (H, W) float32, meters
out["focallength_px"]                # estimated when not given
```

iPhone photos carry the focal length in EXIF; the CLI reads it automatically for true metric scale.

## Gallery

| Swan boats, Ninh Bình | Long corridor, Imperial City, Huế | Douc langurs, Sơn Trà |
|---|---|---|
| <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/swan_boats.webp" width="100%" alt="3D parallax of abandoned swan boats"> | <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/hue_corridor.webp" width="100%" alt="3D parallax of a long wooden corridor in the Imperial City of Hue"> | <img src="https://raw.githubusercontent.com/ntoantoan/depthpro-mlx/main/assets/douc_langur.webp" width="100%" alt="3D parallax of douc langurs on a branch"> |

## Benchmarks

MacBook Pro M4 Pro (24 GB), 3024×2268 input, median of 5 runs:

| Backend | Precision | Time | Peak memory | Error vs. PyTorch fp32 (AbsRel) |
|---|---|---|---|---|
| PyTorch MPS (official) | fp32 | 4.60 s | 12.8 GB | – |
| PyTorch MPS (official) | fp16 | 3.76 s | 7.0 GB | – |
| **MLX (this repo)** | fp32 | 4.17 s | 15.0 GB | 1.7e-6 |
| **MLX (this repo)** | **fp16** | **3.06 s** | 7.5 GB | 6.2e-4 |
| MLX (this repo) | bf16 | 3.06 s | 7.5 GB | 2.9e-3 |

Depth Pro runs a DINOv2-L ViT over 37 overlapping 384 px tiles, so it is compute-bound. In our tests, 8-bit and 4-bit weight quantization saved memory but ran slower (3.4 s).

Reproduce:

```bash
bash third_party/get_reference.sh            # clones apple/ml-depth-pro + checkpoint
uv sync --group reference
uv run python tools/bench/bench_torch.py --save outputs/ref_depth_fp32.npy
uv run python -m depthpro_mlx.model.convert   # torch checkpoint -> weights/depth_pro.safetensors
uv run python tools/bench/parity.py --dtype float16
```

## How it works

1. **Depth.** Depth Pro builds a 3-level image pyramid, encodes 5×5 + 3×3 + 1 overlapping tiles with a shared DINOv2-L, fuses them with a DPT-style decoder, and predicts canonical inverse depth plus the field of view. Scaling the output by the focal length gives metric depth.
2. **3D.** Pixels are unprojected with the pinhole model. Points in a thin band around depth discontinuities are dropped, since they carry foreground color at background depth ("flying pixels").
3. **Render.** A z-buffered point splatter with 2× supersampling. Disocclusions are filled by a depth-aware push-pull: color and depth are packed into one key, a max-pyramid is built, and holes take the *farthest* surface on their border, so the background never borrows the subject's color. Camera radius is capped so near/far parallax stays under 3.5% of the frame, and the zoom-in is derived from the largest shift so the frame never leaves the photo.
4. **Dolly zoom.** The camera pulls back by `b` while the focal length scales by `(z_s + b) / z_s`, keeping the subject at depth `z_s` fixed in size. `b` is capped so near content never shrinks below 85%.
5. **Refocus.** Thin-lens circle of confusion `r = k·|1/z − 1/z_f|` on a precomputed stack of disk-blurred images in linear light, with a highlight boost for bokeh. The focus sweeps uniformly in diopters.
6. **Spatial photo.** Two parallel pinhole cameras at ±31.5 mm render the eyes; a small Swift helper writes them with ImageIO as a `StereoPair` group with `SimplifiedPinhole` intrinsics and extrinsics, compiled once and cached in `~/.cache/depthpro-mlx`.
7. **Web viewer.** Two grid meshes (photo, and background layer behind it) unprojected from 16-bit inverse depth packed into PNGs, drawn with WebGL2. Depth is compressed around the subject (`z' = pivot·(z/pivot)^γ`) so a leaf at 40 cm in front of a sky doesn't take the whole parallax budget. Where a photo triangle stretches across an edge (detected from screen-space UV derivatives) it fades out and the inpainted layer shows through.
8. **Background layer.** Depth steps larger than 6% between pixels 2 px apart mark discontinuities; their far-side depth is spread over 2.5% of the width, and foreground pixels nearer than it form the mask. LaMa (ONNX, 512 px tiles with feathered overlaps, at ~1024 px) fills it; the layer's depth is the far side, smoothed within the mask.
9. **Video.** Focal length = median of Depth Pro's estimates on 5 frames, fixed for the clip. Log-depth is blended with the previous frame warped by DIS optical flow, weight 0.6 × (1.5 − forward-backward error in px, clipped to 0..1). Cinematic `track` follows the median depth around the focus point with a lagged (EMA) focus puller.

## Development

See [CONTRIBUTING.md](https://github.com/ntoantoan/depthpro-mlx/blob/main/CONTRIBUTING.md): `uv sync`, then `uv run pytest -m "not slow"` (no weights needed) and `uv run ruff check`. CI runs on Apple Silicon GitHub runners.

## Credits

- [Depth Pro](https://arxiv.org/abs/2410.02073) by Bochkovskii et al., Apple. Weights are redistributed under Apple's [license](https://huggingface.co/ntoan199/DepthPro-mlx/blob/main/LICENSE). This project is not affiliated with Apple.
- [LaMa](https://github.com/advimman/lama) by Suvorov et al. (Apache-2.0), via the ONNX port [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX).
- Demo photos and videos: Wikimedia Commons contributors, see [assets/CREDITS.md](https://github.com/ntoantoan/depthpro-mlx/blob/main/assets/CREDITS.md).

```bibtex
@inproceedings{Bochkovskii2024:arxiv,
  author    = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun},
  title     = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
  url       = {https://arxiv.org/abs/2410.02073},
}
```

## License

| Component | License |
|---|---|
| Code in this repository (and the Docker image's own layers) | [MIT](https://github.com/ntoantoan/depthpro-mlx/blob/main/LICENSE) (the model files are a port of Apple's sample code, used under its terms) |
| Depth Pro weights (downloaded from the Hub) | Apple's [Depth Pro license](https://huggingface.co/ntoan199/DepthPro-mlx/blob/main/LICENSE), redistributed unchanged except for the format conversion |
| LaMa weights (`inpaint` extra) | Apache-2.0 |
| Demo photos and videos | CC0 / CC BY / CC BY-SA, see [assets/CREDITS.md](https://github.com/ntoantoan/depthpro-mlx/blob/main/assets/CREDITS.md) |

The Depth Pro weights are **not** under an OSI-approved license. Apple grants a personal, non-exclusive license to use, modify and redistribute them, and does not explicitly address commercial use. If you plan to ship them in a product, read Apple's terms yourself and, if in doubt, ask a lawyer. This project is not affiliated with or endorsed by Apple.
