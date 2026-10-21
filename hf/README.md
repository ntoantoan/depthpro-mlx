---
license: other
license_name: apple-sample-code-license
license_link: LICENSE
library_name: mlx
pipeline_tag: depth-estimation
base_model: apple/DepthPro
tags:
  - mlx
  - depth-estimation
  - monocular-depth
  - metric-depth
  - apple-silicon
---

# Depth Pro — MLX

[Apple Depth Pro](https://github.com/apple/ml-depth-pro) (sharp monocular **metric** depth from a single image) converted to [MLX](https://github.com/ml-explore/mlx) for Apple Silicon.

Code, CLI and 3D photo pipeline: **[github.com/ntoantoan/depthpro-mlx](https://github.com/ntoantoan/depthpro-mlx)**

## Usage

```bash
pip install git+https://github.com/ntoantoan/depthpro-mlx
depthpro-mlx photo.jpg --ply     # depth.png, pointcloud.ply, 3d.mp4
```

```python
import numpy as np
from PIL import Image
from depthpro_mlx import load_model, predict

model = load_model()                      # downloads these weights
image = np.array(Image.open("photo.jpg").convert("RGB"))
out = predict(model, image)               # f_px=... if you know the focal length
out["depth"]                              # (H, W) depth in meters
out["focallength_px"]                     # estimated focal length
```

## Files

| File | Description |
|---|---|
| `depth_pro.safetensors` | All 952M parameters in float16, NHWC conv layout |

## Accuracy and speed

Parity against the official PyTorch implementation on the reference image (3024×2268):

| | Time (M4 Pro) | Peak memory | AbsRel vs. PyTorch fp32 |
|---|---|---|---|
| PyTorch MPS fp16 | 3.76 s | 7.0 GB | – |
| **MLX fp16** | **3.06 s** | 7.5 GB | 6.2e-4 |
| MLX fp32 | 4.17 s | 15.0 GB | 1.7e-6 |

## License

These weights are a format conversion of Apple's released Depth Pro weights and are redistributed under Apple's original license terms (see `LICENSE`). This conversion is not affiliated with or endorsed by Apple.

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
