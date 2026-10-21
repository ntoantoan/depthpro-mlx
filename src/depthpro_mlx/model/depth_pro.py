"""Depth Pro in MLX.

A faithful port of apple/ml-depth-pro. Module and attribute names mirror the
PyTorch reference so checkpoint keys map one-to-one; only tensor layouts differ
(MLX convolutions are NHWC).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .layers import ViT, resize_bilinear, tokens_to_grid

ENCODER_DIMS = (256, 512, 1024, 1024)
HOOK_BLOCK_IDS = (5, 11)
DECODER_FEATURES = 256


def _apply(layers: list, x: mx.array) -> mx.array:
    for layer in layers:
        x = layer(x)
    return x


def _project_upsample(dim_in: int, dim_out: int, upsample_layers: int, dim_int: int | None = None) -> list:
    dim_int = dim_out if dim_int is None else dim_int
    blocks = [nn.Conv2d(dim_in, dim_int, kernel_size=1, bias=False)]
    blocks += [
        nn.ConvTranspose2d(dim_int if i == 0 else dim_out, dim_out, kernel_size=2, stride=2, bias=False)
        for i in range(upsample_layers)
    ]
    return blocks


class DepthProEncoder(nn.Module):
    """Multi-resolution encoder: image pyramid -> overlapping 384px patches -> shared ViT."""

    def __init__(self, dims_encoder=ENCODER_DIMS, decoder_features: int = DECODER_FEATURES):
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = ViT()
        self.image_encoder = ViT()
        self.out_size = self.patch_encoder.grid_size  # 24
        embed = self.patch_encoder.embed_dim

        self.upsample_latent0 = _project_upsample(embed, decoder_features, 3, dim_int=self.dims_encoder[0])
        self.upsample_latent1 = _project_upsample(embed, self.dims_encoder[0], 2)
        self.upsample0 = _project_upsample(embed, self.dims_encoder[1], 1)
        self.upsample1 = _project_upsample(embed, self.dims_encoder[2], 1)
        self.upsample2 = _project_upsample(embed, self.dims_encoder[3], 1)
        self.upsample_lowres = nn.ConvTranspose2d(embed, self.dims_encoder[3], kernel_size=2, stride=2, bias=True)
        self.fuse_lowres = nn.Conv2d(2 * self.dims_encoder[3], self.dims_encoder[3], kernel_size=1, bias=True)

    @property
    def img_size(self) -> int:
        return self.patch_encoder.img_size * 4  # 1536

    @staticmethod
    def split(x: mx.array, overlap_ratio: float) -> mx.array:
        patch_size = 384
        stride = int(patch_size * (1 - overlap_ratio))
        steps = int(math.ceil((x.shape[2] - patch_size) / stride)) + 1
        patches = [
            x[:, j * stride : j * stride + patch_size, i * stride : i * stride + patch_size, :]
            for j in range(steps)
            for i in range(steps)
        ]
        return mx.concatenate(patches, axis=0)

    @staticmethod
    def merge(x: mx.array, batch_size: int, padding: int) -> mx.array:
        steps = int(math.sqrt(x.shape[0] // batch_size))
        rows = []
        idx = 0
        for j in range(steps):
            row = []
            for i in range(steps):
                out = x[batch_size * idx : batch_size * (idx + 1)]
                h0 = padding if j != 0 else 0
                h1 = out.shape[1] - padding if j != steps - 1 else out.shape[1]
                w0 = padding if i != 0 else 0
                w1 = out.shape[2] - padding if i != steps - 1 else out.shape[2]
                row.append(out[:, h0:h1, w0:w1, :])
                idx += 1
            rows.append(mx.concatenate(row, axis=2))
        return mx.concatenate(rows, axis=1)

    def __call__(self, x: mx.array) -> list[mx.array]:
        b, h, w, _ = x.shape
        x0 = x
        x1 = resize_bilinear(x, h // 2, w // 2)
        x2 = resize_bilinear(x, h // 4, w // 4)

        x0_patches = self.split(x0, overlap_ratio=0.25)  # 5x5 @ 384
        x1_patches = self.split(x1, overlap_ratio=0.5)  # 3x3 @ 384
        x2_patches = x2  # 1x1 @ 384
        n0, n1 = x0_patches.shape[0], x1_patches.shape[0]

        pyramid = mx.concatenate([x0_patches, x1_patches, x2_patches], axis=0)
        tokens, (hook0, hook1) = self.patch_encoder(pyramid, return_blocks=HOOK_BLOCK_IDS)
        encodings = tokens_to_grid(tokens, self.out_size)

        latent0 = self.merge(tokens_to_grid(hook0, self.out_size)[: b * 25], b, padding=3)
        latent1 = self.merge(tokens_to_grid(hook1, self.out_size)[: b * 25], b, padding=3)

        x0_features = self.merge(encodings[:n0], b, padding=3)  # 96x96
        x1_features = self.merge(encodings[n0 : n0 + n1], b, padding=6)  # 48x48
        x2_features = encodings[n0 + n1 :]  # 24x24

        global_tokens, _ = self.image_encoder(x2_patches)
        global_features = tokens_to_grid(global_tokens, self.out_size)

        latent0 = _apply(self.upsample_latent0, latent0)
        latent1 = _apply(self.upsample_latent1, latent1)
        x0_features = _apply(self.upsample0, x0_features)
        x1_features = _apply(self.upsample1, x1_features)
        x2_features = _apply(self.upsample2, x2_features)

        global_features = self.upsample_lowres(global_features)
        global_features = self.fuse_lowres(mx.concatenate([x2_features, global_features], axis=-1))

        return [latent0, latent1, x0_features, x1_features, global_features]


class ResidualBlock(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.residual = [
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        return x + _apply(self.residual, x)


class FeatureFusionBlock2d(nn.Module):
    def __init__(self, num_features: int, deconv: bool):
        super().__init__()
        self.resnet1 = ResidualBlock(num_features)
        self.resnet2 = ResidualBlock(num_features)
        self.use_deconv = deconv
        if deconv:
            self.deconv = nn.ConvTranspose2d(num_features, num_features, kernel_size=2, stride=2, bias=False)
        self.out_conv = nn.Conv2d(num_features, num_features, kernel_size=1, bias=True)

    def __call__(self, x0: mx.array, x1: mx.array | None = None) -> mx.array:
        x = x0
        if x1 is not None:
            x = x + self.resnet1(x1)
        x = self.resnet2(x)
        if self.use_deconv:
            x = self.deconv(x)
        return self.out_conv(x)


class MultiresConvDecoder(nn.Module):
    def __init__(self, dims_encoder: list[int], dim_decoder: int):
        super().__init__()
        self.dim_decoder = dim_decoder
        convs = [
            nn.Identity()
            if dims_encoder[0] == dim_decoder
            else nn.Conv2d(dims_encoder[0], dim_decoder, kernel_size=1, bias=False)
        ]
        convs += [nn.Conv2d(d, dim_decoder, kernel_size=3, padding=1, bias=False) for d in dims_encoder[1:]]
        self.convs = convs
        self.fusions = [FeatureFusionBlock2d(dim_decoder, deconv=(i != 0)) for i in range(len(dims_encoder))]

    def __call__(self, encodings: list[mx.array]) -> tuple[mx.array, mx.array]:
        features = self.convs[-1](encodings[-1])
        lowres_features = features
        features = self.fusions[-1](features)
        for i in range(len(encodings) - 2, -1, -1):
            features = self.fusions[i](features, self.convs[i](encodings[i]))
        return features, lowres_features


class FOVNetwork(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = [ViT(), nn.Linear(1024, num_features // 2)]
        self.downsample = [nn.Conv2d(num_features, num_features // 2, kernel_size=3, stride=2, padding=1), nn.ReLU()]
        self.head = [
            nn.Conv2d(num_features // 2, num_features // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features // 4, num_features // 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features // 8, 1, kernel_size=6, stride=1, padding=0),
        ]

    def __call__(self, x: mx.array, lowres_feature: mx.array) -> mx.array:
        _, h, w, _ = x.shape
        x = resize_bilinear(x, h // 4, w // 4)
        vit, proj = self.encoder
        tokens, _ = vit(x)
        tokens = proj(tokens)
        lowres_feature = _apply(self.downsample, lowres_feature)
        x = tokens_to_grid(tokens, vit.grid_size) + lowres_feature
        return _apply(self.head, x)


class DepthPro(nn.Module):
    def __init__(self, use_fov_head: bool = True):
        super().__init__()
        self.encoder = DepthProEncoder()
        self.decoder = MultiresConvDecoder([DECODER_FEATURES] + list(ENCODER_DIMS), DECODER_FEATURES)
        d = DECODER_FEATURES
        self.head = [
            nn.Conv2d(d, d // 2, kernel_size=3, padding=1),
            nn.ConvTranspose2d(d // 2, d // 2, kernel_size=2, stride=2, bias=True),
            nn.Conv2d(d // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(),
        ]
        if use_fov_head:
            self.fov = FOVNetwork(d)

    @property
    def img_size(self) -> int:
        return self.encoder.img_size

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array | None]:
        """x: (B, 1536, 1536, 3) normalized to [-1, 1].
        Returns canonical inverse depth (B, 1536, 1536, 1) and FOV in degrees (B,)."""
        encodings = self.encoder(x)
        features, features_0 = self.decoder(encodings)
        canonical_inverse_depth = _apply(self.head, features)
        fov_deg = None
        if hasattr(self, "fov"):
            fov_deg = self.fov(x, mx.stop_gradient(features_0)).reshape(-1)
        return canonical_inverse_depth, fov_deg

    def infer(self, image: mx.array, f_px: float | None = None) -> dict:
        """image: (H, W, 3) or (1, H, W, 3), already normalized to [-1, 1].

        Returns metric depth in meters at the input resolution and the focal
        length in pixels (estimated when ``f_px`` is not given)."""
        if image.ndim == 3:
            image = image[None]
        _, h, w, _ = image.shape
        x = resize_bilinear(image, self.img_size, self.img_size)

        canonical_inverse_depth, fov_deg = self(x)
        if f_px is None:
            f_px = 0.5 * w / mx.tan(0.5 * mx.radians(fov_deg.astype(mx.float32)))
        f_px = mx.array(f_px, dtype=mx.float32).reshape(-1)

        inverse_depth = canonical_inverse_depth.astype(mx.float32) * (w / f_px)[:, None, None, None]
        inverse_depth = resize_bilinear(inverse_depth, h, w)
        depth = 1.0 / mx.clip(inverse_depth, 1e-4, 1e4)
        return {"depth": depth[0, ..., 0], "focallength_px": f_px[0]}
