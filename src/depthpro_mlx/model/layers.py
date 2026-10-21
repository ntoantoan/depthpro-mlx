"""Shared building blocks. All tensors are NHWC (MLX convention)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _source_index(out_size: int, in_size: int) -> tuple[mx.array, mx.array, mx.array]:
    """Sampling positions matching torch bilinear interpolation with align_corners=False."""
    scale = in_size / out_size
    src = (mx.arange(out_size, dtype=mx.float32) + 0.5) * scale - 0.5
    src = mx.maximum(src, 0.0)
    i0 = mx.floor(src).astype(mx.int32)
    i0 = mx.minimum(i0, in_size - 1)
    i1 = mx.minimum(i0 + 1, in_size - 1)
    w1 = src - i0.astype(mx.float32)
    return i0, i1, w1


def resize_bilinear(x: mx.array, height: int, width: int) -> mx.array:
    """Bilinear resize of an NHWC tensor, numerically equivalent to
    ``torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)``."""
    _, h, w, _ = x.shape
    if (h, w) == (height, width):
        return x
    dtype = x.dtype

    y0, y1, wy = _source_index(height, h)
    x0, x1, wx = _source_index(width, w)
    wy = wy.astype(dtype)[None, :, None, None]
    wx = wx.astype(dtype)[None, None, :, None]

    top = mx.take(x, y0, axis=1)
    bottom = mx.take(x, y1, axis=1)
    rows = top + (bottom - top) * wy

    left = mx.take(rows, x0, axis=2)
    right = mx.take(rows, x1, axis=2)
    return left + (right - left) * wx


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.transpose(2, 0, 3, 1, 4)
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(x.transpose(0, 2, 1, 3).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block with LayerScale (DINOv2 / timm layout)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)  # (B, H/p, W/p, C)
        b, h, w, c = x.shape
        return x.reshape(b, h * w, c)


class ViT(nn.Module):
    """DINOv2 ViT-L/16 at 384px, as used by every Depth Pro backbone."""

    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, num_patches + 1, embed_dim))
        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim)
        self.blocks = [Block(embed_dim, num_heads) for _ in range(depth)]
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    @property
    def grid_size(self) -> int:
        return self.img_size // self.patch_size

    def __call__(self, x: mx.array, return_blocks: tuple[int, ...] = ()) -> tuple[mx.array, list[mx.array]]:
        """Return the normalized token sequence (with class token) and the raw
        outputs of the requested intermediate blocks."""
        x = self.patch_embed(x)
        cls = mx.broadcast_to(self.cls_token, (x.shape[0], 1, self.embed_dim)).astype(x.dtype)
        x = mx.concatenate([cls, x], axis=1) + self.pos_embed.astype(x.dtype)

        intermediates = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in return_blocks:
                intermediates.append(x)
        return self.norm(x), intermediates


def tokens_to_grid(tokens: mx.array, size: int, cls_token_offset: int = 1) -> mx.array:
    """Drop the class token and reshape (B, N, C) tokens to an NHWC grid."""
    if cls_token_offset:
        tokens = tokens[:, cls_token_offset:, :]
    b, _, c = tokens.shape
    return tokens.reshape(b, size, size, c)
