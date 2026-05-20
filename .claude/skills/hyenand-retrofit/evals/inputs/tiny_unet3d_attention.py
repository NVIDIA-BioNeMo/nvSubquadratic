"""Tiny 3D U-Net with bottleneck self-attention over spatial positions.

Used as a small test target for attention -> HyenaND retrofits on
feature-map hosts. The bottleneck attention operates directly on
[B, C, D, H, W] channel-first feature maps (no token sequence in the
host API). Internally it uses ``F.scaled_dot_product_attention`` rather
than ``nn.MultiheadAttention``, so there is no MHA slot to swap — the
retrofit must replace the whole ``SpatialAttention3D`` module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two 3x3x3 convolutions with GroupNorm and SiLU."""

    def __init__(self, in_ch: int, out_ch: int):
        """Stack conv layers from ``in_ch`` to ``out_ch`` channels."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conv block on ``[B, C, D, H, W]``."""
        return self.net(x)


class SpatialAttention3D(nn.Module):
    """Self-attention over spatial positions of a 3D feature map.

    Input/output: [B, C, D, H, W]. Retrofit target: replace this whole
    module with a Hyena mixer that operates on channel-last 3D feature
    maps. No CLS / prefix tokens.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        """Build QKV projection and output proj for ``dim`` channels."""
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply full-spatial self-attention; preserves ``[B, C, D, H, W]``."""
        B, C, D, H, W = x.shape
        N = D * H * W
        h = x.flatten(2).transpose(1, 2)  # [B, N, C]
        h = self.norm(h)
        qkv = self.qkv(h).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, heads, N, head_dim]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out.transpose(1, 2).view(B, C, D, H, W)


class TinyUNet3D(nn.Module):
    """Tiny 3D U-Net for 32x32x32 volumes, 2-class segmentation."""

    def __init__(self, in_channels: int = 1, base_ch: int = 24, out_channels: int = 2):
        """Wire encoder, bottleneck attention, decoder, and segmentation head."""
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base_ch)
        self.pool = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base_ch, base_ch * 2)
        self.bottleneck = SpatialAttention3D(base_ch * 2, num_heads=4)
        self.up = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_ch * 2, base_ch)
        self.head = nn.Conv3d(base_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, attend at bottleneck, decode; return per-voxel logits."""
        e1 = self.enc1(x)  # [B, base_ch, 32, 32, 32]
        e2 = self.enc2(self.pool(e1))  # [B, 2*base_ch, 16, 16, 16]
        b = e2 + self.bottleneck(e2)  # residual attention
        u = self.up(b)  # [B, base_ch, 32, 32, 32]
        d = self.dec1(torch.cat([u, e1], dim=1))
        return self.head(d)


if __name__ == "__main__":
    model = TinyUNet3D()
    x = torch.randn(2, 1, 32, 32, 32)
    y = model(x)
    print(y.shape)  # [2, 2, 32, 32, 32]
