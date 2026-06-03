"""Tiny ViT with standard multi-head self-attention.

Hand-written transformer for 64x64 RGB images, 4 transformer blocks,
hidden_dim=128, num_heads=4. Patches are 8x8 (so 8x8 = 64 patches).
Used as a small test target for attention -> HyenaND retrofits.
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Conv-based patch embedding that flattens 2D patches into a token sequence."""

    def __init__(self, image_size: int = 64, patch_size: int = 8, in_channels: int = 3, hidden_dim: int = 128):
        """Configure the patch-projection conv for the given image / patch size."""
        super().__init__()
        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image ``[B, C, H, W]`` to patch tokens ``[B, N, C]``."""
        x = self.proj(x)  # [B, C, H, W]
        return x.flatten(2).transpose(1, 2)  # [B, N, C]


class MLP(nn.Module):
    """Two-layer GELU MLP used as the transformer feed-forward sublayer."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        """Build a ``dim -> dim * mlp_ratio -> dim`` MLP with GELU activation."""
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``fc1 -> GELU -> fc2`` to ``[B, N, C]``."""
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head self-attention and an MLP."""

    def __init__(self, dim: int = 128, num_heads: int = 4, mlp_ratio: float = 4.0):
        """Configure attention and MLP sublayers for hidden size ``dim``."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and MLP residual sublayers to ``[B, N, C]``."""
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """4-block ViT for 64x64 images, 10-class classification."""

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        num_classes: int = 10,
    ):
        """Build patch embedding, CLS token, ``num_blocks`` transformer blocks, and classifier head."""
        super().__init__()
        self.embed = PatchEmbed(image_size, patch_size, in_channels=3, hidden_dim=hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed.num_patches + 1, hidden_dim))
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for input images ``[B, 3, H, W]``."""
        x = self.embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])
        return self.head(x)


if __name__ == "__main__":
    model = TinyViT()
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    print(y.shape)  # [2, 10]
