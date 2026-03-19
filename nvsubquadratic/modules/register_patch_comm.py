"""Register-patch communication modules for distributed register strategies.

When registers are evenly distributed in the token sequence and stripped before
Hyena's 2D spatial convolution, they need an explicit pathway to read from patch
tokens. Two configurable options:

- RegisterCrossAttention: lightweight multi-head cross-attention where registers
  (queries) attend to patches (keys/values). Provides global information flow.
- RegisterLocalPooling: each register averages features from its neighboring
  patches. Provides local information flow with no learned parameters (except
  an optional projection).

Reference: Mamba-R (arXiv:2405.14858) — evenly distributed registers.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegisterCrossAttention(nn.Module):
    """Lightweight cross-attention: registers attend to patches.

    Q = Linear(registers), K = Linear(patches), V = Linear(patches).
    Uses scaled dot-product attention with optional QK L2-normalization.

    Args:
        hidden_dim: Token embedding dimension.
        num_heads: Number of attention heads. hidden_dim must be divisible by num_heads.
        qk_norm: If True, apply L2 normalization to Q and K (matches Hyena convention).
        proj_bias: If True, use bias in Q/K/V/out projections.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        qk_norm: bool = True,
        proj_bias: bool = False,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qk_norm = qk_norm

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=proj_bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=proj_bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=proj_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=proj_bias)

        self._init_weights()

    def _init_weights(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.trunc_normal_(proj.weight, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(self, registers: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """Cross-attend registers to patches.

        Args:
            registers: [B, R, C] register tokens.
            patches: [B, P, C] patch tokens.

        Returns:
            Updated registers: [B, R, C].
        """
        B, R, C = registers.shape
        P = patches.shape[1]
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(registers).reshape(B, R, H, D).transpose(1, 2)  # [B, H, R, D]
        k = self.k_proj(patches).reshape(B, P, H, D).transpose(1, 2)    # [B, H, P, D]
        v = self.v_proj(patches).reshape(B, P, H, D).transpose(1, 2)    # [B, H, P, D]

        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, R, P]
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # [B, H, R, D]
        out = out.transpose(1, 2).reshape(B, R, C)  # [B, R, C]
        return self.out_proj(out)

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, qk_norm={self.qk_norm}"


class RegisterLocalPooling(nn.Module):
    """Local average pooling: each register reads from its neighboring patches.

    With ``num_registers`` registers evenly distributed among ``num_patches``
    patches, register *i* averages patches ``[i*stride : (i+1)*stride]`` where
    ``stride = num_patches // num_registers``.

    Args:
        hidden_dim: Token embedding dimension.
        num_registers: Number of register tokens.
        num_patches: Number of patch tokens.
        use_proj: If True, apply a linear projection after pooling.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_registers: int,
        num_patches: int,
        use_proj: bool = False,
    ):
        super().__init__()
        assert num_patches % num_registers == 0, (
            f"num_patches ({num_patches}) must be divisible by num_registers ({num_registers})"
        )
        self.num_registers = num_registers
        self.num_patches = num_patches
        self.stride = num_patches // num_registers

        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False) if use_proj else nn.Identity()
        if use_proj:
            nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, registers: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """Pool neighboring patches into each register.

        Args:
            registers: [B, R, C] register tokens (unused, present for API consistency).
            patches: [B, P, C] patch tokens.

        Returns:
            Pooled register updates: [B, R, C].
        """
        B, P, C = patches.shape
        # Group patches by register neighborhood
        grouped = patches.reshape(B, self.num_registers, self.stride, C)  # [B, R, stride, C]
        pooled = grouped.mean(dim=2)  # [B, R, C]
        return self.proj(pooled)

    def extra_repr(self) -> str:
        return f"num_registers={self.num_registers}, num_patches={self.num_patches}, stride={self.stride}"
