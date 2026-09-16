# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Swin-style 2x2 patch merging for hierarchical ViT-5 / Hyena networks.

Halves both spatial dims and (typically) doubles the channel dim. Two input
layouts are supported:

* ``has_register_row=False``: pure-spatial sequence ``[B, H*W, C]`` reshaped to
  a 2D grid, 2x2-merged, normalized, and projected.

* ``has_register_row=True``: ``[B, grid_w + H*W, C]`` where the first ``grid_w``
  tokens form a "register row" used by ``ViT5HierarchicalClassificationNet``
  with ``layout="register_row"``.  The patch grid is merged as
  above; register tokens are projected independently with their own linear so
  the FiLM conditioning signal survives the channel-dim change, then re-padded
  to the new (halved) grid width.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate


def _merge_spatial_cosets(x: torch.Tensor) -> torch.Tensor:
    """Concatenate even-grid 2x2 neighbours in TL, BL, TR, BR order."""
    return torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)


class PatchMerging(nn.Module):
    """2x2 patch merging with optional register-row passthrough.

    Args:
        in_dim: Input channel dimension.
        out_dim: Output channel dimension (Swin-T uses ``out_dim = 2 * in_dim``).
        grid_h: Patch-grid height before merging.  Must be even.
        grid_w: Patch-grid width before merging.  Must be even.
        norm_cfg: LazyConfig for the post-concat norm.  Must be configured with
            ``dim = 4 * in_dim`` since it operates on concatenated 2x2 features.
        num_registers: Number of register tokens at the start of the register
            row.  Only used when ``has_register_row=True``.
        has_register_row: When True, the first ``grid_w`` tokens of the input
            sequence are treated as a register row (regs + zero pad) and
            passed through a dedicated ``Linear(in_dim, out_dim)`` projection.
            The output register row is repacked to width ``grid_w // 2``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_h: int,
        grid_w: int,
        norm_cfg: LazyConfig,
        num_registers: int = 0,
        has_register_row: bool = False,
    ):
        """Initialise PatchMerging weights and validate grid dimensions."""
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or grid_h <= 0 or grid_w <= 0:
            raise ValueError("Channel and grid dimensions must be positive.")
        if num_registers < 0 or (num_registers and not has_register_row):
            raise ValueError("num_registers must be nonnegative and requires has_register_row=True.")
        if grid_h % 2 != 0 or grid_w % 2 != 0:
            raise ValueError(f"grid_h={grid_h}, grid_w={grid_w} must both be even for 2x2 merging")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.out_grid_h = grid_h // 2
        self.out_grid_w = grid_w // 2
        self.has_register_row = has_register_row
        self.num_registers = num_registers

        self.norm = instantiate(norm_cfg)
        for p in self.norm.parameters():
            p._no_weight_decay = True
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.reduction.weight, std=0.02)

        if has_register_row:
            if num_registers > self.out_grid_w:
                raise ValueError(f"num_registers ({num_registers}) must fit in halved grid_w ({self.out_grid_w})")
            self.reg_proj = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.trunc_normal_(self.reg_proj.weight, std=0.02)
            pad_size = self.out_grid_w - num_registers
            if pad_size > 0:
                self.register_buffer("reg_zero_pad", torch.zeros(1, pad_size, out_dim), persistent=False)
            else:
                self.reg_zero_pad = None
        else:
            self.reg_proj = None
            self.reg_zero_pad = None

    def flop_count(self) -> int:
        """FLOPs for one merging step (one sample).

        Breakdown:
          * norm on (out_grid_h * out_grid_w) tokens at 4*in_dim channels.
          * reduction linear: 2 * (out_grid_h * out_grid_w) * (4*in_dim) * out_dim.
          * (register row only) reg_proj: 2 * num_registers * in_dim * out_dim.

        Returns:
            int: Approximate arithmetic FLOPs, excluding memory movement.
        """
        T_out = self.out_grid_h * self.out_grid_w
        # Norm is configured for dim=4*in_dim and is called on T_out tokens.
        if hasattr(self.norm, "flop_count"):
            flops = self.norm.flop_count(T_out)
        elif isinstance(self.norm, nn.LayerNorm):
            flops = 2 * T_out * 4 * self.in_dim
        elif isinstance(self.norm, nn.Identity):
            flops = 0
        else:
            raise NotImplementedError("The configured norm must expose flop_count().")
        flops += 2 * T_out * 4 * self.in_dim * self.out_dim
        if self.has_register_row:
            flops += 2 * self.num_registers * self.in_dim * self.out_dim
        return flops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``[B, T_in, in_dim]`` token sequence.  ``T_in = grid_h*grid_w``
               in the pure-spatial case; ``grid_w + grid_h*grid_w`` when a
               register row is present.

        Returns:
            ``[B, T_out, out_dim]`` where ``T_out`` is ``out_grid_h*out_grid_w``
            (pure-spatial) or ``out_grid_w + out_grid_h*out_grid_w`` (with
            register row).
        """
        B = x.shape[0]

        if self.has_register_row:
            regs_row = x[:, : self.grid_w, :]  # [B, grid_w, C] full row incl. pad
            regs = regs_row[:, : self.num_registers, :]  # [B, num_regs, C]
            patches_flat = x[:, self.grid_w :, :]  # [B, H*W, C]
        else:
            patches_flat = x

        patches = rearrange(patches_flat, "b (h w) c -> b h w c", h=self.grid_h, w=self.grid_w)

        merged = _merge_spatial_cosets(patches)

        merged = self.norm(merged)
        merged = self.reduction(merged)
        merged_flat = rearrange(merged, "b h w c -> b (h w) c")

        if not self.has_register_row:
            return merged_flat

        regs_proj = self.reg_proj(regs)  # [B, num_regs, out_dim]
        if self.reg_zero_pad is not None:
            pad = self.reg_zero_pad.to(regs_proj).expand(B, -1, -1)
            out = torch.cat([regs_proj, pad, merged_flat], dim=1)
        else:
            out = torch.cat([regs_proj, merged_flat], dim=1)
        return out

    def extra_repr(self) -> str:
        """Return a compact string summary of the module configuration."""
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"grid={self.grid_h}x{self.grid_w}->{self.out_grid_h}x{self.out_grid_w}, "
            f"register_row={self.has_register_row}, num_registers={self.num_registers}"
        )


class PatchEmbedHierarchical(nn.Module):
    """4×4 strided convolution stem for hierarchical architectures.

    Converts a channels-last image ``(B, H_img, W_img, C_in)`` to a
    coarse feature map ``(B, H_img//4, W_img//4, embed_dim)`` in channels-last
    format, followed by a LayerNorm (matching VMamba's ``patch_norm=True``).

    Args:
        in_channels: Number of input channels (3 for RGB).
        embed_dim: Output channel dimension.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 96):
        """Initialize the stride-four convolution and LayerNorm."""
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4, bias=True)
        self.norm = nn.LayerNorm(embed_dim)
        for param in self.norm.parameters():
            param._no_weight_decay = True
        self.proj.bias._no_weight_decay = True
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, H, W, C)`` channels-last image tensor.

        Returns:
            ``(B, H//4, W//4, embed_dim)`` channels-last feature map.
        """
        x = rearrange(x, "b h w c -> b c h w")
        x = self.proj(x)  # (B, embed_dim, H/4, W/4)
        x = rearrange(x, "b c h w -> b h w c")  # channels-last
        x = self.norm(x)
        return x

    def flop_count(self, H: int, W: int) -> int:
        """FLOPs for stem convolution + LayerNorm.

        Conv2d (kernel 4×4, stride 4): 2 * in_ch * embed_dim * 4 * 4 * (H/4 * W/4).
        LayerNorm: 2 * (H/4 * W/4) * embed_dim.

        Args:
            H: Input image height.
            W: Input image width.

        Returns:
            int: Approximate per-sample FLOPs under the two-ops-per-feature norm convention.
        """
        C_in = self.proj.in_channels
        C_out = self.proj.out_channels
        H_out, W_out = H // 4, W // 4
        conv_flops = 2 * C_in * C_out * 4 * 4 * H_out * W_out
        norm_flops = 2 * H_out * W_out * C_out
        return conv_flops + norm_flops


class PatchMerging2D(nn.Module):
    """Discrete 2×2 spatial downsampler following Swin Transformer / VMamba.

    Takes a channels-last spatial feature map ``(B, H, W, C)`` and produces
    ``(B, ceil(H/2), ceil(W/2), 2*C)``.

    Operation (exact Swin/VMamba recipe):
        1. Pad H/W to even if necessary.
        2. Slice into 4 sub-grids on a 2×2 stride.
        3. Concatenate along the channel axis → 4C channels.
        4. Apply LayerNorm(4C).
        5. Linear(4C → 2C, bias=False).

    Args:
        dim: Input channel dimension C.
    """

    def __init__(self, dim: int):
        """Initialize LayerNorm and the channel-doubling projection."""
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(4 * dim)
        for param in self.norm.parameters():
            param._no_weight_decay = True
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        nn.init.trunc_normal_(self.reduction.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, H, W, C)`` channels-last tensor.

        Returns:
            ``(B, ceil(H/2), ceil(W/2), 2*C)`` channels-last tensor.
        """
        H, W = x.shape[1], x.shape[2]
        if H % 2 != 0 or W % 2 != 0:
            # Pad spatial dims to even size: (last_dim, W_pad, H_pad)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x = _merge_spatial_cosets(x)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2C)
        return x

    def flop_count(self, H: int, W: int) -> int:
        """FLOPs for one PatchMerging2D call on an (H, W) input.

        - Concatenation: memory movement, excluded from arithmetic FLOPs.
        - LayerNorm(4C) on H/2 * W/2 tokens: 2 * (H/2*W/2) * 4C.
        - Linear(4C → 2C): 2 * (H/2*W/2) * 4C * 2C.

        Args:
            H: Input feature-grid height, rounded up when odd.
            W: Input feature-grid width, rounded up when odd.

        Returns:
            int: Approximate per-sample FLOPs under the two-ops-per-feature norm convention.
        """
        H_out, W_out = (H + 1) // 2, (W + 1) // 2
        tokens = H_out * W_out
        norm_flops = 2 * tokens * 4 * self.dim
        linear_flops = 2 * tokens * 4 * self.dim * 2 * self.dim
        return norm_flops + linear_flops
