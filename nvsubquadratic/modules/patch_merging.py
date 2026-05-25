"""Swin-style 2x2 patch merging for hierarchical ViT-5 / Hyena networks.

Halves both spatial dims and (typically) doubles the channel dim. Two input
layouts are supported:

* ``has_register_row=False``: pure-spatial sequence ``[B, H*W, C]`` reshaped to
  a 2D grid, 2x2-merged, normalized, and projected.

* ``has_register_row=True``: ``[B, grid_w + H*W, C]`` where the first ``grid_w``
  tokens form a "register row" (the layout used by ``ViT5ClassificationNet``
  with ``prepend_registers=True`` and no CLS).  The patch grid is merged as
  above; register tokens are projected independently with their own linear so
  the FiLM conditioning signal survives the channel-dim change, then re-padded
  to the new (halved) grid width.
"""

import torch
import torch.nn as nn
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate


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
        """
        T_out = self.out_grid_h * self.out_grid_w
        # Norm is configured for dim=4*in_dim and is called on T_out tokens.
        flops = self.norm.flop_count(T_out)
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

        # Standard Swin 2x2 spatial-merge: gather four spatial cosets and concat on channels.
        x0 = patches[:, 0::2, 0::2, :]
        x1 = patches[:, 1::2, 0::2, :]
        x2 = patches[:, 0::2, 1::2, :]
        x3 = patches[:, 1::2, 1::2, :]
        merged = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]

        merged = self.norm(merged)
        merged = self.reduction(merged)
        merged_flat = rearrange(merged, "b h w c -> b (h w) c")

        if not self.has_register_row:
            return merged_flat

        regs_proj = self.reg_proj(regs)  # [B, num_regs, out_dim]
        if self.reg_zero_pad is not None:
            pad = self.reg_zero_pad.expand(B, -1, -1)
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
