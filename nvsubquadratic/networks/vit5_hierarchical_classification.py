"""Hierarchical ViT-5 classification network (Swin-style stages + patch merging).

Differs from ``ViT5ClassificationNet`` in three ways:

1. **Hierarchical stages.**  Blocks are organized into ``num_stages`` groups
   with a ``PatchMerging`` between consecutive stages.  Each stage halves the
   spatial resolution and (typically) doubles the channel dim, mirroring
   Swin-T's ``[2, 2, 6, 2]`` / ``[C, 2C, 4C, 8C]`` structure.

2. **GAP readout only.**  No CLS token.  Two layouts are supported:
     * ``pure``: tokens are a flat patch grid ``[B, H*W, C]``.
     * ``register_row``: the first ``grid_w`` tokens form a register row used
       by the Hyena mixer for FiLM conditioning; GAP excludes them.

3. **Per-stage block configs.**  The caller supplies one block config per
   stage (replicated across that stage's blocks) plus the patch-merging
   config for the transition into it.
"""

from typing import List, Literal

import torch
import torch.nn as nn
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ViT5HierarchicalClassificationNet(nn.Module):
    """Swin-style hierarchical ViT-5 classifier with optional register-row layout.

    Args:
        in_channels: Input image channels (3 for RGB).
        num_classes: Number of output classes.
        image_size: Input image size (assumes square).
        initial_patch_size: Stride of the initial Conv2d patch embed.  The
            stage-0 grid is ``(image_size // initial_patch_size)`` per side.
        stage_dims: Channel dim per stage, ``len == num_stages``.
            ``stage_dims[0]`` is the patch-embed output dim.
        stage_depths: Number of blocks per stage, ``len == num_stages``.
        stage_block_cfgs: Per-stage block LazyConfig.  Each is replicated
            ``stage_depths[i]`` times.  The block is expected to be a
            ``ViT5ResidualBlock`` whose sequence mixer matches the stage's
            grid size and hidden dim.
        patch_merge_cfgs: PatchMerging LazyConfig for each stage transition.
            Length ``num_stages - 1``.  ``patch_merge_cfgs[i]`` runs between
            stage ``i`` and stage ``i+1``.
        norm_cfg: Final norm before the head (configured for ``dim =
            stage_dims[-1]``).
        layout: ``"pure"`` (no registers) or ``"register_row"`` (registers
            prepended as the first row of the 2D grid at every stage).
        num_registers: Number of register tokens (only used for
            ``layout="register_row"``).  Constant across stages; padding is
            recomputed per stage to match the halved grid width.
        reg_init: Register init scheme — ``"trunc_normal"`` (default) or
            ``"zeros"``.
        dropout_rate: Dropout before the head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        image_size: int,
        initial_patch_size: int,
        stage_dims: List[int],
        stage_depths: List[int],
        stage_block_cfgs: List[LazyConfig],
        patch_merge_cfgs: List[LazyConfig],
        norm_cfg: LazyConfig,
        layout: Literal["pure", "register_row"] = "pure",
        num_registers: int = 0,
        reg_init: Literal["trunc_normal", "zeros"] = "trunc_normal",
        dropout_rate: float = 0.0,
    ):
        """Initialise the hierarchical ViT-5 network and validate stage configs."""
        super().__init__()
        num_stages = len(stage_dims)
        if len(stage_depths) != num_stages:
            raise ValueError(f"stage_depths len ({len(stage_depths)}) != num_stages ({num_stages})")
        if len(stage_block_cfgs) != num_stages:
            raise ValueError(f"stage_block_cfgs len ({len(stage_block_cfgs)}) != num_stages ({num_stages})")
        if len(patch_merge_cfgs) != num_stages - 1:
            raise ValueError(
                f"patch_merge_cfgs len ({len(patch_merge_cfgs)}) must equal num_stages - 1 ({num_stages - 1})"
            )
        if layout not in ("pure", "register_row"):
            raise ValueError(f"layout must be 'pure' or 'register_row', got {layout!r}")
        if layout == "register_row" and num_registers <= 0:
            raise ValueError("layout='register_row' requires num_registers > 0")

        self.num_stages = num_stages
        self.stage_dims = list(stage_dims)
        self.stage_depths = list(stage_depths)
        self.layout = layout
        self.num_registers = num_registers
        self.image_size = image_size
        self.initial_patch_size = initial_patch_size
        self._reg_init = reg_init
        self.num_classes = num_classes

        # Per-stage grid widths (used by GAP to skip the register row).
        # Stage i's grid is (initial_grid // 2**i) per side.
        initial_grid = image_size // initial_patch_size
        if initial_grid * initial_patch_size != image_size:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by initial_patch_size ({initial_patch_size})"
            )
        self.stage_grid_sides = [initial_grid // (2**i) for i in range(num_stages)]
        if self.stage_grid_sides[-1] < 1:
            raise ValueError(
                f"Final stage grid collapses to {self.stage_grid_sides[-1]} — too many stages for grid {initial_grid}"
            )

        if layout == "register_row":
            final_grid_w = self.stage_grid_sides[-1]
            if num_registers > final_grid_w:
                raise ValueError(
                    f"num_registers ({num_registers}) must fit in the final stage grid_w ({final_grid_w})"
                )

        # Patch embedding: matches the v5_patch convention (no bias; APE absorbs offset).
        self.patch_embed = nn.Conv2d(
            in_channels,
            stage_dims[0],
            kernel_size=initial_patch_size,
            stride=initial_patch_size,
            padding=0,
            bias=False,
        )

        num_patches_0 = initial_grid * initial_grid
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches_0, stage_dims[0]))
        self.pos_embed._no_weight_decay = True

        if layout == "register_row":
            self.reg_token = nn.Parameter(torch.zeros(1, num_registers, stage_dims[0]))
            self.reg_token._no_weight_decay = True
            pad_size = initial_grid - num_registers
            if pad_size > 0:
                self.register_buffer("reg_zero_pad", torch.zeros(1, pad_size, stage_dims[0]), persistent=False)
            else:
                self.reg_zero_pad = None
        else:
            self.reg_token = None
            self.reg_zero_pad = None

        # Build stages: ModuleList of ModuleList, one block-list per stage.
        self.stage_blocks = nn.ModuleList(
            [
                nn.ModuleList([instantiate(cfg) for _ in range(stage_depths[i])])
                for i, cfg in enumerate(stage_block_cfgs)
            ]
        )
        # Patch merges between stages.
        self.patch_merges = nn.ModuleList([instantiate(cfg) for cfg in patch_merge_cfgs])

        self.out_norm = instantiate(norm_cfg)
        for p in self.out_norm.parameters():
            p._no_weight_decay = True
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.out_proj = nn.Linear(stage_dims[-1], num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.reg_token is not None:
            if self._reg_init == "zeros":
                nn.init.zeros_(self.reg_token)
            else:
                nn.init.trunc_normal_(self.reg_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)

    def flop_count(self, inference: bool = False) -> int:
        """FLOPs for one forward pass (one sample).

        Breakdown:
          1. Conv2d patch embed: 2 * in_ch * stage_dims[0] * P^2 * num_patches_0.
          2. APE add: num_patches_0 * stage_dims[0].
          3. For each stage:
             - Each block: block.flop_count(T_stage, inference) where T_stage
               counts the register row when present.
          4. For each transition: patch_merge.flop_count() (already a per-sample count).
          5. Final norm on 1 GAP'd token: out_norm.flop_count(1).
          6. GAP itself: num_patches_last * stage_dims[-1] adds.
          7. Head: 2 * stage_dims[-1] * num_classes.
        """
        D0 = self.stage_dims[0]
        P = self.initial_patch_size
        num_patches_0 = self.stage_grid_sides[0] ** 2
        in_ch = self.patch_embed.in_channels

        flops = 0
        flops += 2 * in_ch * D0 * P * P * num_patches_0
        flops += num_patches_0 * D0

        for i in range(self.num_stages):
            grid = self.stage_grid_sides[i]
            T = grid * grid + (grid if self.layout == "register_row" else 0)
            for block in self.stage_blocks[i]:
                flops += block.flop_count(T, inference=inference)
            if i < self.num_stages - 1:
                flops += self.patch_merges[i].flop_count()

        last_grid = self.stage_grid_sides[-1]
        last_patches = last_grid * last_grid
        flops += last_patches * self.stage_dims[-1]  # GAP add
        flops += self.out_norm.flop_count(1)
        flops += 2 * self.stage_dims[-1] * self.num_classes
        return flops

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_and_condition: Dict with key ``"input"`` of shape
                ``[B, H, W, C]`` (channels-last) and optional ``"condition"``
                (unused).

        Returns:
            ``{"logits": [B, num_classes]}``.
        """
        x = input_and_condition["input"]  # [B, H, W, C]
        x = rearrange(x, "b h w c -> b c h w")
        x = self.patch_embed(x)  # [B, D0, grid, grid]
        x = rearrange(x, "b c h w -> b (h w) c")  # [B, num_patches_0, D0]
        x = x + self.pos_embed

        B = x.shape[0]
        grid_w = self.stage_grid_sides[0]

        if self.layout == "register_row":
            regs = self.reg_token.expand(B, -1, -1)
            if self.reg_zero_pad is not None:
                pad = self.reg_zero_pad.expand(B, -1, -1)
                x = torch.cat([regs, pad, x], dim=1)
            else:
                x = torch.cat([regs, x], dim=1)

        # Stages with patch merging between.
        for i, block_list in enumerate(self.stage_blocks):
            for block in block_list:
                x = block(x)
            if i < self.num_stages - 1:
                x = self.patch_merges[i](x)
                grid_w = self.stage_grid_sides[i + 1]

        # GAP over patch tokens (skip register row if present).
        if self.layout == "register_row":
            out = x[:, grid_w:].mean(dim=1)
        else:
            out = x.mean(dim=1)

        out = self.out_norm(out)
        out = self.dropout(out)
        logits = self.out_proj(out)
        return {"logits": logits}
