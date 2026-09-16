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


"""ViT-5 Hierarchical Classification Network.

4-stage pyramid architecture following Swin Transformer / VMamba:

  Stem (4×4 conv)  →  [Stage 1 blocks]  →  PatchMerging
    → [Stage 2 blocks]  →  PatchMerging
    → [Stage 3 blocks]  →  PatchMerging
    → [Stage 4 blocks]  →  LayerNorm  →  GAP  →  Linear head

Key differences from ``ViT5ClassificationNet``:
  - No CLS token, register tokens, absolute positional embeddings, or padding.
  - Channels-last ``(B, H, W, C)`` throughout: ``ViT5ResidualBlock`` and its
    sub-modules (LayerNorm, MLP, GRN) all operate on the last dimension and
    are therefore layout-agnostic.
  - Per-stage hidden dimension: each stage's blocks must be configured with the
    correct ``hidden_dim`` for their stage.
  - Global Average Pooling (GAP) readout — matching Swin / VMamba.
  - Only Hyena blocks supported in the initial implementation (no attention).

``ViT5ResidualBlock`` reuse notes:
  - ``sequence_mixer_cfg`` should point to a ``QKVSequenceMixer(Hyena(...))``
    that accepts ``(B, H, W, C)`` directly.  CKConvND reads
    ``spatial_dims = x.shape[1:-1]`` at runtime, so it adapts to any H×W.
  - ``register_start_idx`` and ``register_pooling_cfg`` are left at their
    defaults (0 / None) — no registers in the hierarchical setup.
  - GRN (``grn_cfg``) is optional but recommended for Hyena blocks.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.patch_merging import PatchEmbedHierarchical, PatchMerging2D


def _compute_drop_path_rates(
    max_rate: float,
    stage_depths: list[int],
    schedule: str = "linear",
) -> list[list[float]]:
    """Compute per-block drop-path rates with a linear ramp across all stages.

    Args:
        max_rate: Maximum stochastic depth rate (applied to the last block).
        stage_depths: Number of blocks per stage, e.g. [2, 2, 6, 2].
        schedule: ``"linear"`` (ramp 0→max across all blocks) or ``"constant"``.

    Returns:
        Nested list ``rates[stage][block]`` of floats.
    """
    total = sum(stage_depths)
    if schedule == "constant":
        flat = [max_rate] * total
    elif total <= 1:
        flat = [0.0] * total
    else:
        flat = [max_rate * i / (total - 1) for i in range(total)]

    rates = []
    idx = 0
    for depth in stage_depths:
        rates.append(flat[idx : idx + depth])
        idx += depth
    return rates


@dataclass
class StageSpec:
    """Specification for one hierarchical stage.

    Args:
        num_blocks: Number of residual blocks in this stage.
        hidden_dim: Channel width for all blocks in this stage.
        block_cfg: Shared ``LazyConfig`` for ``ViT5ResidualBlock``.
            The same config is instantiated ``num_blocks`` times with
            per-block ``drop_path_rate`` injected.
    """

    num_blocks: int
    hidden_dim: int
    block_cfg: LazyConfig


class ViT5HierarchicalNet(nn.Module):
    """Hierarchical Hyena network with Swin-style Patch Merging.

    Args:
        in_channels: Input image channels (3 for RGB).
        num_classes: Number of output classes.
        stage_specs: List of ``StageSpec`` for the 4 stages.  Must have
            exactly 4 entries, ordered from finest to coarsest resolution.
        max_drop_path_rate: Maximum stochastic depth drop probability.
            Per-block rates are linearly ramped from 0 to this value
            across the total block count.
        drop_path_schedule: ``"linear"`` or ``"constant"``.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        stage_specs: list[StageSpec],
        max_drop_path_rate: float = 0.0,
        drop_path_schedule: str = "linear",
    ):
        """Initialize the stem, stage blocks, spatial mergers and classifier."""
        super().__init__()
        if len(stage_specs) != 4:
            raise ValueError(f"Expected 4 stages, got {len(stage_specs)}")

        if any(s.num_blocks < 0 or s.hidden_dim <= 0 for s in stage_specs):
            raise ValueError("Stage depths must be nonnegative and dimensions positive.")
        if any(right.hidden_dim != 2 * left.hidden_dim for left, right in itertools.pairwise(stage_specs)):
            raise ValueError("Stage widths must double at each PatchMerging2D transition.")
        if drop_path_schedule not in ("linear", "constant"):
            raise ValueError("drop_path_schedule must be linear or constant.")
        stage_depths = [s.num_blocks for s in stage_specs]
        drop_path_rates = _compute_drop_path_rates(max_drop_path_rate, stage_depths, drop_path_schedule)

        # Stem: maps (B, H, W, 3) → (B, H/4, W/4, C1)
        self.stem = PatchEmbedHierarchical(
            in_channels=in_channels,
            embed_dim=stage_specs[0].hidden_dim,
        )

        # Stage blocks and downsampling layers
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i, spec in enumerate(stage_specs):
            blocks = nn.ModuleList(
                [instantiate(spec.block_cfg, drop_path_rate=drop_path_rates[i][j]) for j in range(spec.num_blocks)]
            )
            self.stages.append(blocks)

            if i < len(stage_specs) - 1:
                self.downsamplers.append(PatchMerging2D(dim=spec.hidden_dim))

        # Final norm + head (matches Swin/VMamba: LN → GAP → Linear)
        final_dim = stage_specs[-1].hidden_dim
        self.out_norm = nn.LayerNorm(final_dim)
        # Follow ViT5ClassificationNet's explicit no-decay normalization policy.
        for param in self.out_norm.parameters():
            param._no_weight_decay = True
        self.head = nn.Linear(final_dim, num_classes, bias=True)
        self.head.bias._no_weight_decay = True

        self._init_head()

    @property
    def out_proj(self) -> nn.Linear:
        """Return the classifier for the wrapper; checkpoint keys remain ``head.*``.

        When replacing the classifier for transfer learning, filter
        ``network.head`` with ``DropKeysFromCheckpoint``, not ``network.out_proj``.
        """
        return self.head

    def _init_head(self) -> None:
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_and_condition: Dict with key ``"input"`` containing an image
                tensor of shape ``(B, H, W, C)`` in channels-last format.

        Returns:
            Dict with key ``"logits"`` of shape ``(B, num_classes)``.
        """
        x = input_and_condition["input"]  # (B, H_img, W_img, C_in) channels-last

        x = self.stem(x)  # (B, H/4, W/4, C1)

        for i, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x)  # (B, H_s, W_s, C_s)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)  # (B, H_s/2, W_s/2, 2*C_s)

        x = self.out_norm(x)  # (B, H4, W4, C4)
        x = x.mean(dim=(1, 2))  # GAP: (B, C4)
        logits = self.head(x)  # (B, num_classes)

        return {"logits": logits}

    def flop_count(self, image_size: int = 224) -> int:
        """Approximate FLOPs for one forward pass (single sample).

        Counts stem + per-stage blocks + downsampling layers + head.
        Block FLOPs use the ``flop_count`` API of ``ViT5ResidualBlock``.

        Args:
            image_size: Assumed square input resolution (default 224).

        Returns:
            Total FLOPs as an integer.
        """
        flops = 0

        # Stem
        flops += self.stem.flop_count(image_size, image_size)

        H = image_size // 4  # 56 for 224
        for i, stage_blocks in enumerate(self.stages):
            W = H
            num_tokens = H * W
            for block in stage_blocks:
                flops += block.flop_count(num_tokens)
            if i < len(self.downsamplers):
                flops += self.downsamplers[i].flop_count(H, W)
                H = (H + 1) // 2

        # Final norm (LayerNorm on H4*W4 tokens)
        num_tokens_final = H * H
        final_dim = self.out_norm.normalized_shape[0]
        flops += 2 * num_tokens_final * final_dim

        # GAP: free (reduction, no multiply-adds counted)

        # Head linear
        flops += 2 * final_dim * self.head.out_features

        return flops
