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

# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""General-purpose residual network backbone.

:class:`ResidualNetwork` is a flexible, task-agnostic sequence-of-blocks backbone
used for regression, generation, and spatial-recall tasks.  It wires together a
configurable stack of :class:`~nvsubquadratic.modules.residual_block.ResidualBlock`
instances (or any compatible block) via the :mod:`nvsubquadratic.lazy_config`
system, enabling operator swaps (Hyena / Attention / CKConv / Mamba) purely
through config without code changes.

**Architecture**

.. code-block:: text

    input [B, *spatial, in_channels]
        ↓  dropout_in
        ↓  in_proj          → [B, *spatial, hidden_dim]
        ↓  block × N        (each block also receives the optional condition)
        ↓  out_norm
        ↓  out_proj         → [B, *spatial, out_channels]
        ↓  readout crop     (optional, for spatial-recall tasks)
    output {"logits": [B, *target_spatial, out_channels]}

**Conditioning**

An optional ``condition_in_proj`` linearly projects an external conditioning
signal (e.g. a class embedding or a global context vector) into ``hidden_dim``
before it is fed to each block's conditioning branch (e.g. FiLM / cross-attention
in :class:`~nvsubquadratic.modules.residual_block.ResidualBlock`).

**Readout crop**

When ``target_size`` is set the network extracts the bottom-right
``target_size`` spatial region of the output before returning.  This is used
for spatial-recall tasks where the model must predict a target patch embedded
inside a larger context window.  A ``target_size`` element of ``1`` collapses
(squeezes) that spatial dimension, e.g. ``(1, H, W)`` for a 2-D slice of a
3-D input.

**Gradient checkpointing**

Set ``gradient_checkpointing=True`` to recompute activations during the backward
pass instead of storing them, trading compute for memory at large scale.

Adapted from https://github.com/implicit-long-convs/ccnn_v2.
"""

from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ResidualNetwork(nn.Module):
    """General-purpose residual network backbone (see module docstring for architecture).

    All sub-modules (projections, norm, blocks) are instantiated from
    :class:`~nvsubquadratic.lazy_config.LazyConfig` objects so the architecture
    can be fully configured from YAML/JSON without subclassing.

    **Tensor layout**

    All tensors are in **channels-last** format: ``[B, *spatial, C]``.  The
    ``data_dim`` argument records the number of spatial axes (1 for sequences,
    2 for images, 3 for volumes) and is used to convert a scalar ``target_size``
    into a per-axis tuple.

    **Output format**

    :meth:`forward` always returns a ``dict`` with key ``"logits"`` whose value
    has shape ``[B, *spatial_out, out_channels]``.  When ``target_size=None``,
    ``spatial_out = spatial``; otherwise it is the cropped target region.

    Attributes:
        in_channels (int): Input channel count.
        out_channels (int): Output channel count / number of classes.
        num_blocks (int): Number of stacked residual blocks.
        hidden_dim (int): Internal feature dimension used throughout the trunk.
        data_dim (int): Number of spatial axes (1/2/3).
        gradient_checkpointing (bool): Recompute activations on backward.
        target_size (tuple | None): Per-axis readout crop size, or ``None``.
        dropout_in (nn.Module): Input dropout / augmentation applied first.
        in_proj (nn.Module): ``in_channels → hidden_dim`` linear projection.
        condition_in_proj (nn.Module | None): Optional ``hidden_dim → hidden_dim``
            projection for the conditioning signal.
        blocks (nn.ModuleList): Stack of ``num_blocks`` residual blocks.
        out_norm (nn.Module): Post-trunk normalisation (weight-decay excluded).
        out_proj (nn.Module): ``hidden_dim → out_channels`` readout projection.

    Args:
        in_channels: Number of input signal channels.
        out_channels: Number of output channels (e.g. vocabulary / class count).
        num_blocks: Depth of the residual tower.
        hidden_dim: Width of the residual tower.
        data_dim: Spatial dimensionality (1, 2, or 3).
        in_proj_cfg: LazyConfig for the input projection (typically ``nn.Linear``).
        out_proj_cfg: LazyConfig for the output projection.
        norm_cfg: LazyConfig for the output normalisation layer.
        block_cfg: LazyConfig for each residual block; instantiated ``num_blocks``
            times.
        dropout_in_cfg: LazyConfig for the input dropout layer.
        condition_in_proj_cfg: Optional LazyConfig for the condition projection.
            Pass ``None`` for unconditional networks.
        target_size: Readout crop size.  ``int`` → same size on every spatial
            axis.  ``tuple`` → per-axis sizes (use ``1`` to squeeze that axis).
            ``None`` → return the full output.
        gradient_checkpointing: Enable activation recomputation in :meth:`forward`
            to reduce peak memory at the cost of extra compute.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        hidden_dim: int,
        data_dim: int,
        in_proj_cfg: LazyConfig,
        out_proj_cfg: LazyConfig,
        norm_cfg: LazyConfig,
        block_cfg: LazyConfig,
        dropout_in_cfg: LazyConfig,
        condition_in_proj_cfg: LazyConfig | None = None,
        target_size: int | Sequence[int] | None = None,
        gradient_checkpointing: bool = False,
    ):
        """Instantiate all sub-modules from LazyConfig objects.

        Args:
            in_channels: Number of input signal channels.
            out_channels: Number of output channels.
            num_blocks: Number of residual blocks to stack.
            hidden_dim: Internal feature width.
            data_dim: Spatial dimensionality (1 / 2 / 3).
            in_proj_cfg: Config for the input projection.
            out_proj_cfg: Config for the output projection.
            norm_cfg: Config for the output norm layer.
            block_cfg: Config for each residual block (instantiated N times).
            dropout_in_cfg: Config for input dropout.
            condition_in_proj_cfg: Optional config for condition projection.
            target_size: Readout crop specification (see class docstring).
            gradient_checkpointing: Recompute activations during backward pass.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.gradient_checkpointing = gradient_checkpointing

        # Instantiate dropout_in
        self.dropout_in = instantiate(dropout_in_cfg)

        # Instantiate input projection for the network
        self.in_proj = instantiate(in_proj_cfg)

        if condition_in_proj_cfg is not None:
            # Instantiate condition input projection for the network
            self.condition_in_proj = instantiate(
                condition_in_proj_cfg, in_features=hidden_dim, out_features=hidden_dim
            )
        else:
            self.condition_in_proj = None

        # Create residual blocks
        self.blocks = nn.ModuleList([instantiate(block_cfg) for _ in range(num_blocks)])

        # Instantiate output norm
        self.out_norm = instantiate(norm_cfg)
        # Exclude self.out_norm from the parameter group with weight decay
        for param in self.out_norm.parameters():
            param._no_weight_decay = True

        # Instantiate output projection
        self.out_proj = instantiate(out_proj_cfg)

        # Target size for readout -- only used for spatial recall tasks for now.
        # Convert to tuple for consistent handling
        if target_size is None:
            self.target_size = None
        elif isinstance(target_size, int):
            self.target_size = (target_size,) * data_dim
        else:
            self.target_size = tuple(target_size)

    def _get_readout_region(self, x: torch.Tensor) -> torch.Tensor:
        """Get the readout region (bottom-right target_size region) of the input tensor.

        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, out_channels].

        Returns:
            torch.Tensor: Readout region. Shape depends on target_size:
                - For target_size=(L,): [batch_size, L, out_channels]
                - For target_size=(H, W): [batch_size, H, W, out_channels]
                - For target_size=(D, H, W): [batch_size, D, H, W, out_channels]
                - For target_size=(1, H, W) on 3D input: [batch_size, H, W, out_channels] (squeezed)
        """
        spatial_ndim = x.ndim - 2  # Exclude batch and channel dims

        if len(self.target_size) != spatial_ndim:
            raise ValueError(
                f"target_size has {len(self.target_size)} dimensions but input has {spatial_ndim} spatial dimensions. "
                f"target_size={self.target_size}, input shape={x.shape}"
            )

        # Build slice/index for each spatial dimension
        # x shape: [batch, *spatial_dims, channels]
        # Using integer index (-1) auto-removes dimension, slice(-size, None) keeps it
        slices = [slice(None)]  # batch dimension
        for size in self.target_size:
            if size == 1:
                slices.append(-1)  # integer index removes dimension
            else:
                slices.append(slice(-size, None))
        slices.append(slice(None))  # channel dimension

        return x[tuple(slices)]

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the full forward pass: project → blocks → norm → project → crop.

        Args:
            input_and_condition: Dictionary with two keys:

                * ``"input"`` — signal tensor of shape ``[B, *spatial, in_channels]``.
                * ``"condition"`` — optional conditioning tensor of shape
                  ``[B, *spatial_cond, hidden_dim]``, or ``None`` when
                  ``condition_in_proj_cfg`` was not provided.

        Returns:
            dict[str, torch.Tensor]: Single-key dict:

            * ``"logits"`` — shape ``[B, *spatial_out, out_channels]`` where
              ``spatial_out`` equals ``spatial`` unless ``target_size`` is set,
              in which case it is the cropped readout region.
        """
        # Extract the input and condition from the dictionary
        x, condition = input_and_condition["input"], input_and_condition["condition"]

        # Apply in_dropout to the input
        x = self.dropout_in(x)
        # Apply input projection
        x = self.in_proj(x)

        # Apply condition input projection if provided
        if self.condition_in_proj is not None:
            assert condition is not None, "Condition must be provided if condition input projection is provided"
            condition = self.condition_in_proj(condition)

        # Apply residual blocks (with or without condition)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, condition, use_reentrant=False)
            else:
                x = block(x, condition)

        # Apply output norm
        x = self.out_norm(x)
        # Apply output projection
        x = self.out_proj(x)

        # Get the readout region if target size is provided
        if self.target_size is not None:
            x = self._get_readout_region(x)

        return {"logits": x}
