# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General-purpose ViT5-style network for dense ND prediction.

This module is the ViT5 analogue of the repository's general-purpose dense
network wrapper: it accepts channels-last dense ND inputs, patchifies them to a
token sequence, applies ViT5-style residual blocks, then unpatchifies back to a
dense channels-last output.

TODO: add ``target_size`` support for parity with
``nvsubquadratic.networks.general_purpose_resnet.ResidualNetwork``.

TODO: add ``condition_in_proj_cfg`` support so external conditioning can be
projected at the network level, matching the general-purpose ResNet wrapper.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ViT5GeneralPurposeNet(nn.Module):
    """ViT5-style general-purpose dense network with learnable register tokens.

    The model patchifies an ND input into a patch grid, flattens the patches to
    tokens, prepends auxiliary tokens, applies ViT5-style residual blocks, then
    reshapes patch tokens back to the patch grid and unpatchifies to dense
    outputs.

    TODO: add ``target_size`` support for parity with the general-purpose ResNet.

    TODO: add ``condition_in_proj_cfg`` support for parity with the
    general-purpose ResNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        num_blocks: int,
        data_dim: int,
        patch_size: int | Sequence[int],
        input_size: int | Sequence[int],
        num_registers: int,
        in_proj_cfg: LazyConfig,
        out_proj_cfg: LazyConfig,
        block_cfg: LazyConfig,
        norm_cfg: LazyConfig,
        dropout_rate: float = 0.0,
        use_cls_token: bool = False,
        prepend_registers: bool = True,
    ):
        """Initialize the ViT5 general-purpose dense network.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_dim: Transformer hidden dimension (D).
            num_blocks: Number of Transformer blocks.
            data_dim: Dimensionality of the spatial data (e.g. 1, 2, 3).
            patch_size: Spatial shape of the patch (e.g. `P` for `PxP` patches in 2D). Can be a scalar or a tuple.
            input_size: Spatial shape of the dense input (e.g. `H` for `H` width/height). Can be a scalar or a tuple.
            num_registers: Number of learnable register tokens to use.
            in_proj_cfg: LazyConfig for the linear mapping applied to patchified dense inputs to reach hidden_dim.
            out_proj_cfg: LazyConfig for the linear mapping applied to the unpatchified grid to reach out_channels.
            block_cfg: LazyConfig to instantiate each Transformer block.
            norm_cfg: LazyConfig for the output normalization layer applied before the out_proj.
            dropout_rate: Dropout rate applied right before the output projection. Let zero to disable.
            use_cls_token: If True, prepends a learnable [CLS] token to the sequence.
            prepend_registers: If True, registers will be placed at the beginning of the sequence
                (just after the CLS token if present). If False, registers are appended to the end.
                Note that prepended registers are automatically zero-padded to form a clean slice along
                spatial dimensions (vital for spatial multi-dimensional mixers like Hyena).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.data_dim = data_dim
        self.num_blocks = num_blocks
        self.num_registers = num_registers
        self.use_cls_token = use_cls_token
        self.prepend_registers = prepend_registers

        self.input_size = self._normalize_nd_value(input_size, data_dim, "input_size")
        self.patch_size = self._normalize_nd_value(patch_size, data_dim, "patch_size")
        self.patch_grid_shape = tuple(size // patch for size, patch in zip(self.input_size, self.patch_size))
        self.num_patches = math.prod(self.patch_grid_shape)

        self.in_proj = instantiate(in_proj_cfg, in_features=in_channels, out_features=hidden_dim)
        self.out_proj = instantiate(out_proj_cfg, in_features=hidden_dim, out_features=out_channels)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.cls_token._no_weight_decay = True
        else:
            self.cls_token = None

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.pos_embed._no_weight_decay = True

        if num_registers > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            self.reg_token._no_weight_decay = True
        else:
            self.reg_token = None

        if len(self.patch_grid_shape) > 1:
            self._register_plane_size = math.prod(self.patch_grid_shape[1:])
        else:
            self._register_plane_size = None

        # Calculate prefix (CLS + registers) size
        self._prefix_len = 0
        if self.use_cls_token:
            self._prefix_len += 1
        if self.num_registers > 0 and self.prepend_registers:
            self._prefix_len += self.num_registers

        # Zero-padding buffer for dense ND models: fills the prefix space to gracefully reshape
        # as a unified multi-dimensional block internally by maintaining clean slice geometries.
        if self._prefix_len > 0 and self._register_plane_size is not None:
            assert self._prefix_len <= self._register_plane_size, (
                f"prefix_len ({self._prefix_len}) > slice size ({self._register_plane_size}); "
                "prefix tokens must fit in a single slice along spatial dimensions"
            )
            pad_size = self._register_plane_size - self._prefix_len
            if pad_size > 0:
                self.register_buffer("reg_zero_pad", torch.zeros(1, pad_size, hidden_dim), persistent=False)
            else:
                self.reg_zero_pad = None
            self._padded_prefix_len = self._register_plane_size
        else:
            self.reg_zero_pad = None
            self._padded_prefix_len = self._prefix_len

        # Transformer blocks

        self.out_norm = instantiate(norm_cfg)
        for param in self.out_norm.parameters():
            param._no_weight_decay = True

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self._init_weights()

    @staticmethod
    def _normalize_nd_value(value: int | Sequence[int], data_dim: int, name: str) -> tuple[int, ...]:
        if isinstance(value, int):
            return (value,) * data_dim
        value = tuple(value)
        if len(value) != data_dim:
            raise ValueError(f"{name} must have length {data_dim}, got {len(value)}")
        return value

    def _init_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.reg_token is not None:
            nn.init.trunc_normal_(self.reg_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        start = self._padded_prefix_len
        end = start + self.num_patches
        return x[:, start:end, :]

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_and_condition: Dict with key ``input`` and optional ``condition``.

        Returns:
            Dict with dense prediction tensor in ``logits``.
        """
        # 1. Patchify density via input projecting mapper
        x = input_and_condition["input"]
        x = self.in_proj(x)

        patch_grid_shape = tuple(x.shape[1:-1])
        if patch_grid_shape != self.patch_grid_shape:
            raise ValueError(
                f"Runtime patch grid {patch_grid_shape} does not match configured shape {self.patch_grid_shape}."
            )

        # 2. Add absolute positional embeddings over the flattened sequence
        batch_size = x.shape[0]
        x = rearrange(x, "b ... c -> b (...) c")
        x = x + self.pos_embed

        # 3. Prepend CLS token (when enabled)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # 4. Insert register tokens
        if self.reg_token is not None:
            reg_tokens = self.reg_token.expand(batch_size, -1, -1)
            if self.prepend_registers:
                if self.cls_token is not None:
                    # [CLS, regs, patches]
                    x = torch.cat([x[:, :1, :], reg_tokens, x[:, 1:, :]], dim=1)
                else:
                    # [regs, patches]
                    x = torch.cat([reg_tokens, x], dim=1)
            else:
                # [patches, regs] or [CLS, patches, regs]
                x = torch.cat([x, reg_tokens], dim=1)

        # 5. Automatically uniformly align prefix tokens (CLS + prepended registers) geometrically
        # so ND sequential mixers reshape identically into N-planes.
        if self.reg_zero_pad is not None:
            pad = self.reg_zero_pad.expand(batch_size, -1, -1)
            # [prefix, zero_pad, patches]
            x = torch.cat([x[:, : self._prefix_len, :], pad, x[:, self._prefix_len :, :]], dim=1)

        # 6. Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # 7. Unpatchify dense representation filtering out CLS/Registers & optional zero padding
        x = self._extract_patch_tokens(x)
        x = self.out_norm(x)
        x = self.dropout(x)
        x = x.reshape(batch_size, *self.patch_grid_shape, self.hidden_dim)
        x = self.out_proj(x)

        return {"logits": x}
