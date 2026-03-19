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

        self.blocks = nn.ModuleList([instantiate(block_cfg) for _ in range(num_blocks)])

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
        start = 1 if self.use_cls_token else 0

        if self.prepend_registers and self.num_registers > 0:
            start += self.num_registers
            end = start + self.num_patches
            return x[:, start:end, :]

        end = x.shape[1] - self.num_registers if self.num_registers > 0 else x.shape[1]
        return x[:, start:end, :]

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_and_condition: Dict with key ``input`` and optional ``condition``.

        Returns:
            Dict with dense prediction tensor in ``logits``.
        """
        x = input_and_condition["input"]
        x = self.in_proj(x)

        patch_grid_shape = tuple(x.shape[1:-1])
        if patch_grid_shape != self.patch_grid_shape:
            raise ValueError(
                f"Runtime patch grid {patch_grid_shape} does not match configured shape {self.patch_grid_shape}."
            )

        batch_size = x.shape[0]
        x = rearrange(x, "b ... c -> b (...) c")
        x = x + self.pos_embed

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        if self.reg_token is not None:
            reg_tokens = self.reg_token.expand(batch_size, -1, -1)
            if self.prepend_registers:
                if self.cls_token is not None:
                    x = torch.cat([x[:, :1, :], reg_tokens, x[:, 1:, :]], dim=1)
                else:
                    x = torch.cat([reg_tokens, x], dim=1)
            else:
                x = torch.cat([x, reg_tokens], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self._extract_patch_tokens(x)
        x = self.out_norm(x)
        x = self.dropout(x)
        x = x.reshape(batch_size, *self.patch_grid_shape, self.hidden_dim)
        x = self.out_proj(x)

        return {"logits": x}
