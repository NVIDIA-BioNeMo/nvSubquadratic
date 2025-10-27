# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# David W. Romero, 2025-09-09

"""Residual block implementation for ND signals, composed of a sequence mixer and an MLP."""

from __future__ import annotations

from typing import Optional

import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ResidualBlock(torch.nn.Module):
    """Residual block for ND signals, composed of a sequence mixer and an MLP."""

    def __init__(
        self,
        sequence_mixer_cfg: LazyConfig,
        sequence_mixer_norm_cfg: LazyConfig,
        condition_mixer_cfg: LazyConfig,
        condition_mixer_norm_cfg: LazyConfig,
        mlp_cfg: LazyConfig,
        mlp_norm_cfg: LazyConfig,
        dropout_cfg: LazyConfig,
    ):
        """Initialize the ResidualBlock.

        Args:
            sequence_mixer_cfg: LazyConfig for the sequence mixer layer.
            sequence_mixer_norm_cfg: LazyConfig for the sequence mixer norm.
            condition_mixer_cfg: LazyConfig for the condition mixer layer.
            condition_mixer_norm_cfg: LazyConfig for the condition mixer norm.
            mlp_cfg: LazyConfig for the MLP layer.
            mlp_norm_cfg: LazyConfig for the MLP norm.
            dropout_cfg: LazyConfig for the dropout layer.
        """
        if sequence_mixer_cfg.__target__ == torch.nn.Identity:
            assert sequence_mixer_norm_cfg.__target__ == torch.nn.Identity, (
                "Sequence mixer norm must be Identity if sequence mixer is Identity"
            )
        if mlp_cfg.__target__ == torch.nn.Identity:
            assert mlp_norm_cfg.__target__ == torch.nn.Identity, "MLP norm must be Identity if MLP is Identity"
        if condition_mixer_cfg.__target__ == torch.nn.Identity:
            assert condition_mixer_norm_cfg.__target__ == torch.nn.Identity, (
                "Condition mixer norm must be Identity if condition mixer is Identity"
            )

        super().__init__()
        # Instantiate sequence mixer layer
        self.sequence_mixer = instantiate(sequence_mixer_cfg)
        # Instantiate input norm
        self.input_norm = instantiate(sequence_mixer_norm_cfg)
        # Exclude self.input_norm from the parameter group with weight decay
        for param in self.input_norm.parameters():
            param._no_weight_decay = True

        # Instantiate cross attention layer
        self.condition_mixer = instantiate(condition_mixer_cfg)
        # Instantiate cross attention norm
        self.condition_mixer_norm = instantiate(condition_mixer_norm_cfg)
        # Exclude self.condition_mixer_norm from the parameter group with weight decay
        for param in self.condition_mixer_norm.parameters():
            param._no_weight_decay = True

        # Instantiate MLP layer
        self.mlp = instantiate(mlp_cfg)
        # Instantiate MLP norm
        self.mlp_norm = instantiate(mlp_norm_cfg)
        # Exclude self.mlp_norm from the parameter group with weight decay
        for param in self.mlp_norm.parameters():
            param._no_weight_decay = True

        # Instantiate dropout
        self.dropout = instantiate(dropout_cfg)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *spatial_dims, num_hidden_channels)
            condition (torch.Tensor): Condition tensor of shape (batch_size, *spatial_dims_condition, num_hidden_channels)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *spatial_dims, num_hidden_channels)
        """
        # Mixer branch
        if not isinstance(self.sequence_mixer, torch.nn.Identity):
            residual = x
            x = self.input_norm(x)
            x = self.sequence_mixer(x)
            x = self.dropout(x)
            x = x + residual

        # Cross attention branch
        if not isinstance(self.condition_mixer, torch.nn.Identity):
            assert condition is not None, "Condition must be provided if condition mixer is not Identity."
            residual = x
            x = self.condition_mixer_norm(x)
            x = self.condition_mixer(x, condition)
            x = self.dropout(x)
            x = x + residual

        # MLP branch
        if not isinstance(self.mlp, torch.nn.Identity):
            residual = x
            x = self.mlp_norm(x)
            x = self.mlp(x)
            x = self.dropout(x)
            x = x + residual
        return x


