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


"""Residual block implementation for ND signals, composed of a sequence mixer and an MLP."""

import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ResidualBlock(torch.nn.Module):
    """Residual block."""

    def __init__(
        self,
        sequence_mixer_cfg: LazyConfig,
        mlp_cfg: LazyConfig,
        norm_cfg: LazyConfig,
        dropout_cfg: LazyConfig,
    ):
        """Initialize the ResidualBlock.

        Args:
            sequence_mixer_cfg: LazyConfig for the sequence mixer layer.
            mlp_cfg: LazyConfig for the MLP layer.
            norm_cfg: LazyConfig for the input and MLP norms.
            dropout_cfg: LazyConfig for the dropout layer.
        """
        super().__init__()
        # Instantiate sequence mixer layer
        self.sequence_mixer = instantiate(sequence_mixer_cfg)

        # Instantiate MLP layer
        self.mlp = instantiate(mlp_cfg)

        # Instantiate input and MLP norms
        self.input_norm = instantiate(norm_cfg)
        # Exclude self.input_norm from the parameter group with weight decay
        for param in self.input_norm.parameters():
            param._no_wd = True
        self.mlp_norm = instantiate(norm_cfg)
        # Exclude self.mlp_norm from the parameter group with weight decay
        for param in self.mlp_norm.parameters():
            param._no_wd = True

        # Instantiate dropout
        self.dropout = instantiate(dropout_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *spatial_dims, num_hidden_channels)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *spatial_dims, num_hidden_channels)
        """
        # Mixer branch
        residual = x
        x = self.input_norm(x)
        x = self.sequence_mixer(x)
        x = self.dropout(x)
        x = x + residual

        # MLP branch
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        return x
