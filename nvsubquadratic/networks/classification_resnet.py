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


"""Simple implementation of a ResNet for classification."""

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ClassificationResNet(nn.Module):
    """Simple implementation of a ResNet for classification."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        hidden_dim: int,
        in_proj_cfg: LazyConfig,
        out_proj_cfg: LazyConfig,
        norm_cfg: LazyConfig,
        block_cfg: LazyConfig,
        dropout_in_cfg: LazyConfig,
    ):
        """Initialize the ClassificationResNet.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of blocks
            hidden_dim: Number of hidden dimensions
            in_proj_cfg: Configuration for the input projection
            out_proj_cfg: Configuration for the output projection
            norm_cfg: Configuration for the normalization
            block_cfg: Configuration for the residual block
            dropout_in_cfg: Configuration for the dropout in layer (applied to the input)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim

        # Instantiate dropout_in
        self.dropout_in = instantiate(dropout_in_cfg)

        # Instantiate input projection for the network
        self.in_proj = instantiate(in_proj_cfg, in_features=in_channels, out_features=hidden_dim)

        # Create residual blocks
        blocks = []
        for i in range(num_blocks):
            print(f"Block {i}/{num_blocks}")

            blocks.append(instantiate(block_cfg))
        self.blocks = nn.Sequential(*blocks)

        # Instantiate output norm
        self.out_norm = instantiate(norm_cfg)
        # Exclude self.out_norm from the parameter group with weight decay
        for param in self.out_norm.parameters():
            param._no_wd = True

        # Instantiate output projection
        self.out_proj = instantiate(out_proj_cfg, in_features=hidden_dim, out_features=out_channels)

    def forward(self, x):
        """Forward pass of the ClassificationResNet.

        Args:
            x: Input tensor of shape (batch_size, *spatial_dims, num_channels)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Apply in_dropout to the input
        x = self.dropout_in(x)
        # Apply input projection
        x = self.in_proj(x)
        # Apply residual blocks
        x = self.blocks(x)
        # Average over the spatial dimensions
        x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
        x = x.mean(dim=1)
        # Apply output norm
        x = self.out_norm(x)
        # Apply output projection
        x = self.out_proj(x)
        return x
