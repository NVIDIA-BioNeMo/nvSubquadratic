# David W. Romero, 2025-09-09

"""Simple implementation of a ResNet for classification."""

import torch
import torch.nn as nn

from nvsubquadratic.src.utils.lazy_config import LazyConfig, instantiate


class ClassificationResNet(nn.Module):
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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim

        # Instantiate dropout_in
        self.dropout_in = instantiate(dropout_in_cfg)

        # Instantiate input projection for the network
        self.in_proj = instantiate(in_proj_cfg, in_channels=in_channels, out_channels=hidden_dim)

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
        self.out_proj = instantiate(out_proj_cfg, in_channels=hidden_dim, out_channels=out_channels)

    def forward(self, x):
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
