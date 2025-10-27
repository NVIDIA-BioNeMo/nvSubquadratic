# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Simple implementation of a ResNet for general purpose tasks."""

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ResidualNetwork(nn.Module):
    """Simple implementation of a Residual Network for general purpose tasks.

    It assumes:
    - the input tensor is of shape (batch_size, *spatial_dims, in_channels).
    - the output tensor is of shape (batch_size, *spatial_dims, out_channels).

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_blocks (int): Number of blocks
        hidden_dim (int): Number of hidden dimensions
        in_proj_cfg (LazyConfig): Configuration for the input projection
        out_proj_cfg (LazyConfig): Configuration for the output projection
        norm_cfg (LazyConfig): Configuration for the normalization
        block_cfg (LazyConfig): Configuration for the residual block
        dropout_in_cfg (LazyConfig): Configuration for the dropout in layer (applied to the input)
    """

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
        """Initialize the ResidualNetwork."""
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
            param._no_weight_decay = True

        # Instantiate output projection
        self.out_proj = instantiate(out_proj_cfg, in_features=hidden_dim, out_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *spatial_dims, self.in_channels)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *spatial_dims, self.out_channels)
        """
        # Apply in_dropout to the input
        x = self.dropout_in(x)
        # Apply input projection
        x = self.in_proj(x)
        # Apply residual blocks
        x = self.blocks(x)
        # Apply output norm
        x = self.out_norm(x)
        # Apply output projection
        x = self.out_proj(x)
        return x
