# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Simple implementation of a ResNet for classification."""

import torch

from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork

class ClassificationResNet(ResidualNetwork):
    """Simple implementation of a ResNet for classification.

    It assumes:
    - the input tensor is of shape (batch_size, *spatial_dims, in_channels).
    - the output tensor is of shape (batch_size, num_classes).

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ClassificationResNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *spatial_dims, self.in_channels)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, self.out_channels)
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
