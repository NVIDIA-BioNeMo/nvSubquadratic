# David W. Romero, 2025-09-09

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
