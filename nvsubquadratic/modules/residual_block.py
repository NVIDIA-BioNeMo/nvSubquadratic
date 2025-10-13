# TODO: Add license header here

"""Residual block implementation for ND signals, composed of a sequence mixer and an MLP."""

import inspect
from typing import Optional

import torch
import torch.distributed as dist

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

    def forward(self, x: torch.Tensor, cp_group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        """Forward pass of the residual block with optional Context Parallelism.

        Args:
            x: Input tensor of shape (batch_size, *spatial_dims, num_hidden_channels)
            cp_group: Optional process group for Context Parallelism

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *spatial_dims, num_hidden_channels)
        """
        # Mixer branch
        residual = x
        x = self.input_norm(x)

        # Pass cp_group to sequence mixer if it supports it
        mixer_forward_sig = inspect.signature(self.sequence_mixer.forward)
        if "cp_group" in mixer_forward_sig.parameters:
            x = self.sequence_mixer(x, cp_group=cp_group)
        elif cp_group is not None:
            raise ValueError("cp_group is not supported by sequence mixer.")
        else:
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
