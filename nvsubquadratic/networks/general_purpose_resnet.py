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
        condition_in_proj_cfg (LazyConfig | None): Configuration for the condition input projection or None if no condition is used.
            If provided, the condition tensor is of shape [B, * spatial_dims_condition, hidden_dim].
            If not provided, the condition tensor is None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        hidden_dim: int,
        data_dim: int,
        in_proj_cfg: LazyConfig,
        out_proj_cfg: LazyConfig,
        norm_cfg: LazyConfig,
        block_cfg: LazyConfig,
        dropout_in_cfg: LazyConfig,
        condition_in_proj_cfg: LazyConfig | None = None,
        target_size: int | None = None,
    ):
        """Initialize the ResidualNetwork."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim

        # Instantiate dropout_in
        self.dropout_in = instantiate(dropout_in_cfg)

        # Instantiate input projection for the network
        self.in_proj = instantiate(in_proj_cfg)

        if condition_in_proj_cfg is not None:
            # Instantiate condition input projection for the network
            self.condition_in_proj = instantiate(
                condition_in_proj_cfg, in_features=hidden_dim, out_features=hidden_dim
            )
        else:
            self.condition_in_proj = None

        # Create residual blocks
        self.blocks = nn.ModuleList([instantiate(block_cfg) for _ in range(num_blocks)])

        # Instantiate output norm
        self.out_norm = instantiate(norm_cfg)
        # Exclude self.out_norm from the parameter group with weight decay
        for param in self.out_norm.parameters():
            param._no_weight_decay = True

        # Instantiate output projection
        self.out_proj = instantiate(out_proj_cfg)

        # Target size for readout -- only used for spatial recall tasks for now.
        self.target_size = target_size

    def _get_readout_region(self, x: torch.Tensor) -> torch.Tensor:
        """Get the readout region (bottom-right target_size region) of the input tensor.

        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, out_channels].

        Returns:
            torch.Tensor: Readout region of shape [batch_size, *(target_size,)*spatial_dims, out_channels].
        """
        if x.ndim == 1 + 2:  # 1D input - [batch_size, seq_len, hidden_dim]
            return x[:, -self.target_size :, :]
        elif x.ndim == 1 + 3:  # 2D input - [batch_size, height, width, hidden_dim]
            return x[:, -self.target_size :, -self.target_size :, :]
        elif x.ndim == 1 + 4:  # 3D input - [batch_size, depth, height, width, hidden_dim]
            return x[:, -self.target_size :, -self.target_size :, -self.target_size :, :]
        else:
            raise ValueError(f"Unexpected input dimension: {x.ndim}. Expected 1D, 2D or 3D spatial dimensions.")

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the ResidualNetwork.

        Args:
            input_and_condition: A dictionary containing the input and condition.
                Keys: "input" and "condition".

            - input: Input tensor of shape [B, * spatial_dims, hidden_dim].
            - condition: Condition tensor of shape [B, * spatial_dims_condition, hidden_dim].

        Returns:
            Dict[str, torch.Tensor]:
                - "logits": tensor of shape [B, * spatial_dims, out_channels].
        """
        # Extract the input and condition from the dictionary
        x, condition = input_and_condition["input"], input_and_condition["condition"]

        # Apply in_dropout to the input
        x = self.dropout_in(x)
        # Apply input projection
        x = self.in_proj(x)

        # Apply condition input projection if provided
        if self.condition_in_proj is not None:
            assert condition is not None, "Condition must be provided if condition input projection is provided"
            condition = self.condition_in_proj(condition)

        # Apply residual blocks (with or without condition)
        for block in self.blocks:
            x = block(x, condition)

        # Apply output norm
        x = self.out_norm(x)
        # Apply output projection
        x = self.out_proj(x)

        # Get the readout region if target size is provided
        if self.target_size is not None:
            x = self._get_readout_region(x)

        return {"logits": x}
