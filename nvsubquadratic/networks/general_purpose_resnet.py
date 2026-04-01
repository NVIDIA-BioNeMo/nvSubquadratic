# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Simple implementation of a ResNet for general purpose tasks."""

from typing import Sequence

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
        target_size (int | tuple | None): Size of the readout region. Can be:
            - int: Same size for all spatial dimensions (e.g., 16 for 16x16 in 2D)
            - tuple: Different size per dimension. For 3D spatial recall where the target
              is a 2D image on the last depth slice, use (1, H, W) to extract only the
              last depth slice with HxW spatial region.
            - None: No readout extraction (return full output)
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
        target_size: int | Sequence[int] | None = None,
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
        # Convert to tuple for consistent handling
        if target_size is None:
            self.target_size = None
        elif isinstance(target_size, int):
            self.target_size = (target_size,) * data_dim
        else:
            self.target_size = tuple(target_size)

    def _get_readout_region(self, x: torch.Tensor) -> torch.Tensor:
        """Get the readout region (bottom-right target_size region) of the input tensor.

        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, out_channels].

        Returns:
            torch.Tensor: Readout region. Shape depends on target_size:
                - For target_size=(L,): [batch_size, L, out_channels]
                - For target_size=(H, W): [batch_size, H, W, out_channels]
                - For target_size=(D, H, W): [batch_size, D, H, W, out_channels]
                - For target_size=(1, H, W) on 3D input: [batch_size, H, W, out_channels] (squeezed)
        """
        spatial_ndim = x.ndim - 2  # Exclude batch and channel dims

        if len(self.target_size) != spatial_ndim:
            raise ValueError(
                f"target_size has {len(self.target_size)} dimensions but input has {spatial_ndim} spatial dimensions. "
                f"target_size={self.target_size}, input shape={x.shape}"
            )

        # Build slice/index for each spatial dimension
        # x shape: [batch, *spatial_dims, channels]
        # Using integer index (-1) auto-removes dimension, slice(-size, None) keeps it
        slices = [slice(None)]  # batch dimension
        for size in self.target_size:
            if size == 1:
                slices.append(-1)  # integer index removes dimension
            else:
                slices.append(slice(-size, None))
        slices.append(slice(None))  # channel dimension

        return x[tuple(slices)]

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
