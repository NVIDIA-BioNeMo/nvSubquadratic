# TODO: Add license header here


"""Mamba mixer layer for ND signals."""

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class Mamba(torch.nn.Module):
    """Mamba mixer layer for ND signals.

    ND signals are handled by reshaping the input to [B, flatten (* spatial_dims), hidden_dim] and then passing it to the core layer.

    If bidirectionality is enabled, an additional Mamba layer is instantiated and used to process the reversed input.
    The output of this layer is reversed and added to the output of the core (forward) layer.

    The output is then reshaped back to [B, * spatial_dims, hidden_dim].
    """

    def __init__(
        self,
        mamba_layer_cfg: LazyConfig,
        bidirectional: bool = False,
    ):
        """Initialize the Mamba mixer layer.

        Args:
            mamba_layer_cfg: LazyConfig - LazyConfig for the Mamba layer.
            bidirectional: bool - Whether to use a bidirectional Mamba layer.
        """
        super().__init__()
        self.bidirectional = bidirectional

        self.core_layer = instantiate(mamba_layer_cfg)
        # If bidirectional, we need to instantiate a reversed Mamba layer
        if self.bidirectional:
            self.core_layer_rev = instantiate(mamba_layer_cfg)

    def forward(self, x):
        """Forward pass of the Mamba mixer layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, * spatial_dims, hidden_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, * spatial_dims, hidden_dim)
        """
        x_shape = x.shape
        # Reshape input to [B, flatten (* spatial_dims), hidden_dim
        x = rearrange(x, "b ... c -> b (...) c")

        # Forward pass through the core layer. It expects an input of shape [B, seq_len, hidden_dim].
        out = self.core_layer(x)

        # If bidirectional, reverse the input, apply the inverted layer, reverse back and add to
        # output of the core (forward) layer
        if self.bidirectional:
            out_rev = self.core_layer_rev(torch.flip(x, dims=[1]))
            out = out + torch.flip(out_rev, dims=[1])

        # Reshape output to original [B, * spatial_dims, hidden_dim] shape
        out = out.reshape(*x_shape)
        return out
