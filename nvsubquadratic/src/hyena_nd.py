# David W. Romero, 2025-09-09

"""Hyena-style global convolutional mixer implementation for ND signals."""


import math

import torch
from einops import rearrange

from nvsubquadratic.src.utils.lazy_config import LazyConfig, instantiate


class Hyena(torch.nn.Module):
    """
    Hyena-style global convolutional mixer.
    """

    def __init__(
        self,
        global_conv_cfg: LazyConfig,
        short_conv_cfg: LazyConfig,
        gate_nonlinear_cfg: LazyConfig,
    ):
        super().__init__()
        # Core global convs: feature and gate branches
        self.global_conv = instantiate(global_conv_cfg)
        self.short_conv = instantiate(short_conv_cfg)
        assert isinstance(self.short_conv, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)), f"Short conv must be an instance of torch.nn.ConvNd (1d, 2d, or 3d). Current type: {type(self.short_conv)}"

        # Initialize the short conv
        bound = math.sqrt(1.0 / math.prod(self.short_conv.kernel_size))
        torch.nn.init.uniform_(self.short_conv.weight, -bound, bound)
        if self.short_conv.bias is not None:
            torch.nn.init.zeros_(self.short_conv.bias)

        self.gate_nonlinear = instantiate(gate_nonlinear_cfg)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Reshape query, key, and value to [B, C, * spatial_dims]
        query = rearrange(query, "b ... c -> b c ...")
        key = rearrange(key, "b ... c -> b c ...")
        value = rearrange(value, "b ... c -> b c ...")

        if not isinstance(self.short_conv, torch.nn.Identity):
            # Concatenate query, key, and value, apply the short conv projection and split again
            x = torch.cat([query, key, value], dim=1)
            x = self.short_conv(x)
            # Split query, key, and value
            query, key, value = x.split(query.shape[1], dim=1)
            # Avoid in-place ops on views returned by split
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # First gate
        # z = query * self.gate_nonlinear(key)
        query.mul_(self.gate_nonlinear(key))

        # Apply global convolution
        y = self.global_conv(query, is_bhl_input=True)

        # Second gate
        # y = self.gate_nonlinear(y) * value in-place
        value.mul_(self.gate_nonlinear(y))

        # Reshape back to [B, * spatial_dims, C]
        return rearrange(value, "b c ... -> b ... c")
