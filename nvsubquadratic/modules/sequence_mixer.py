# TODO: Add license header here


"""QKV-based sequence mixer implementation for ND signals."""

from typing import Callable

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class QKVSequenceMixer(torch.nn.Module):
    """QKV sequence mixer with configurable projections and initialization.

    Wraps an inner mixer (e.g. Hyena) with linear QKV input and output
    projections, mirroring the structure of ``ViT5Attention``.
    """

    def __init__(
        self,
        hidden_dim: int,
        mixer_cfg: LazyConfig,
        qkv_bias: bool = False,
        out_proj_bias: bool = False,
        init_method_in: Callable[[int], Callable[[torch.Tensor], torch.Tensor]] | None = None,
        init_method_out: Callable[[int], Callable[[torch.Tensor], torch.Tensor]] | None = None,
        channels_first: bool = False,
    ):
        """Initialize the QKV sequence mixer.

        Args:
            hidden_dim: Hidden dimension.
            mixer_cfg: LazyConfig for the inner sequence mixer layer.
            qkv_bias: Whether the combined QKV projection has a bias term.
            out_proj_bias: Whether the output projection has a bias term.
            init_method_in: Optional curried initializer ``fn(dim) -> fn(tensor)``
                for the QKV projection weights (and zero-init for bias if present).
            init_method_out: Optional curried initializer ``fn(dim) -> fn(tensor)``
                for the output projection weights (and zero-init for bias if present).
            channels_first: When True, rearrange to BCHW before splitting QKV
                and pass channels-first tensors to the inner mixer (which must
                accept ``channels_first_io=True``).  Eliminates 2 rearranges
                per block compared to the default path.
        """
        super().__init__()

        self.channels_first = channels_first
        self.mixer = instantiate(mixer_cfg)

        self.qkv_proj = torch.nn.Linear(hidden_dim, 3 * hidden_dim, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=out_proj_bias)

        if init_method_in is not None:
            init_method_in(hidden_dim)(self.qkv_proj.weight.data)
            if qkv_bias:
                torch.nn.init.zeros_(self.qkv_proj.bias)
        if init_method_out is not None:
            init_method_out(hidden_dim)(self.out_proj.weight.data)
            if out_proj_bias:
                torch.nn.init.zeros_(self.out_proj.bias)

    def flop_count(self, spatial_dims: tuple[int, ...], inference: bool = False) -> int:
        """Count FLOPs for QKV projections + inner mixer + output projection.

        Let D = ``self.qkv_proj.in_features`` (hidden_dim),
            T = prod(spatial_dims) (number of spatial positions).

        FLOPs breakdown:
          1. QKV projection (Linear(D, 3D)):  2 * T * D * 3D = 6 * T * D²
             Three projections (query, key, value) packed into one linear.
          2. Inner mixer (e.g. Hyena):
             Delegated to ``self.mixer.flop_count(spatial_dims, inference)``.
          3. Output projection (Linear(D, D)):  2 * T * D * D = 2 * T * D²

        Total: 8 * T * D² + inner_mixer_flops.

        Args:
            spatial_dims: Spatial dimensions of the signal, e.g. (H, W).
                Linear projections operate on T = prod(spatial_dims) tokens.
            inference: Passed through to the inner mixer.

        Returns:
            Total FLOPs as an integer.
        """
        D = self.qkv_proj.in_features
        T = 1
        for s in spatial_dims:
            T *= s

        flops = 0
        # QKV projection
        flops += 2 * T * D * self.qkv_proj.out_features
        # Inner mixer
        flops += self.mixer.flop_count(spatial_dims, inference=inference)
        # Output projection
        flops += 2 * T * self.out_proj.in_features * self.out_proj.out_features
        return flops

    def forward(
        self, x: torch.Tensor, cp_group: torch.distributed.ProcessGroup = None, **mixer_kwargs
    ) -> torch.Tensor:
        """Forward pass of the QKV sequence mixer.

        Args:
            x: torch.Tensor - The input tensor of shape [batch_size, *spatial_dims, hidden_dim].
            cp_group: torch.distributed.ProcessGroup - Context parallel process group.
            **mixer_kwargs: Forwarded to the inner mixer (e.g. ``conditioning`` for FiLM).

        Returns:
            torch.Tensor - The output tensor of shape [batch_size, *spatial_dims, hidden_dim].
        """
        # Q, K, V projections via single linear (channels-last)
        qkv = self.qkv_proj(x)

        if self.channels_first:
            # Rearrange once to BCHW, then split on channel dim.
            # The mixer receives BCHW tensors and returns BCHW.
            qkv = rearrange(qkv, "b ... c -> b c ...")
            q, k, v = torch.chunk(qkv, 3, dim=1)
            x = self.mixer(q, k, v, cp_group, **mixer_kwargs)
            x = rearrange(x, "b c ... -> b ... c")
        else:
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            x = self.mixer(q, k, v, cp_group, **mixer_kwargs)

        # Output projection
        x = self.out_proj(x)
        return x
