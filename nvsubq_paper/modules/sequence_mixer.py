# TODO: Add license header here


"""QKV-based sequence mixer implementation for ND signals."""

from typing import Callable

import torch

from nvsubq_paper.lazy_config import LazyConfig, instantiate


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
        """
        super().__init__()

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
        # Q, K, V projections via single linear
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Sequence mixer (e.g., self-attention, hyena, etc.)
        x = self.mixer(q, k, v, cp_group, **mixer_kwargs)
        # Output projection
        x = self.out_proj(x)
        return x
