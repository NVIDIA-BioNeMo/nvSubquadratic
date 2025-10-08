# TODO: Add license header here


"""QKV-based sequence mixer implementation for ND signals."""

from typing import Callable

import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class QKVSequenceMixer(torch.nn.Module):
    """QKV sequence mixer."""

    def __init__(
        self,
        hidden_dim: int,
        mixer_cfg: LazyConfig,
        init_method_in: Callable[[torch.Tensor], torch.Tensor] | None = None,
        init_method_out: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """Initialize the QKV sequence mixer.

        Args:
            hidden_dim: Hidden dimension.
            mixer_cfg: LazyConfig for the sequence mixer layer.
            init_method_in: Optional initialization method for the QKV projection.
            init_method_out: Optional initialization method for the output projection.
        """
        super().__init__()

        # Instantiate sequence mixer layer (expects a module taking q, k, v)
        self.mixer = instantiate(mixer_cfg)

        # Combined QKV projection (no bias)
        self.qkv_proj = torch.nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        # Output projection
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Initialize projections
        if init_method_in is not None:
            init_method_in(hidden_dim)(self.qkv_proj.weight.data)
        if init_method_out is not None:
            init_method_out(hidden_dim)(self.out_proj.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the QKV sequence mixer."""
        # Q, K, V projections via single linear
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Sequence mixer (e.g., self-attention, hyena, etc.)
        x = self.mixer(q, k, v)
        # Output projection
        x = self.out_proj(x)
        return x
