# David W. Romero, 2025-09-09

"""QKV-based sequence mixer implementation for ND signals."""

import torch
from typing import Callable

from nvsubquadratic.src.utils.lazy_config import LazyConfig, instantiate

class QKVSequenceMixer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mixer_cfg: LazyConfig,
        init_method_in: Callable[[torch.Tensor], torch.Tensor] = None,
        init_method_out: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
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
        # Q, K, V projections via single linear
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Sequence mixer (e.g., self-attention, hyena, etc.)
        x = self.mixer(q, k, v)
        # Output projection
        x = self.out_proj(x)
        return x
