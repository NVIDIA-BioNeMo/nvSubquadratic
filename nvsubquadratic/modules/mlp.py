# TODO: Add license header here


"""MLP implementation for ND signals."""

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class MLP(nn.Module):
    """A flexible Multi-Layer Perceptron that supports various activation functions including GLU variants.

    Always uses exactly two layers with equal input and output dimensions.

    Args:
        dim: Input and output dimension
        expansion_factor: Factor to expand the hidden dimension
        activation: Activation function to use ('relu', 'gelu', 'silu', 'glu', 'swiglu')
        dropout: Dropout rate
        bias: Whether to use bias in linear layers
        residual: Whether to use residual connections
    """

    def __init__(
        self,
        dim: int,
        activation: Literal["relu", "gelu", "silu", "glu", "swiglu"],
        dropout_cfg: LazyConfig,
        expansion_factor: float = 2.0,
        bias: bool = False,
        init_method_in: Callable[[torch.Tensor], torch.Tensor] | None = None,
        init_method_out: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """Initialize the MLP.

        Args:
            dim: Input and output dimension.
            activation: Activation function to use.
            dropout_cfg: LazyConfig for the dropout layer.
            expansion_factor: Factor to expand the hidden dimension.
            bias: Whether to use bias in linear layers.
            init_method_in: Optional initialization method for the first layer.
            init_method_out: Optional initialization method for the second layer.
        """
        assert bias is False, f"Modern MLPs do not use bias. Got {bias}"

        super().__init__()
        self.hidden_dim = int(dim * expansion_factor)
        self.activation = activation

        # Check if using GLU variants which require double width for hidden layers
        self.is_glu_variant = activation in ["glu", "swiglu"]
        glu_factor = 2 if self.is_glu_variant else 1

        # Construct first layer
        self.layer1 = nn.Linear(dim, self.hidden_dim * glu_factor, bias=bias)
        if init_method_in is not None:
            init_method_in(self.hidden_dim * glu_factor)(self.layer1.weight.data)

        # Construct dropout
        self.dropout = instantiate(dropout_cfg)

        # Construct second layer
        self.layer2 = nn.Linear(self.hidden_dim, dim, bias=bias)
        if init_method_out is not None:
            init_method_out(dim)(self.layer2.weight.data)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation function to the input."""
        if self.activation in ["relu", "gelu", "silu"]:
            return getattr(F, self.activation)(x)
        elif self.activation == "glu":
            # Use torch's built-in GLU function
            return F.glu(x, dim=-1)
        elif self.activation == "swiglu":
            # SwiGLU: SiLU(x) * y
            a, b = torch.chunk(x, 2, dim=-1)
            return b * F.silu(a)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        # Apply first layer
        x = self.layer1(x)
        # Apply activation
        x = self._apply_activation(x)
        # Apply dropout
        x = self.dropout(x)
        # Second layer
        x = self.layer2(x)
        return x
