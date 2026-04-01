# TODO: Add license header here


"""MLP implementation for ND signals.

Supports two backends:

* ``"torch"`` — pure PyTorch (``nn.Linear`` → activation → ``nn.Linear``).
* ``"quack"`` — QuACK fused GEMM+activation kernels (Hopper/Blackwell only).
  Requires ``quack-kernels >= 0.3.0``, ``bias=False``, dims divisible by 8,
  and a supported activation (``glu``, ``swiglu``).  Raises at init if any
  constraint is violated.
"""

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvsubquadratic.lazy_config import LazyConfig, instantiate


# ── QuACK availability ────────────────────────────────────────────────────────
_QUACK_MLP_MIN_VERSION = (0, 3, 0)

try:
    import quack as _quack_pkg
    from packaging.version import Version as _V

    _quack_version = _quack_pkg.__version__
    _quack_version_ok = _V(_quack_version) >= _V(".".join(str(v) for v in _QUACK_MLP_MIN_VERSION))
except Exception:
    _quack_version = None
    _quack_version_ok = False

if _quack_version_ok:
    try:
        from quack.linear import (
            act_linear_func as _quack_act_linear,
        )
        from quack.linear import (
            gated_linear_func as _quack_gated_linear,
        )
        from quack.linear import (
            linear_act_func as _quack_linear_act,
        )
        from quack.linear import (
            linear_gated_func as _quack_linear_gated,
        )

        _quack_mlp_available = True
    except ImportError:
        _quack_mlp_available = False
else:
    _quack_mlp_available = False

# Our activation name → QuACK activation name.
_QUACK_ACT_MAP: dict[str, str] = {
    "glu": "glu",
    "swiglu": "swiglu",
    "gelu": "gelu_tanh_approx",
    "relu": "relu",
}
_QUACK_GATED_ACTS = {"glu", "swiglu", "geglu", "reglu"}


def _split_to_interleaved(weight: torch.Tensor) -> torch.Tensor:
    """Convert gated-layer weight from PyTorch split layout to QuACK interleaved.

    PyTorch ``F.glu`` splits the output at the midpoint:
        gate = out[..., :N],  value = out[..., N:]

    QuACK's fused gated kernels use interleaved layout:
        gate = out[..., 0::2],  value = out[..., 1::2]

    This rearranges the *rows* of a ``(2N, D)`` weight matrix so that
    the GEMM output lands in QuACK's expected layout.  Uses
    ``torch.stack`` + ``reshape`` so autograd can differentiate through it.
    """
    N = weight.shape[0] // 2
    # QuACK interleaved layout: even rows = value, odd rows = gate.
    return torch.stack([weight[N:], weight[:N]], dim=1).reshape_as(weight)


def _quack_mlp_forward(
    x: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """Dispatch to the appropriate QuACK fused kernel (gated vs non-gated)."""
    quack_act = _QUACK_ACT_MAP[activation]
    if activation in _QUACK_GATED_ACTS:
        w1 = _split_to_interleaved(weight1)
        preact, postact = _quack_linear_gated(
            x,
            w1,
            quack_act,
            store_preact=torch.is_grad_enabled(),
        )
        return _quack_gated_linear(preact, weight2, postact, activation=quack_act)
    else:
        preact, postact = _quack_linear_act(
            x,
            weight1,
            quack_act,
            store_preact=torch.is_grad_enabled(),
        )
        return _quack_act_linear(preact, weight2, postact, activation=quack_act)


# ── Validation helpers ────────────────────────────────────────────────────────


def _validate_quack_backend(
    activation: str,
    bias: bool,
    dim: int,
    hidden_dim: int,
) -> None:
    """Raise ``ValueError`` at init time if QuACK constraints are not met."""
    min_ver = ".".join(str(v) for v in _QUACK_MLP_MIN_VERSION)

    if not _quack_mlp_available:
        if _quack_version is not None and not _quack_version_ok:
            raise ValueError(
                f"MLP backend='quack' requires quack-kernels >= {min_ver}, "
                f"but {_quack_version} is installed. "
                f"Run: pip install --upgrade quack-kernels"
            )
        raise ValueError(
            f"MLP backend='quack' requires quack-kernels >= {min_ver}, "
            f"but it is not installed. "
            f"Run: pip install quack-kernels"
        )

    if activation not in _QUACK_ACT_MAP:
        raise ValueError(
            f"MLP backend='quack' does not support activation='{activation}'. Supported: {sorted(_QUACK_ACT_MAP)}."
        )

    if bias:
        raise ValueError("MLP backend='quack' requires bias=False. QuACK fused GEMM kernels do not support bias.")

    if dim % 8 != 0 or hidden_dim % 8 != 0:
        raise ValueError(
            f"MLP backend='quack' requires dim and hidden_dim divisible by 8, got dim={dim}, hidden_dim={hidden_dim}."
        )


class MLP(nn.Module):
    """Two-layer MLP with optional QuACK fused GEMM+activation backend.

    Args:
        dim: Input and output dimension.
        activation: Activation function ('relu', 'gelu', 'silu', 'glu', 'swiglu').
        dropout_cfg: LazyConfig for the dropout layer.
        expansion_factor: Hidden-dim multiplier.
        bias: Whether to use bias in linear layers.
        backend: ``"torch"`` for pure PyTorch, ``"quack"`` for fused kernels.
        init_method_in: Optional init for first layer weights.
        init_method_out: Optional init for second layer weights.
    """

    def __init__(
        self,
        dim: int,
        activation: Literal["relu", "gelu", "silu", "glu", "swiglu"],
        dropout_cfg: LazyConfig,
        expansion_factor: float = 2.0,
        bias: bool = False,
        backend: Literal["torch", "quack"] = "torch",
        init_method_in: Callable[[torch.Tensor], torch.Tensor] | None = None,
        init_method_out: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """Initialize the MLP."""
        super().__init__()
        self.hidden_dim = int(dim * expansion_factor)
        self.activation = activation
        self.backend = backend

        # Check if using GLU variants which require double width for hidden layers
        self.is_glu_variant = activation in ["glu", "swiglu"]
        glu_factor = 2 if self.is_glu_variant else 1

        # Validate QuACK constraints eagerly
        if backend == "quack":
            _validate_quack_backend(activation, bias, dim, self.hidden_dim)

        # Construct first layer
        self.layer1 = nn.Linear(dim, self.hidden_dim * glu_factor, bias=bias)
        if init_method_in is not None:
            init_method_in(self.hidden_dim * glu_factor)(self.layer1.weight.data)
            if self.layer1.bias is not None:
                nn.init.zeros_(self.layer1.bias)

        # Construct dropout
        self.dropout = instantiate(dropout_cfg)

        # Construct second layer
        self.layer2 = nn.Linear(self.hidden_dim, dim, bias=bias)
        if init_method_out is not None:
            init_method_out(dim)(self.layer2.weight.data)
            if self.layer2.bias is not None:
                nn.init.zeros_(self.layer2.bias)

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

    def flop_count(self, num_tokens: int) -> int:
        """Count FLOPs for a two-layer MLP applied to ``num_tokens`` tokens.

        Structure: Linear(dim, hidden_dim * glu_factor) -> activation -> Linear(hidden_dim, dim).

        FLOPs breakdown (T = num_tokens):
          1. layer1 (Linear):  2 * T * ``self.layer1.in_features`` * ``self.layer1.out_features``
             For GLU/SwiGLU, out_features = 2 * hidden_dim (gate doubles width).
          2. Activation: T * ``self.hidden_dim`` (elementwise).
             For GLU variants, an additional T * ``self.hidden_dim`` for the
             gate multiply (SiLU on one half + elementwise product).
          3. layer2 (Linear):  2 * T * ``self.layer2.in_features`` * ``self.layer2.out_features``

        Convention: 1 MAC = 2 FLOPs for linear layers.
                    Activations count as 1 FLOP per element.

        Args:
            num_tokens: Number of tokens (positions) the MLP is applied to.

        Returns:
            Total FLOPs as an integer.
        """
        T = num_tokens
        flops = 0
        # layer1
        flops += 2 * T * self.layer1.in_features * self.layer1.out_features
        # Activation
        if self.is_glu_variant:
            # SiLU/sigmoid on one half + elementwise gate product
            flops += 2 * T * self.hidden_dim
        else:
            flops += T * self.hidden_dim
        # layer2
        flops += 2 * T * self.layer2.in_features * self.layer2.out_features
        return flops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        if self.backend == "quack":
            return _quack_mlp_forward(x, self.layer1.weight, self.layer2.weight, self.activation)

        # PyTorch path
        x = self.layer1(x)
        x = self._apply_activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
