"""FiLM (Feature-wise Linear Modulation) components for kernel conditioning.

Provides:
- KernelFiLMGenerator: MLP that maps a conditioning vector to per-layer (gamma, beta) pairs
  for modulating SIREN hidden layers.
- RegisterPooling: Learnable weighted average over register tokens to produce a single
  conditioning vector per sample.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


FiLMParameterization = Literal["residual", "direct"]
FiLMInitType = Literal["identity", "small_random"]


class KernelFiLMGenerator(nn.Module):
    """Generates per-layer FiLM (gamma, beta) pairs from a conditioning vector.

    Maps [B, cond_dim] -> list of num_film_layers x (gamma [B, kernel_hidden_dim], beta [B, kernel_hidden_dim]).

    Args:
        cond_dim: Dimensionality of the conditioning input.
        kernel_hidden_dim: Hidden dimension of the SIREN layers to modulate.
        num_film_layers: Number of (gamma, beta) pairs to produce (one per SIREN hidden layer).
        film_hidden_dim: Hidden dimension of the FiLM generator MLP (bottleneck).
        film_parameterization: How the FiLM modulation is applied:

            - ``"residual"``: Modulation is ``(1 + gamma) * h + beta``; the +1 is
              folded into gamma in :meth:`forward`.  Identity = gamma=0, beta=0.
            - ``"direct"``: Modulation is ``gamma * h + beta``.
              Identity = gamma=1, beta=0.
        no_weight_decay: Controls weight decay for FiLM parameters.

            - ``True``: all parameters excluded from weight decay (``_no_weight_decay=True``).
            - ``float``: all parameters placed in a dedicated optimizer group
              with this weight decay value (``_weight_decay=<value>``).
              Useful for mild regularization (e.g. ``1e-3``) without full WD.
            - ``False`` (default): parameters use the global optimizer weight decay.
        init_type: How the output layer of the MLP is initialized:

            - ``"identity"``: Output weights=0, bias set to the identity point
              for the chosen parameterization (bias=0 for residual, bias=(1,0)
              for direct).  Exact identity at init.
            - ``"small_random"``: Same bias as ``"identity"`` but with output
              weights drawn from N(0, ``init_std``) to break symmetry.
              Near-identity at init.
        init_std: Standard deviation for output-layer weight init when
            ``init_type="small_random"``.  Ignored for ``"identity"``.
        gamma_max: When set, applies a soft tanh bound to every gamma output
            so that the raw deviation from identity stays in
            ``[-gamma_max, gamma_max]`` (bounded re-parameterization,
            cf. AdaLN-Zero / DP-aware AdaLN-Zero).  Requires
            ``film_parameterization="residual"``.
    """

    def __init__(  # noqa: D107
        self,
        cond_dim: int,
        kernel_hidden_dim: int,
        num_film_layers: int,
        film_hidden_dim: int = 64,
        film_parameterization: FiLMParameterization = "residual",
        no_weight_decay: bool | float = False,
        init_type: FiLMInitType = "small_random",
        init_std: float = 1e-4,
        gamma_max: float | None = None,
    ):
        super().__init__()
        self.num_film_layers = num_film_layers
        self.kernel_hidden_dim = kernel_hidden_dim
        self.residual = film_parameterization == "residual"
        self.gamma_max = gamma_max
        if gamma_max is not None and not self.residual:
            raise ValueError(
                "gamma_max requires film_parameterization='residual' "
                "(tanh bound is defined around the identity point gamma=1)."
            )

        out_dim = num_film_layers * 2 * kernel_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(film_hidden_dim, out_dim),
        )

        # Initialize the MLP output layer
        self._init_weights(init_type, init_std)

        # Mark parameters for weight decay handling
        if isinstance(no_weight_decay, float):
            for param in self.parameters():
                param._weight_decay = no_weight_decay
        elif no_weight_decay:
            for param in self.parameters():
                param._no_weight_decay = True

    def _init_weights(self, init_type: FiLMInitType, init_std: float):
        """Initialize the MLP output layer according to ``init_type``.

        The bias is always set to the identity point for the chosen
        parameterization: zero for residual (since ``(1+0)*h+0 = h``),
        or ``(gamma=1, beta=0)`` for direct (since ``1*h+0 = h``).

        ``"small_random"`` additionally gives the output weights a small random
        perturbation to break the zero-weight saddle point.
        """
        final_linear = self.mlp[-1]

        if init_type == "identity":
            nn.init.zeros_(final_linear.weight)
        elif init_type == "small_random":
            nn.init.normal_(final_linear.weight, mean=0.0, std=init_std)
        else:
            raise ValueError(f"Unknown init_type: {init_type!r}. Expected 'identity' or 'small_random'.")

        with torch.no_grad():
            bias = final_linear.bias
            if self.residual:
                bias.zero_()
            else:
                for i in range(self.num_film_layers):
                    offset = i * 2 * self.kernel_hidden_dim
                    bias[offset : offset + self.kernel_hidden_dim].fill_(1.0)
                    bias[offset + self.kernel_hidden_dim : offset + 2 * self.kernel_hidden_dim].fill_(0.0)

    def forward(self, conditioning: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate FiLM parameters from the conditioning vector.

        Args:
            conditioning: [B, cond_dim]

        Returns:
            List of (gamma, beta) tuples, each [B, kernel_hidden_dim].
            Callers always apply ``gamma * h + beta``.  When using residual
            parameterization, the +1 offset is already folded into gamma here.
        """
        out = self.mlp(conditioning)  # [B, num_film_layers * 2 * kernel_hidden_dim]
        chunks = out.chunk(self.num_film_layers, dim=-1)  # num_film_layers x [B, 2 * kernel_hidden_dim]
        pairs = [c.chunk(2, dim=-1) for c in chunks]  # list of (gamma, beta) tuples
        if self.residual:
            if self.gamma_max is not None:
                gm = self.gamma_max
                pairs = [(1.0 + gm * torch.tanh(gamma / gm), beta) for gamma, beta in pairs]
            else:
                pairs = [(1 + gamma, beta) for gamma, beta in pairs]
        return pairs


class RegisterPooling(nn.Module):
    """Learnable weighted average over register tokens.

    Produces a single conditioning vector per sample from multiple register tokens.

    Args:
        num_registers: Number of register tokens to pool over.
    """

    def __init__(self, num_registers: int):  # noqa: D107
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_registers))
        self.logits._no_weight_decay = True

    def forward(self, registers: torch.Tensor) -> torch.Tensor:
        """Pool register tokens into a single vector.

        Args:
            registers: [B, num_registers, C]

        Returns:
            [B, C] pooled conditioning vector.
        """
        weights = F.softmax(self.logits, dim=0)  # [num_registers]
        return torch.einsum("r, b r c -> b c", weights, registers)

    def extra_repr(self) -> str:  # noqa: D102
        return f"num_registers={self.logits.shape[0]}"
