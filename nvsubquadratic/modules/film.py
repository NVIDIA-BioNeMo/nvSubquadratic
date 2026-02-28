"""FiLM (Feature-wise Linear Modulation) components for kernel conditioning.

Provides:
- KernelFiLMGenerator: MLP that maps a conditioning vector to per-layer (gamma, beta) pairs
  for modulating SIREN hidden layers.
- RegisterPooling: Learnable weighted average over register tokens to produce a single
  conditioning vector per sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelFiLMGenerator(nn.Module):
    """Generates per-layer FiLM (gamma, beta) pairs from a conditioning vector.

    Maps [B, cond_dim] -> list of num_film_layers x (gamma [B, kernel_hidden_dim], beta [B, kernel_hidden_dim]).

    Args:
        cond_dim: Dimensionality of the conditioning input.
        kernel_hidden_dim: Hidden dimension of the SIREN layers to modulate.
        num_film_layers: Number of (gamma, beta) pairs to produce (one per SIREN hidden layer).
        film_hidden_dim: Hidden dimension of the FiLM generator MLP (bottleneck).
    """

    def __init__(
        self,
        cond_dim: int,
        kernel_hidden_dim: int,
        num_film_layers: int,
        film_hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_film_layers = num_film_layers
        self.kernel_hidden_dim = kernel_hidden_dim

        out_dim = num_film_layers * 2 * kernel_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(film_hidden_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize so that gamma ~1 and beta ~0 at the start (identity modulation)."""
        final_linear = self.mlp[-1]
        nn.init.zeros_(final_linear.weight)
        with torch.no_grad():
            bias = final_linear.bias
            for i in range(self.num_film_layers):
                offset = i * 2 * self.kernel_hidden_dim
                bias[offset : offset + self.kernel_hidden_dim].fill_(1.0)  # gamma -> 1
                bias[offset + self.kernel_hidden_dim : offset + 2 * self.kernel_hidden_dim].fill_(0.0)  # beta -> 0

    def forward(self, conditioning: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate FiLM parameters from the conditioning vector.

        Args:
            conditioning: [B, cond_dim]

        Returns:
            List of (gamma, beta) tuples, each [B, kernel_hidden_dim].
        """
        out = self.mlp(conditioning)  # [B, num_film_layers * 2 * kernel_hidden_dim]
        chunks = out.chunk(self.num_film_layers, dim=-1)  # num_film_layers x [B, 2 * kernel_hidden_dim]
        return [c.chunk(2, dim=-1) for c in chunks]  # list of (gamma, beta) tuples


class RegisterPooling(nn.Module):
    """Learnable weighted average over register tokens.

    Produces a single conditioning vector per sample from multiple register tokens.

    Args:
        num_registers: Number of register tokens to pool over.
    """

    def __init__(self, num_registers: int):
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

    def extra_repr(self) -> str:
        return f"num_registers={self.logits.shape[0]}"
