"""FiLM (Feature-wise Linear Modulation) components for kernel conditioning.

Provides:
- KernelFiLMGenerator: MLP that maps a conditioning vector to per-layer (gamma, beta) pairs
  for modulating SIREN hidden layers.
- RegisterPooling: Learnable weighted average over register tokens to produce a single
  conditioning vector per sample.
- RegisterCompressConcat: Compress each register token via a shared linear layer and
  concatenate, producing a conditioning vector that preserves per-register identity.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelFiLMGenerator(nn.Module):
    """Generates per-layer FiLM (gamma, beta) pairs from a conditioning vector.

    Maps [B, cond_dim] -> list of num_film_layers x (gamma [B, kernel_hidden_dim], beta [B, kernel_hidden_dim]).

    The output bias is initialized to ``(gamma=1, beta=0)`` per layer so that
    ``gamma * h + beta = h`` at init (identity modulation).  All biases in the
    MLP are **always** excluded from weight decay.

    Args:
        cond_dim: Dimensionality of the conditioning input.
        kernel_hidden_dim: Hidden dimension of the SIREN layers to modulate.
        num_film_layers: Number of (gamma, beta) pairs to produce (one per SIREN hidden layer).
        film_hidden_dim: Hidden dimension of the FiLM generator MLP (bottleneck).
        no_weight_decay: Controls weight decay for FiLM **weight** parameters.
            All biases are always excluded from weight decay regardless of
            this setting.

            - ``True``: all parameters excluded from weight decay (``_no_weight_decay=True``).
            - ``float``: weight parameters placed in a dedicated optimizer group
              with this weight decay value (``_weight_decay=<value>``).
              Useful for mild regularization (e.g. ``1e-3``) without full WD.
            - ``False`` (default): weight parameters use the global optimizer weight decay.
        init_type: How the output layer of the MLP is initialized:

            - ``"identity"``: Output weights=0, bias=(gamma=1, beta=0).
              Exact identity at init.
            - ``"small_random"``: Same bias but with output weights drawn from
              N(0, ``init_std``) to break symmetry.  Near-identity at init.
        init_std: Standard deviation for output-layer weight init when
            ``init_type="small_random"``.  Ignored for ``"identity"``.
    """

    def __init__(  # noqa: D107
        self,
        cond_dim: int,
        kernel_hidden_dim: int,
        num_film_layers: int,
        film_hidden_dim: int = 64,
        no_weight_decay: bool | float = False,
        init_type: Literal["identity", "small_random"] = "identity",
        init_std: float = 1e-4,
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

        self._init_weights(init_type, init_std)

        # Biases are always excluded from weight decay.
        for module in self.mlp.modules():
            if hasattr(module, "bias") and module.bias is not None:
                module.bias._no_weight_decay = True

        # Mark weight parameters for weight decay handling
        if isinstance(no_weight_decay, float):
            for name, param in self.named_parameters():
                if not getattr(param, "_no_weight_decay", False):
                    param._weight_decay = no_weight_decay
        elif no_weight_decay:
            for param in self.parameters():
                param._no_weight_decay = True

    def _init_weights(self, init_type: Literal["identity", "small_random"], init_std: float):
        """Initialize the MLP output layer according to ``init_type``.

        The bias is always set to ``(gamma=1, beta=0)`` per layer so that
        ``gamma * h + beta = h`` (identity modulation) at init.

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
            for i in range(self.num_film_layers):
                offset = i * 2 * self.kernel_hidden_dim
                bias[offset : offset + self.kernel_hidden_dim].fill_(1.0)  # gamma -> 1
                bias[offset + self.kernel_hidden_dim : offset + 2 * self.kernel_hidden_dim].fill_(0.0)  # beta -> 0

    def flop_count(self) -> int:
        """Count FLOPs for the FiLM generator MLP (one sample).

        The MLP maps a single conditioning vector [cond_dim] to FiLM
        parameters [num_film_layers * 2 * kernel_hidden_dim]:

          Linear(cond_dim, film_hidden_dim)  ->  GELU  ->  Linear(film_hidden_dim, out_dim)

        FLOPs breakdown:
          1. First Linear:    2 * cond_dim * film_hidden_dim
             (cond_dim = ``self.mlp[0].in_features``,
              film_hidden_dim = ``self.mlp[0].out_features``)
          2. GELU activation:  film_hidden_dim  (elementwise)
          3. Second Linear:   2 * film_hidden_dim * out_dim
             (out_dim = num_film_layers * 2 * kernel_hidden_dim
              = ``self.mlp[2].out_features``)

        This runs once per sample per CKConvND layer that uses FiLM.

        Returns:
            Total FLOPs as an integer.
        """
        linear1 = self.mlp[0]
        linear2 = self.mlp[2]
        flops = 0
        flops += 2 * linear1.in_features * linear1.out_features
        flops += linear1.out_features  # GELU
        flops += 2 * linear2.in_features * linear2.out_features
        return flops

    def forward(self, conditioning: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate FiLM parameters from the conditioning vector.

        Args:
            conditioning: [B, cond_dim]

        Returns:
            List of (gamma, beta) tuples, each [B, kernel_hidden_dim].
            Callers apply ``gamma * h + beta``.
        """
        out = self.mlp(conditioning)  # [B, num_film_layers * 2 * kernel_hidden_dim]
        chunks = out.chunk(self.num_film_layers, dim=-1)  # num_film_layers x [B, 2 * kernel_hidden_dim]
        return [
            c.chunk(2, dim=-1) for c in chunks
        ]  # num_film_layers x (gamma [B, kernel_hidden_dim], beta [B, kernel_hidden_dim])


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

    def flop_count(self, dim: int) -> int:
        """Count FLOPs for learnable weighted average over register tokens.

        Operations (R = ``self.logits.shape[0]`` = num_registers, D = dim):
          1. Softmax over R logits:  ~3 * R  (exp + sum + divide, amortized).
          2. Weighted sum via einsum("r, b r c -> b c"):
             R * D multiplies + (R - 1) * D adds  ≈  2 * R * D.

        Total: 3 * R + 2 * R * D.

        Args:
            dim: Channel dimension (C) of the register tokens.

        Returns:
            Total FLOPs as an integer.
        """
        num_registers = self.logits.shape[0]
        return 3 * num_registers + 2 * num_registers * dim

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


class RegisterCompressConcat(nn.Module):
    """Compress each register token and concatenate into a single conditioning vector.

    Each register token is passed through a shared linear projection that reduces
    its dimensionality from ``hidden_dim`` to ``compressed_dim``.  The compressed
    tokens are then concatenated along the feature axis, producing a conditioning
    vector of size ``num_registers * compressed_dim`` that preserves per-register
    identity (unlike :class:`RegisterPooling` which averages them).

    Inspired by Mamba-Reg (Wang et al., 2024) which distributes and individually
    reads out register tokens rather than pooling them.

    Args:
        num_registers: Number of register tokens expected.
        hidden_dim: Channel dimension of each register token.
        compressed_dim: Output dimension per register after compression.
            The final conditioning vector has size ``num_registers * compressed_dim``.
    """

    def __init__(self, num_registers: int, hidden_dim: int, compressed_dim: int):  # noqa: D107
        super().__init__()
        self.num_registers = num_registers
        self.compressed_dim = compressed_dim
        self.compress = nn.Linear(hidden_dim, compressed_dim, bias=False)

    @property
    def out_dim(self) -> int:
        """Dimensionality of the output conditioning vector."""
        return self.num_registers * self.compressed_dim

    def flop_count(self, dim: int) -> int:
        """Count FLOPs for compress-and-concatenate (one sample).

        Operations (R = num_registers, D_in = ``dim``, D_out = compressed_dim):
          1. Shared linear applied R times:  R * 2 * D_in * D_out
          2. Concatenation is a view/copy, not a FLOP.

        Args:
            dim: Channel dimension of the register tokens (should match ``hidden_dim``).

        Returns:
            Total FLOPs as an integer.
        """
        return self.num_registers * 2 * dim * self.compressed_dim

    def forward(self, registers: torch.Tensor) -> torch.Tensor:
        """Compress each register and concatenate.

        Args:
            registers: [B, num_registers, hidden_dim]

        Returns:
            [B, num_registers * compressed_dim] conditioning vector.
        """
        compressed = self.compress(registers)  # [B, R, compressed_dim]
        return compressed.flatten(start_dim=1)  # [B, R * compressed_dim]

    def extra_repr(self) -> str:  # noqa: D102
        return f"num_registers={self.num_registers}, compressed_dim={self.compressed_dim}, out_dim={self.out_dim}"
