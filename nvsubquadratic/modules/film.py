"""FiLM (Feature-wise Linear Modulation) components for kernel conditioning.

**What is FiLM?**

FiLM conditions a feature map ``x`` on an external signal ``c`` (e.g. a
timestep embedding, class label, or physics parameter vector) via an affine
transformation:

    y = γ(c) ⊙ x + β(c)

where ``γ(c)`` (scale) and ``β(c)`` (shift) are learned functions of the
conditioning vector and ``⊙`` denotes elementwise multiplication.  The
parameters ``γ`` and ``β`` are **not** fixed — they are the outputs of a
small neural network (the *FiLM generator*) evaluated at runtime.

In this codebase ``γ(c)`` and ``β(c)`` are vectors in ``ℝ^{kernel_hidden_dim}``
applied pointwise to each SIREN hidden activation (no spatial axis); the standard
spatial-feature-map interpretation from Perez et al. applies when the kernel
network is evaluated independently at every spatial coordinate.

*Reference*: Perez et al., "FiLM: Visual Reasoning with a General
Conditioning Layer", arXiv:1709.07871 (2017).

**Why is FiLM useful?**

* **Diffusion models** — inject the noising timestep *t* so that each
  residual block adapts its scale and shift to the current noise level
  (see DiT, Peebles & Xie, 2022).
* **Class conditioning** — modulate intermediate feature maps by a class
  embedding, enabling a single model to behave differently across classes
  without separate heads.
* **Physics / PDE parameter conditioning** — allow a learned PDE solver to
  generalise across equation parameters (viscosity, Reynolds number, etc.)
  by FiLM-conditioning the implicit neural operator kernel.
* **SIREN kernel modulation** — within :mod:`nvsubquadratic.modules.kernels_nd`,
  FiLM adjusts the hidden activations of each SIREN layer (SIREN = Sinusoidal
  Representation Network, Sitzmann et al. 2020, arXiv:2006.09661), effectively
  steering the learned convolution kernel per sample or per timestep.

**Conditioning signal sources**

The conditioning vector ``c ∈ ℝ^{cond_dim}`` can come from any source —
a timestep MLP, class embedding lookup, or physics parameter encoder.
In diffusion or class-conditioned settings any ``[B, cond_dim]`` tensor is
accepted; within this codebase it typically originates from register tokens
appended to the sequence.  See :class:`RegisterPooling` and
:class:`RegisterCompressConcat` for the two register encoders provided here, and
:class:`KernelFiLMGenerator` for the FiLM generator that consumes the result.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelFiLMGenerator(nn.Module):
    """MLP that generates per-layer FiLM (γ, β) pairs from a conditioning vector.

    Given a conditioning signal ``c ∈ ℝ^{cond_dim}`` (e.g. from register tokens
    processed by :class:`RegisterPooling` or :class:`RegisterCompressConcat`),
    this module produces one ``(γ_l, β_l)`` pair per SIREN hidden layer ``l``
    (SIREN = Sinusoidal Representation Network, Sitzmann et al. 2020,
    arXiv:2006.09661; see :mod:`nvsubquadratic.modules.kernels_nd`):

        h_l ← γ_l(c) ⊙ h_l + β_l(c)

    The generator itself is a two-layer MLP with a GELU non-linearity:

        c → Linear(cond_dim, film_hidden_dim) → GELU
          → Linear(film_hidden_dim, num_film_layers × 2 × kernel_hidden_dim)

    The flat output is split into ``num_film_layers`` chunks; each chunk is
    further split in half to give ``(γ_l, β_l) ∈ ℝ^{kernel_hidden_dim}``.

    **Initialization strategy** — The output layer is initialized so that at
    the start of training ``γ_l = 1`` and ``β_l = 0`` for every layer, making
    FiLM an *identity* modulation.  This prevents early instability when
    the conditioning signal is still uninformative.  The ``"small_random"``
    variant perturbs the output weights slightly to break weight-symmetry
    while keeping the bias-induced identity.

    **Weight-decay handling** — All biases are permanently excluded from
    weight decay (``_no_weight_decay = True``).  Weight matrices can be
    excluded entirely (``no_weight_decay=True``) or assigned a custom decay
    value (``no_weight_decay=<float>``).

    Attributes:
        num_film_layers (int): Number of ``(γ, β)`` pairs produced.
        kernel_hidden_dim (int): Feature dimension of each SIREN hidden layer.
        mlp (nn.Sequential): Two-layer MLP mapping ``[*, cond_dim]`` →
            ``[*, num_film_layers × 2 × kernel_hidden_dim]`` via a
            ``film_hidden_dim``-dimensional bottleneck (Linear → GELU → Linear).

    Args:
        cond_dim: Dimensionality of the conditioning input ``c``.
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
        if num_film_layers < 1:
            raise ValueError(f"num_film_layers must be >= 1, got {num_film_layers}")
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

        Args:
            init_type: Initialization strategy — ``"identity"`` zeros the output
                weights; ``"small_random"`` draws them from N(0, ``init_std``).
            init_std: Standard deviation used when ``init_type="small_random"``.

        Raises:
            ValueError: If ``init_type`` is not ``"identity"`` or ``"small_random"``.
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
        """Generate per-layer FiLM parameters from the conditioning vector.

        Runs the two-layer MLP on ``conditioning`` and splits the flat output
        into ``num_film_layers`` ``(γ, β)`` pairs.  Each pair should be applied
        by the SIREN caller as ``h_l ← γ_l ⊙ h_l + β_l``.

        Args:
            conditioning: Conditioning vector of shape ``[B, cond_dim]``.  Typically
                produced by :class:`RegisterPooling` or :class:`RegisterCompressConcat`.

        Returns:
            A list of ``num_film_layers`` tuples ``(gamma, beta)``, where each
            tensor has shape ``[B, kernel_hidden_dim]``.  Index ``0`` corresponds
            to the first (shallowest) SIREN hidden layer and index
            ``num_film_layers - 1`` to the deepest.
        """
        out = self.mlp(conditioning)  # [B, num_film_layers * 2 * kernel_hidden_dim]
        chunks = out.chunk(self.num_film_layers, dim=-1)  # num_film_layers x [B, 2 * kernel_hidden_dim]
        return [
            c.chunk(2, dim=-1) for c in chunks
        ]  # num_film_layers x (gamma [B, kernel_hidden_dim], beta [B, kernel_hidden_dim])


class RegisterPooling(nn.Module):
    """Learnable softmax-weighted average over register tokens.

    Produces a single conditioning vector ``c ∈ ℝ^C`` per sample from a set of
    ``R`` register tokens.  The pooling weights are learned scalars ``w ∈ ℝ^R``
    passed through a softmax, so the contribution of each register is
    non-negative and the weights sum to one:

        c = Σ_r  softmax(w)_r · x_r

    This is a lightweight alternative to :class:`RegisterCompressConcat` when a
    scalar summary per register is sufficient.  All register information is
    blended into a single ``[B, C]`` vector that can be passed directly to
    :class:`KernelFiLMGenerator`.

    The learned logit vector ``w`` is always excluded from weight decay via
    ``_no_weight_decay = True``.

    Attributes:
        logits (nn.Parameter): Learnable unnormalized pooling weights of shape
            ``[num_registers]``, initialised to zero (uniform softmax at init).

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
        """Pool register tokens into a single conditioning vector.

        Applies softmax to the learned logits and computes a weighted sum over
        the register dimension:

            c_b = Σ_r  softmax(logits)_r · registers_{b,r,:}

        Args:
            registers: Register token tensor of shape ``[B, num_registers, C]``,
                where ``B`` is the batch size and ``C`` is the channel dimension.
                The number of registers along axis 1 must equal ``num_registers``
                passed to ``__init__``; a mismatch will raise a ``RuntimeError``
                from ``torch.einsum``.

        Returns:
            Pooled conditioning vector of shape ``[B, C]``.
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

    Formally, for a batch of register tensors ``X ∈ ℝ^{B × R × D}``:

        compressed_r = W · x_r,   W ∈ ℝ^{compressed_dim × hidden_dim}  (shared across r)
        c = [compressed_0 ‖ compressed_1 ‖ … ‖ compressed_{R-1}]  ∈ ℝ^{R · compressed_dim}

    The output ``c`` is then consumed by :class:`KernelFiLMGenerator` whose
    ``cond_dim`` must equal ``num_registers * compressed_dim`` (see
    :attr:`out_dim`).

    Inspired by Mamba-Reg (Wang et al., 2024) which distributes and individually
    reads out register tokens rather than pooling them.

    Attributes:
        num_registers (int): Number of register tokens expected on the sequence axis.
        hidden_dim (int): Input channel dimension of each register token.
        compressed_dim (int): Output channel dimension per register after compression.
        compress (nn.Linear): Shared weight-only (no bias) projection
            ``hidden_dim → compressed_dim`` applied independently to each register.

    Args:
        num_registers: Number of register tokens expected.
        hidden_dim: Channel dimension of each register token.
        compressed_dim: Output dimension per register after compression.
            The final conditioning vector has size ``num_registers * compressed_dim``.
    """

    def __init__(self, num_registers: int, hidden_dim: int, compressed_dim: int):  # noqa: D107
        super().__init__()
        self.num_registers = num_registers
        self.hidden_dim = hidden_dim
        self.compressed_dim = compressed_dim
        self.compress = nn.Linear(hidden_dim, compressed_dim, bias=False)

    @property
    def out_dim(self) -> int:
        """Dimensionality of the output conditioning vector.

        Returns:
            ``num_registers * compressed_dim``, which must match ``cond_dim``
            of any downstream :class:`KernelFiLMGenerator`.
        """
        return self.num_registers * self.compressed_dim

    def flop_count(self, hidden_dim: int) -> int:
        """Count FLOPs for compress-and-concatenate (one sample).

        Operations (R = num_registers, D_in = ``hidden_dim``, D_out = compressed_dim):
          1. Shared linear applied R times:  R * 2 * D_in * D_out
          2. Concatenation is a view/copy, not a FLOP.

        Args:
            hidden_dim: Channel dimension of the register tokens; must equal the
                ``hidden_dim`` argument passed to ``__init__``.

        Returns:
            Total FLOPs as an integer.
        """
        return self.num_registers * 2 * hidden_dim * self.compressed_dim

    def forward(self, registers: torch.Tensor) -> torch.Tensor:
        """Compress each register token and concatenate into a flat conditioning vector.

        Applies the shared linear projection to every register independently
        (via broadcasting over the register axis), then flattens the register
        and compressed-channel axes into a single vector per sample.

        Args:
            registers: Register token tensor of shape ``[B, num_registers, hidden_dim]``,
                where ``B`` is the batch size.  The number of registers along axis 1
                must equal ``num_registers`` passed to ``__init__``.

        Returns:
            Flat conditioning vector of shape ``[B, num_registers * compressed_dim]``.
            Pass this directly as the ``conditioning`` argument of
            :class:`KernelFiLMGenerator`.
        """
        compressed = self.compress(registers)  # [B, R, compressed_dim]
        return compressed.flatten(start_dim=1)  # [B, R * compressed_dim]

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"num_registers={self.num_registers}, hidden_dim={self.hidden_dim}, "
            f"compressed_dim={self.compressed_dim}, out_dim={self.out_dim}"
        )
