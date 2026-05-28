# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Add license header here


"""Channel-mixing MLP block for ND residual networks.

Within the residual block architecture (see
:mod:`nvsubquadratic.modules.residual_block`), every repeating unit has two
complementary branches:

1. **Sequence mixer** — captures long-range *spatial/temporal* interactions
   (Hyena, Attention, CKConv, Mamba, …).
2. **MLP** *(this module)* — performs *point-wise channel mixing*, refining
   each position's feature vector independently of its neighbours.

The :class:`MLP` class provides a two-layer feed-forward network that acts
as the channel-mixing branch.  Three activation variants are offered:

* **Plain GELU / ReLU / SiLU MLP** (``activation in {"gelu", "relu", "silu"}``):

  .. code-block:: text

      y = W₂(act(W₁(x)))

  ``W₁ : C → H``, ``W₂ : H → C`` where ``H = floor(expansion_factor × C)``.

* **Gated Linear Unit (GLU)** (``activation="glu"``):

  .. code-block:: text

      y = W₂(sigmoid(W₁ₐ(x)) ⊙ W₁ᵦ(x))

  ``W₁`` projects to ``2H``, then is split at the midpoint into a gate half
  and a value half.  ``W₂ : H → C``.

* **SwiGLU** (``activation="swiglu"``, the default in many modern configs):

  .. code-block:: text

      y = W₂(SiLU(W₁ₐ(x)) ⊙ W₁ᵦ(x))

  Same gated structure as GLU but with SiLU in place of sigmoid; follows
  Noam Shazeer's SwiGLU paper (arXiv:2002.05202).

Expansion-ratio semantics
--------------------------
``expansion_factor`` controls the width of the hidden (intermediate) layer
relative to the model dimension ``C``:

    H = floor(expansion_factor * C)

For **plain** activations (GELU, ReLU, SiLU) the total linear-layer parameter
count is ``C*H + H*C = 2*C*H``.  For **gated** activations (GLU, SwiGLU) the
first projection doubles to ``C → 2H``, so the count becomes
``C*2H + H*C = 3*C*H``.  To keep the parameter budget comparable between
plain and gated MLPs, gated configs are typically paired with a smaller
``expansion_factor``:

- Plain GELU at ``expansion_factor=4`` ≈ ``8C²`` parameters.
- SwiGLU at ``expansion_factor=8/3`` ≈ ``8C²`` parameters.
- A commonly used approximation is ``expansion_factor ≈ 4/3`` relative to the
  plain target.

Activation function summary
----------------------------
+----------+----------------------------------------------+----------------------------+
| Name     | Formula                                      | Notes                      |
+==========+==============================================+============================+
| ``relu`` | ``max(0, x)``                                | Plain; sparse activations  |
+----------+----------------------------------------------+----------------------------+
| ``gelu`` | ``x · Φ(x)`` (tanh approximation)           | Plain; smooth, common in   |
|          |                                              | Transformers / BERT-family |
+----------+----------------------------------------------+----------------------------+
| ``silu`` | ``x · σ(x)``                                 | Plain; also called Swish   |
+----------+----------------------------------------------+----------------------------+
| ``glu``  | ``sigmoid(a) ⊙ b``                           | Gated; ``W₁`` → 2H        |
+----------+----------------------------------------------+----------------------------+
| ``swiglu``| ``SiLU(a) ⊙ b``                            | Gated; recommended modern  |
|           |                                              | default (arXiv:2002.05202) |
+----------+----------------------------------------------+----------------------------+

Hardware back-ends
------------------
By default (``backend="torch"``) the MLP uses standard ``nn.Linear`` layers
with a PyTorch activation.  On Hopper/Blackwell GPUs with QuACK kernels
installed, setting ``backend="quack"`` enables fused GEMM+activation kernels
that reduce memory traffic.  QuACK is **experimental** (forward correctness
verified; backward and benchmark still pending) and is currently blocked by a
``NotImplementedError`` in :func:`_validate_quack_backend`.

GRN integration
---------------
The :class:`MLP` itself does not embed a GRN layer; if Global Response
Normalization (see :mod:`nvsubquadratic.modules.grn`) is desired it should be
inserted between the two linear layers by the calling code or by wrapping
:class:`MLP` in a subclass.  GRN is particularly effective inside gated-linear
units (GLU / SwiGLU) where per-channel magnitude carries semantic weight.

Tensor layout
-------------
All tensors use **channels-last** layout: ``(B, *spatial_dims, C)``, where
``B`` is the batch size, ``*spatial_dims`` are one or more spatial axes
(length-1 for 1-D sequences, ``(H, W)`` for 2-D images, etc.), and ``C`` is
the channel dimension.  The MLP is spatially agnostic — it applies the same
two-layer projection independently to every position.
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

    Args:
        weight: Weight matrix of shape ``(2N, D)`` in PyTorch split layout,
            where rows ``[:N]`` are the gate weights and rows ``[N:]`` are the
            value weights.

    Returns:
        torch.Tensor: Weight matrix of the same shape ``(2N, D)`` re-ordered
        into QuACK's interleaved layout (even rows = value, odd rows = gate).
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
    """Dispatch to the appropriate QuACK fused kernel (gated vs non-gated).

    Selects between the gated (GLU/SwiGLU) and non-gated (GELU/ReLU) QuACK
    kernel families, converts weights to QuACK's layout if necessary, and
    executes the fused forward pass.  Intermediate pre-activations are stored
    only when ``torch.is_grad_enabled()`` so that backward can reuse them.

    Args:
        x: Input tensor of shape ``(B, *spatial_dims, C)`` (channels-last).
            Must have the channel dimension divisible by 8 (QuACK constraint).
        weight1: First linear layer weight of shape ``(H * glu_factor, C)``
            where ``H`` is the hidden dimension and ``glu_factor`` is 2 for
            gated activations (GLU / SwiGLU) and 1 otherwise.
        weight2: Second linear layer weight of shape ``(C, H)``.
        activation: One of ``"glu"``, ``"swiglu"``, ``"gelu"``, ``"relu"``.
            Must be present in :data:`_QUACK_ACT_MAP`.

    Returns:
        torch.Tensor: Output tensor of shape ``(B, *spatial_dims, C)``,
        matching the input shape.
    """
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
    """Raise ``ValueError`` at init time if QuACK constraints are not met.

    Called during :class:`MLP.__init__` when ``backend="quack"`` is requested.
    Fails fast with an actionable error message rather than letting a misuse
    surface as an obscure CUDA error at first forward call.

    .. note::
        This function currently raises :exc:`NotImplementedError` unconditionally
        because the QuACK backend is experimental (backward pass and benchmark
        are not yet validated).  Use ``backend="torch"`` for all production use.

    Args:
        activation: Requested activation name.  Must be a key in
            :data:`_QUACK_ACT_MAP` (``"glu"``, ``"swiglu"``, ``"gelu"``,
            ``"relu"``).
        bias: Whether bias terms are enabled.  QuACK fused GEMM kernels do not
            support bias, so ``bias=True`` raises :exc:`ValueError`.
        dim: Input / output channel dimension ``C``.  Must be divisible by 8.
        hidden_dim: Expanded hidden dimension ``H = expansion_factor * C``.
            Must be divisible by 8.

    Raises:
        NotImplementedError: Always, because the QuACK backend is not yet
            production-ready.
        ValueError: If ``quack-kernels`` is not installed or is below
            :data:`_QUACK_MLP_MIN_VERSION`, if ``activation`` is not supported,
            if ``bias=True``, or if ``dim`` / ``hidden_dim`` are not divisible
            by 8.  (These checks are unreachable until the
            :exc:`NotImplementedError` is removed.)
    """
    raise NotImplementedError(
        "MLP backend='quack' is experimental and needs more testing "
        "(forward correctness verified, backward + benchmark pending). "
        "Use backend='torch' for now."
    )

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
    """Point-wise two-layer MLP — the channel-mixing branch of each residual block.

    Acts on each spatial position independently (no cross-position interaction),
    expanding the channel dimension by ``expansion_factor``, applying a
    non-linearity, and projecting back:

    **Plain MLP** (``activation in {"relu", "gelu", "silu"}``):

    .. code-block:: text

        y = W₂( act( W₁(x) ) )

    where ``W₁ : C → H``, ``W₂ : H → C``, and ``H = floor(expansion_factor × C)``.

    **Gated variants** (``activation in {"glu", "swiglu"}``):

    .. code-block:: text

        [a, b] = W₁(x)          # W₁ : C → 2H, split at midpoint
        y = W₂( gate(a) ⊙ b )  # W₂ : H → C

    - *GLU* (``"glu"``): ``gate(a) = sigmoid(a)``
    - *SwiGLU* (``"swiglu"``): ``gate(a) = SiLU(a)``

    Because the first projection ``W₁`` must produce ``2H`` outputs for the
    gate, ``layer1`` has shape ``(2H, C)`` when a gated activation is used
    (see ``glu_factor`` in ``__init__``), while ``layer2`` remains ``(C, H)``.
    This means the **parameter count differs** from the plain variant:

    - Plain: ``C*H + H*C = 2CH`` parameters in the two linear layers.
    - Gated: ``C*2H + H*C = 3CH`` parameters in the two linear layers.

    At ``expansion_factor=2``:

    - Plain (GELU): ``H = 2C``, params ≈ ``4C²``.
    - SwiGLU: ``H = 2C``, params ≈ ``6C²``.

    To keep parameter counts comparable, gated configs often use a smaller
    ``expansion_factor`` (e.g. ``4/3`` rather than ``2``).

    Dropout is inserted between the two layers (applied *after* the
    activation and gate product).

    The ``backend`` argument selects the compute kernel family:

    - ``"torch"`` (default): Standard ``nn.Linear`` + PyTorch activation.
      Works everywhere.
    - ``"quack"`` (*experimental, currently blocked*): QuACK fused GEMM +
      activation kernels targeting Hopper / Blackwell GPUs.  Requires
      ``quack-kernels >= 0.3.0``, ``bias=False``, channel dimensions divisible
      by 8, and a supported activation.  Currently raises
      :exc:`NotImplementedError` at init time pending backward validation.

    Attributes:
        hidden_dim (int): Expanded hidden channel dimension
            ``H = floor(expansion_factor * dim)``.  For gated variants
            this is the *post-gate* width — the actual ``layer1`` output
            width is ``2 * hidden_dim``.
        activation (str): Name of the activation function in use.
            One of ``"relu"``, ``"gelu"``, ``"silu"``, ``"glu"``,
            ``"swiglu"``.
        backend (str): Compute backend, either ``"torch"`` or ``"quack"``.
        is_glu_variant (bool): ``True`` when ``activation`` is ``"glu"`` or
            ``"swiglu"``, indicating that ``layer1`` produces ``2 * hidden_dim``
            outputs and the forward path applies a gating product.
        layer1 (nn.Linear): First linear projection.
            Shape: ``(hidden_dim * glu_factor, dim)`` where ``glu_factor``
            is 2 for gated activations, 1 otherwise.
        dropout (nn.Module): Dropout layer instantiated from ``dropout_cfg``.
            Applied between the activation (or gate product) and ``layer2``.
        layer2 (nn.Linear): Second linear projection.
            Shape: ``(dim, hidden_dim)``.  Projects back to the input
            channel dimension ``C``.
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
        """Initialise the MLP.

        Args:
            dim: Input and output channel dimension ``C``.  Both ``layer1``
                and ``layer2`` preserve this outer dimension (the network's
                residual stream width).
            activation: Non-linearity to apply between the two linear layers.
                Controls whether the MLP is plain or gated:

                - ``"relu"``, ``"gelu"``, ``"silu"``: plain MLP; ``layer1``
                  maps ``C → H``.
                - ``"glu"``: sigmoid-gated MLP; ``layer1`` maps ``C → 2H``
                  and is split into gate + value halves.
                - ``"swiglu"``: SiLU-gated MLP (recommended for modern
                  configs); same gating structure as ``"glu"`` but with
                  SiLU in place of sigmoid.

            dropout_cfg: :class:`~nvsubquadratic.lazy_config.LazyConfig`
                specifying the dropout module inserted between the activation
                and ``layer2``.  Use ``torch.nn.Identity`` (or a
                ``LazyConfig`` that targets it) for no dropout.
            expansion_factor: Multiplier that sets the hidden dimension
                ``H = floor(expansion_factor * dim)``.  Defaults to ``2.0``.
                For gated activations, the actual ``layer1`` output width is
                ``2 * H``; to keep total parameter count similar to a plain
                MLP with ``expansion_factor=2``, use ``expansion_factor ≈ 4/3``
                for SwiGLU / GLU.
            bias: Whether to include additive bias terms in ``layer1`` and
                ``layer2``.  Defaults to ``False``.  Must be ``False`` when
                ``backend="quack"``.
            backend: Compute kernel family.  ``"torch"`` (default) uses
                ``nn.Linear`` + PyTorch activation and runs on any hardware.
                ``"quack"`` enables fused GEMM+activation kernels but is
                currently experimental (raises :exc:`NotImplementedError`
                at init).
            init_method_in: Optional weight initialiser for ``layer1``.
                Expected signature: ``init_method_in(out_features)(weight)``
                — a *curried* callable that first takes the number of output
                features and returns an in-place initialiser.  Bias is always
                zero-initialised when present, regardless of this argument.
                Pass ``None`` to use PyTorch's default Kaiming uniform init.
            init_method_out: Optional weight initialiser for ``layer2``.
                Same curried signature as ``init_method_in``, applied to
                ``layer2.weight``.  Pass ``None`` for PyTorch default init.

        Raises:
            NotImplementedError: If ``backend="quack"`` (experimental; use
                ``backend="torch"`` for now).
            ValueError: If ``backend="quack"`` and any QuACK constraint is
                violated (unsupported activation, ``bias=True``, or dimension
                not divisible by 8).  Currently unreachable due to the
                :exc:`NotImplementedError` raised first.
        """
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
        """Apply the configured activation function to a pre-activation tensor.

        For plain activations (``"relu"``, ``"gelu"``, ``"silu"``) this is a
        simple element-wise call.  For gated variants (``"glu"``, ``"swiglu"``)
        the tensor is split along the last dimension and a gate product is
        applied, halving the channel count from ``2H`` to ``H``.

        Args:
            x: Pre-activation tensor of shape ``(B, *spatial_dims, H')``
               where ``H' = hidden_dim`` for plain activations and
               ``H' = 2 * hidden_dim`` for gated variants.

        Returns:
            torch.Tensor: Post-activation tensor of shape
            ``(B, *spatial_dims, hidden_dim)``.  The channel dimension is
            halved for gated activations (``"glu"``, ``"swiglu"``), unchanged
            for plain ones.

        Raises:
            ValueError: If ``self.activation`` is not one of the supported
                values (``"relu"``, ``"gelu"``, ``"silu"``, ``"glu"``,
                ``"swiglu"``).
        """
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
                Equal to the product of all spatial dimensions, i.e.
                ``H * W`` for 2-D images or ``T`` for 1-D sequences.

        Returns:
            int: Total FLOPs as an integer.
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
        """Apply the two-layer MLP to an ND feature tensor.

        Operates identically on every spatial position — no cross-position
        interaction occurs in this module.  The QuACK path bypasses dropout
        (QuACK kernels fuse both linear layers and the activation into a
        single kernel; dropout must be inserted by the caller if needed).

        Args:
            x: Input feature tensor of shape ``(B, *spatial_dims, C)`` in
                channels-last layout, where ``B`` is the batch size,
                ``spatial_dims`` are one or more spatial axes (e.g. ``(H, W)``
                for 2-D images or ``(T,)`` for 1-D sequences), and ``C``
                is the channel dimension (must equal the ``dim`` passed at
                construction time).

        Returns:
            torch.Tensor: Output tensor of shape ``(B, *spatial_dims, C)``,
            the same shape as ``x``.
        """
        if self.backend == "quack":
            return _quack_mlp_forward(x, self.layer1.weight, self.layer2.weight, self.activation)

        # PyTorch path
        x = self.layer1(x)
        x = self._apply_activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
