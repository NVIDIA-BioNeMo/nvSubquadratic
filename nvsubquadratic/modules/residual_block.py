# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Residual block implementations for ND signals.

This module provides the fundamental repeating unit of the nvsubquadratic
architecture: a residual block that wraps a *sequence mixer* and an *MLP*
with pre-norm and optional conditioning branches.  Stacking many such blocks
forms the full depth of the network (``general_purpose_resnet`` /
``classification_resnet``).

Two block variants are provided:

:class:`ResidualBlock`
    The standard pre-norm residual block used in most networks.  Each forward
    pass applies up to three gated sub-branches:

    1. **Sequence mixer branch** — pre-norm → :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer` → residual add.
    2. **Condition mixer branch** *(optional)* — pre-norm → cross-attention / conditioning mixer → residual add.
    3. **MLP branch** — pre-norm → :class:`~nvsubquadratic.modules.mlp.MLP` → residual add.

    Any branch can be disabled at config time by setting its config target to
    ``torch.nn.Identity``; the corresponding norm must also be Identity.

:class:`AdaLNZeroResidualBlock`
    A DiT-style block that replaces LayerNorm with *Adaptive LayerNorm-Zero*
    (AdaLN-Zero) modulation.  A single zero-initialised linear layer maps the
    conditioning vector to six affine parameters (shift + scale + gate for each
    of the two branches), giving the conditioning signal fine-grained control
    over both branches at initialisation-time stability (outputs ≈ 0).

All tensors follow **channels-last** layout: ``(B, *spatial_dims, C)``.
"""

from typing import Union

import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ResidualBlock(torch.nn.Module):
    """Standard pre-norm residual block for ND signals.

    Each forward pass executes up to three residual sub-branches:

    .. code-block:: text

        x ──[input_norm]──► sequence_mixer ──► dropout ──►(+)──► x'
        x'──[cond_norm]───► condition_mixer ──► dropout ──►(+)──► x''   (optional)
        x''─[mlp_norm]────► mlp ──────────────► dropout ──►(+)──► output

    Any branch is *bypassed entirely* (not just zeroed) when its ``_cfg``
    target is ``torch.nn.Identity``.  This design lets the same class serve
    pure-sequence networks (condition branch disabled), cross-attention
    encoder-decoders (condition branch enabled), and MLP-only ablations.

    All normalisation parameters (``input_norm``, ``condition_mixer_norm``,
    ``mlp_norm``) are tagged with ``_no_weight_decay = True`` so that the
    optimiser can exclude them from weight-decay groups.

    Attributes:
        sequence_mixer (torch.nn.Module): The instantiated sequence-mixing
            operator (e.g. :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`
            wrapping Hyena, Attention, CKConv, or Mamba).  May be
            ``torch.nn.Identity`` when the mixer branch is disabled.
        input_norm (torch.nn.Module): Pre-norm applied before the sequence
            mixer (e.g. LayerNorm or RMSNorm).  May be ``torch.nn.Identity``.
        condition_mixer (torch.nn.Module): Cross-attention or conditioning
            operator applied after the sequence mixer.  May be
            ``torch.nn.Identity`` to disable the conditioning branch entirely.
        condition_mixer_norm (torch.nn.Module): Pre-norm applied before
            ``condition_mixer``.  Must be ``torch.nn.Identity`` when
            ``condition_mixer`` is ``torch.nn.Identity``.
        mlp (torch.nn.Module): Position-wise MLP
            (e.g. :class:`~nvsubquadratic.modules.mlp.MLP`).  May be
            ``torch.nn.Identity`` to disable the MLP branch.
        mlp_norm (torch.nn.Module): Pre-norm applied before the MLP.  Must be
            ``torch.nn.Identity`` when ``mlp`` is ``torch.nn.Identity``.
        dropout (torch.nn.Module): Dropout (or stochastic depth) applied after
            each active branch.  Typically
            :class:`~nvsubquadratic.modules.drop_path.DropPath` or
            ``torch.nn.Dropout``.
    """

    def __init__(
        self,
        sequence_mixer_cfg: LazyConfig,
        sequence_mixer_norm_cfg: LazyConfig,
        condition_mixer_cfg: LazyConfig,
        condition_mixer_norm_cfg: LazyConfig,
        mlp_cfg: LazyConfig,
        mlp_norm_cfg: LazyConfig,
        dropout_cfg: LazyConfig,
    ):
        """Initialise the ResidualBlock.

        All positional sub-modules are supplied as
        :class:`~nvsubquadratic.lazy_config.LazyConfig` objects and
        instantiated here.  Passing ``torch.nn.Identity`` as the target for a
        ``*_cfg`` / ``*_norm_cfg`` pair disables that branch at zero cost (no
        forward computation, no parameters).

        Args:
            sequence_mixer_cfg: LazyConfig for the sequence mixer.  Typical
                targets: :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`
                (which internally wraps Hyena, Attention, CKConv, or Mamba),
                or ``torch.nn.Identity`` to skip the mixer branch entirely.
            sequence_mixer_norm_cfg: LazyConfig for the pre-norm applied
                before the sequence mixer.  **Must** be ``torch.nn.Identity``
                when ``sequence_mixer_cfg`` targets ``torch.nn.Identity``.
            condition_mixer_cfg: LazyConfig for the conditioning / cross-
                attention operator.  Pass ``torch.nn.Identity`` (the default
                in most configs) to disable the conditioning branch.  When
                active, the module's ``forward`` must accept
                ``(x, condition)`` positional arguments.
            condition_mixer_norm_cfg: LazyConfig for the pre-norm applied
                before ``condition_mixer``.  **Must** be ``torch.nn.Identity``
                when ``condition_mixer_cfg`` targets ``torch.nn.Identity``.
            mlp_cfg: LazyConfig for the position-wise MLP.  Typical target:
                :class:`~nvsubquadratic.modules.mlp.MLP`.  Pass
                ``torch.nn.Identity`` to skip the MLP branch.
            mlp_norm_cfg: LazyConfig for the pre-norm applied before the MLP.
                **Must** be ``torch.nn.Identity`` when ``mlp_cfg`` targets
                ``torch.nn.Identity``.
            dropout_cfg: LazyConfig for the dropout / stochastic-depth module
                applied after each active branch.  Typical targets:
                ``torch.nn.Dropout``,
                :class:`~nvsubquadratic.modules.drop_path.DropPath`, or
                ``torch.nn.Identity`` for no dropout.

        Raises:
            AssertionError: If a norm config does not match its corresponding
                module config — i.e. if a mixer/MLP config is ``Identity`` but
                the corresponding norm config is not (or vice versa in the
                forward-only direction).
        """
        if sequence_mixer_cfg.__target__ == torch.nn.Identity:
            assert sequence_mixer_norm_cfg.__target__ == torch.nn.Identity, (
                "Sequence mixer norm must be Identity if sequence mixer is Identity"
            )
        if mlp_cfg.__target__ == torch.nn.Identity:
            assert mlp_norm_cfg.__target__ == torch.nn.Identity, "MLP norm must be Identity if MLP is Identity"
        if condition_mixer_cfg.__target__ == torch.nn.Identity:
            assert condition_mixer_norm_cfg.__target__ == torch.nn.Identity, (
                "Condition mixer norm must be Identity if condition mixer is Identity"
            )

        super().__init__()
        # Instantiate sequence mixer layer
        self.sequence_mixer = instantiate(sequence_mixer_cfg)
        # Instantiate input norm
        self.input_norm = instantiate(sequence_mixer_norm_cfg)
        # Exclude self.input_norm from the parameter group with weight decay
        for param in self.input_norm.parameters():
            param._no_weight_decay = True

        # Instantiate cross attention layer
        self.condition_mixer = instantiate(condition_mixer_cfg)
        # Instantiate cross attention norm
        self.condition_mixer_norm = instantiate(condition_mixer_norm_cfg)
        # Exclude self.condition_mixer_norm from the parameter group with weight decay
        for param in self.condition_mixer_norm.parameters():
            param._no_weight_decay = True

        # Instantiate MLP layer
        self.mlp = instantiate(mlp_cfg)
        # Instantiate MLP norm
        self.mlp_norm = instantiate(mlp_norm_cfg)
        # Exclude self.mlp_norm from the parameter group with weight decay
        for param in self.mlp_norm.parameters():
            param._no_weight_decay = True

        # Instantiate dropout
        self.dropout = instantiate(dropout_cfg)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to the input tensor.

        Executes up to three residual sub-branches in order: sequence mixer,
        conditioning mixer (optional), and MLP (optional).  A branch is
        **skipped entirely** when its corresponding module is
        ``torch.nn.Identity``; the ``condition`` argument is only consumed
        when the conditioning branch is active.

        **Path without conditioning** (``condition_mixer`` is Identity):

        .. code-block:: text

            x → input_norm → sequence_mixer → dropout →(+)→ x'
            x' → mlp_norm → mlp → dropout →(+)→ output

        **Path with conditioning** (``condition_mixer`` is not Identity):

        .. code-block:: text

            x  → input_norm → sequence_mixer → dropout →(+)→ x'
            x' → condition_mixer_norm → condition_mixer(x', condition)
               → dropout →(+)→ x''
            x'' → mlp_norm → mlp → dropout →(+)→ output

        Args:
            x: Input feature tensor of shape
                ``(B, *spatial_dims, C)`` where ``B`` is the batch size,
                ``spatial_dims`` are one or more spatial axes (e.g. ``(H, W)``
                for 2-D images or ``(T,)`` for 1-D sequences), and
                ``C`` is the hidden channel dimension.
            condition: Conditioning tensor used by ``condition_mixer``.  Its
                shape depends on the conditioning operator — a common choice
                is ``(B, *spatial_dims_condition, C)`` for cross-attention, or
                ``(B, C)`` for a global conditioning vector.  This argument is
                **ignored** (and may safely be ``None``) when
                ``condition_mixer`` is ``torch.nn.Identity``.

        Returns:
            torch.Tensor: Output tensor of the same shape as ``x``:
            ``(B, *spatial_dims, C)``.

        Raises:
            AssertionError: If ``condition`` is ``None`` but
                ``condition_mixer`` is not ``torch.nn.Identity`` (i.e. a
                conditioning tensor is required but was not provided).
        """
        # Mixer branch
        if not isinstance(self.sequence_mixer, torch.nn.Identity):
            residual = x
            x = self.input_norm(x)
            x = self.sequence_mixer(x)
            x = self.dropout(x)
            x = x + residual

        # Cross attention branch
        if not isinstance(self.condition_mixer, torch.nn.Identity):
            assert condition is not None, "Condition must be provided if condition mixer is not Identity."
            residual = x
            x = self.condition_mixer_norm(x)
            x = self.condition_mixer(x, condition)
            x = self.dropout(x)
            x = x + residual

        # MLP branch
        if not isinstance(self.mlp, torch.nn.Identity):
            residual = x
            x = self.mlp_norm(x)
            x = self.mlp(x)
            x = self.dropout(x)
            x = x + residual
        return x


class AdaLNZeroResidualBlock(torch.nn.Module):
    """Pre-norm residual block with AdaLN-Zero conditioning (DiT-style).

    Replaces fixed LayerNorm with *Adaptive LayerNorm-Zero* (AdaLN-Zero)
    modulation, following the Scalable Diffusion Transformers (DiT) recipe
    (Peebles & Xie, 2023).  A single zero-initialised linear projection maps
    the conditioning vector to six affine parameters — shift, scale, and gate
    for each of the two branches — so that at initialisation the block outputs
    exactly zero (the residual stream is unchanged).

    **Forward computation** (one block):

    .. code-block:: text

        cond → [optional spatial mean] → condition_norm
             → SiLU → Linear(C, 6C) → split into 6 × (B, C)
               (shift_seq, scale_seq, gate_seq, shift_mlp, scale_mlp, gate_mlp)

        # Sequence mixer branch
        x_norm = sequence_norm(x)                         # pre-norm
        x_mod  = x_norm * (1 + scale_seq) + shift_seq    # AdaLN modulation
        seq_out = sequence_mixer(x_mod, conditioning=cond)
        seq_out = dropout(seq_out) * gate_seq             # zero-init gate
        x = x + seq_out

        # MLP branch
        x_norm = mlp_norm(x)
        x_mod  = x_norm * (1 + scale_mlp) + shift_mlp
        mlp_out = mlp(x_mod)
        mlp_out = dropout(mlp_out) * gate_mlp
        x = x + mlp_out

    The gate vectors are multiplied **after** dropout to provide per-token
    scaling; at init they are zero (because ``condition_proj`` weights are
    zero-initialised), so the block is a skip connection.

    Attributes:
        sequence_mixer (torch.nn.Module): Instantiated sequence-mixing
            operator.  Its ``forward`` must accept a ``conditioning`` keyword
            argument (forwarded from the pooled conditioning vector).
        sequence_norm (torch.nn.Module): Pre-norm applied to ``x`` before
            AdaLN modulation in the sequence mixer branch.
        mlp (torch.nn.Module): Position-wise MLP instantiated from
            ``mlp_cfg``.
        mlp_norm (torch.nn.Module): Pre-norm applied to ``x`` before
            AdaLN modulation in the MLP branch.
        condition_norm (torch.nn.Module): Optional normalisation applied to
            the conditioning vector before it is projected by
            ``condition_proj``.  Tagged ``_no_weight_decay``.
        dropout (torch.nn.Module): Dropout / stochastic-depth applied after
            each branch, before the gate multiply.
        condition_proj (torch.nn.Sequential): ``SiLU → Linear(C, 6C)``
            with **zero-initialised** weights and biases, producing the six
            AdaLN-Zero parameters.  Zero init ensures the block is a pure
            residual connection at the start of training.
    """

    def __init__(
        self,
        sequence_mixer_cfg: LazyConfig,
        sequence_mixer_norm_cfg: LazyConfig,
        mlp_cfg: LazyConfig,
        mlp_norm_cfg: LazyConfig,
        condition_norm_cfg: LazyConfig,
        dropout_cfg: LazyConfig,
        hidden_dim: int,
    ):
        """Initialise the AdaLNZeroResidualBlock.

        Args:
            sequence_mixer_cfg: LazyConfig for the sequence-mixing operator.
                The instantiated module's ``forward`` must accept a
                ``conditioning`` keyword argument (it receives the spatially-
                pooled conditioning vector ``cond`` of shape ``(B, C)``).
            sequence_mixer_norm_cfg: LazyConfig for the pre-norm applied
                before the AdaLN modulation of the sequence mixer branch
                (e.g. LayerNorm over ``hidden_dim``).
            mlp_cfg: LazyConfig for the position-wise MLP.
            mlp_norm_cfg: LazyConfig for the pre-norm applied before the
                AdaLN modulation of the MLP branch.
            condition_norm_cfg: LazyConfig for the normalisation applied to
                the conditioning vector before ``condition_proj``.  Use
                ``torch.nn.Identity`` to skip conditioning normalisation.
            dropout_cfg: LazyConfig for the dropout / stochastic-depth module
                applied after each branch and before the gate multiply.
            hidden_dim: Channel dimension ``C`` shared by all sub-modules.
                Used to size the ``condition_proj`` linear layer
                (``Linear(C, 6*C)``).
        """
        super().__init__()

        # Mixer branch handles spatial/temporal interactions on the residual stream.
        self.sequence_mixer = instantiate(sequence_mixer_cfg)
        self.sequence_norm = instantiate(sequence_mixer_norm_cfg)
        for param in self.sequence_norm.parameters():
            param._no_weight_decay = True

        # MLP branch refines each position independently.
        self.mlp = instantiate(mlp_cfg)
        self.mlp_norm = instantiate(mlp_norm_cfg)
        for param in self.mlp_norm.parameters():
            param._no_weight_decay = True

        # Optional pre-normalization for the conditioning vector.
        self.condition_norm = instantiate(condition_norm_cfg)
        for param in self.condition_norm.parameters():
            param._no_weight_decay = True

        # Shared dropout applied after each residual branch.
        self.dropout = instantiate(dropout_cfg)

        # Single zero-initialised projection (DiT style) producing shift/scale/gate for both branches.
        self.condition_proj = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(hidden_dim, hidden_dim * 6))
        torch.nn.init.zeros_(self.condition_proj[1].weight)
        torch.nn.init.zeros_(self.condition_proj[1].bias)

    def forward(self, x: torch.Tensor, condition: Union[torch.Tensor, None]) -> torch.Tensor:
        """Apply AdaLN-Zero residual mixing conditioned on the provided tensor.

        The conditioning tensor is reduced to a single latent vector per
        sample (spatial mean if it has spatial axes) before being projected
        to six affine parameters.  These parameters modulate the pre-norm
        outputs of both branches via element-wise affine transforms, and gate
        each branch's output before the residual add.

        Args:
            x: Input feature tensor of shape ``(B, *spatial_dims, C)`` where
                ``B`` is the batch size, ``spatial_dims`` are one or more
                spatial axes, and ``C = hidden_dim``.
            condition: Required conditioning tensor.  Shape may be:

                * ``(B, C)`` — a pre-pooled global conditioning vector
                  (e.g. a timestep / class embedding from a diffusion model).
                * ``(B, *spatial_dims_cond, C)`` — any spatial layout; the
                  forward pass reduces it to ``(B, C)`` via a mean over all
                  non-batch, non-channel axes.

                Must not be ``None``.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, *spatial_dims, C)``,
            the same shape as ``x``.

        Raises:
            ValueError: If ``condition`` is ``None``.
        """
        if condition is None:
            raise ValueError("AdaLNZeroResidualBlock requires a conditioning tensor.")

        # Collapse any spatial conditioning down to a single latent vector per item.
        cond = condition  # (B, *spatial?, hidden_dim)
        if cond.ndim >= 3:
            cond = cond.mean(dim=tuple(range(1, cond.ndim - 1)))  # (B, hidden_dim)
        cond = self.condition_norm(cond)  # (B, hidden_dim)

        # Map the conditioning vector to shift/scale/gate triplets for both branches.
        cond_mapped = self.condition_proj(cond)  # (B, 6 * hidden_dim)
        shift_seq, scale_seq, gate_seq, shift_mlp, scale_mlp, gate_mlp = cond_mapped.chunk(
            6, dim=-1
        )  # each (B, hidden_dim)

        def expand(param: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            """Broadcast a [B, hidden_dim] vector across ref's spatial axes."""
            while param.ndim < ref.ndim:
                param = param.unsqueeze(1)  # add singleton dims for broadcasting
            return param.expand(*ref.shape[:-1], param.shape[-1])  # match ref spatial layout

        # Modulate the sequence mixer with AdaLN-Zero and add its residual output.
        seq_norm = self.sequence_norm(x)  # (B, *spatial_dims, hidden_dim)
        seq_mod = seq_norm * (1.0 + expand(scale_seq, seq_norm)) + expand(
            shift_seq, seq_norm
        )  # (B, *spatial_dims, hidden_dim)
        seq_out = self.sequence_mixer(seq_mod, conditioning=cond)  # (B, *spatial_dims, hidden_dim)
        seq_out = self.dropout(seq_out)  # (B, *spatial_dims, hidden_dim)
        seq_out = seq_out * expand(gate_seq, seq_out)  # (B, *spatial_dims, hidden_dim)
        x = x + seq_out  # (B, *spatial_dims, hidden_dim)

        # Apply the same AdaLN-Zero recipe to the MLP branch.
        mlp_norm = self.mlp_norm(x)  # (B, *spatial_dims, hidden_dim)
        mlp_mod = mlp_norm * (1.0 + expand(scale_mlp, mlp_norm)) + expand(
            shift_mlp, mlp_norm
        )  # (B, *spatial_dims, hidden_dim)
        mlp_out = self.mlp(mlp_mod)  # (B, *spatial_dims, hidden_dim)
        mlp_out = self.dropout(mlp_out)  # (B, *spatial_dims, hidden_dim)
        mlp_out = mlp_out * expand(gate_mlp, mlp_out)  # (B, *spatial_dims, hidden_dim)
        x = x + mlp_out  # (B, *spatial_dims, hidden_dim)

        return x
