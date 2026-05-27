"""ViT-5 Residual Block: Pre-norm + Sequence Mixer/MLP with LayerScale and DropPath.

This module provides the specialised residual block used throughout the ViT-5
family of hierarchical vision transformers (Wang et al., 2026).  It differs
from the generic :mod:`nvsubquadratic.modules.residual_block` in the following
ViT-5-specific design choices:

**Structural differences vs. :class:`~nvsubquadratic.modules.residual_block.ResidualBlock`**

1. **No condition-mixer branch** — :class:`ResidualBlock` supports an optional
   cross-attention / conditioning branch between the sequence mixer and the MLP.
   :class:`ViT5ResidualBlock` removes this branch entirely; ViT-5 routes
   conditioning through register-token pooling instead (see point 3).

2. **LayerScale on both branches** — ViT-5 wraps both the sequence-mixer and
   MLP residual updates in independent :class:`~nvsubquadratic.modules.layer_scale.LayerScale`
   modules (initialised to a small constant, typically 1e-4).  The generic
   :class:`ResidualBlock` uses a single shared dropout and no per-branch
   learned scale.

3. **Register-token FiLM conditioning** — When ``register_pooling_cfg`` is
   provided, register tokens are extracted from the *pre-normalised* input,
   pooled into a single conditioning vector per sample, and forwarded to the
   sequence mixer as a ``conditioning`` keyword argument.  This is the primary
   mechanism by which ViT-5 communicates global context (e.g. class identity)
   to local sequence operators such as Hyena with a SIREN kernel.

4. **Optional Global Response Normalization (GRN)** — Following ConvNeXt V2,
   an optional GRN layer can be inserted after the sequence mixer output to
   promote inter-channel competition before the LayerScale + residual add.

5. **Sequence layout** — Input is always ``[B, T, C]`` (batch, tokens,
   channels); the token axis *T* concatenates patch tokens, an optional CLS
   token, register tokens, and optional zero-padding in the order
   ``[patches, (CLS,) registers, (padding,)]``.  Attention blocks receive the
   sequence with padding stripped; Hyena / subquadratic blocks receive the full
   padded sequence so that ``T % grid_w == 0``.  The generic block operates on
   arbitrary ``(B, *spatial_dims, C)`` tensors.

For the generic pre-norm residual block (Hyena / Attention / CKConv / Mamba
with optional cross-attention conditioning), see
:mod:`nvsubquadratic.modules.residual_block`.
"""

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.drop_path import DropPath
from nvsubquadratic.modules.layer_scale import LayerScale


class ViT5ResidualBlock(nn.Module):
    """ViT-5 style residual block with LayerScale and stochastic depth.

    Implements the two-branch pre-norm transformer block used in the ViT-5
    family.  Each forward pass executes:

    .. code-block:: text

        # Branch 1 — sequence mixer
        x_normed = input_norm(x)
        [cond = register_pooling(x_normed[:, s:s+R, :])  # optional]
        mixer_out = sequence_mixer(x_normed[, conditioning=cond])
        [mixer_out = grn(mixer_out)                       # optional]
        x = x + drop_path(ls_attn(mixer_out))

        # Branch 2 — MLP
        x = x + drop_path(ls_mlp(mlp(mlp_norm(x))))

    **Differences vs. the generic** :class:`~nvsubquadratic.modules.residual_block.ResidualBlock`:

    * No condition-mixer branch.  Conditioning is handled by register pooling
      inside branch 1 (see ``register_pooling_cfg``).
    * Each branch has its own :class:`~nvsubquadratic.modules.layer_scale.LayerScale`
      (``ls_attn`` / ``ls_mlp``) rather than a single shared dropout.
    * A single :class:`~nvsubquadratic.modules.drop_path.DropPath` instance is
      shared across both branches (``drop_path``).
    * Input is always ``[B, T, C]`` — the spatial dimensions are fully flattened
      into the token axis, and register tokens occupy known index positions.

    **Register-token conditioning**:

    When ``register_pooling_cfg`` is not ``None`` and ``num_registers > 0``,
    register tokens are extracted from the *normalised* input at positions
    ``[register_start_idx : register_start_idx + num_registers]``, pooled by
    ``register_pooling`` into a ``(B, C)`` conditioning vector, and passed to
    the sequence mixer as ``conditioning=<vector>``.  This lets the mixer
    (typically :class:`~nvsubquadratic.modules.vit5_attention.ViT5Attention` or
    :class:`~nvsubquadratic.modules.vit5_hyena_adapter.ViT5HyenaAdapter`) apply
    FiLM modulation to its internal kernel.

    Attributes:
        input_norm (torch.nn.Module): Pre-norm applied before the sequence
            mixer.  Parameters are tagged ``_no_weight_decay = True``.
        sequence_mixer (torch.nn.Module): Instantiated sequence-mixing
            operator — typically a
            :class:`~nvsubquadratic.modules.vit5_attention.ViT5Attention` or
            :class:`~nvsubquadratic.modules.vit5_hyena_adapter.ViT5HyenaAdapter`
            (both include their own QKV / output projections).
        mlp_norm (torch.nn.Module): Pre-norm applied before the MLP.
            Parameters are tagged ``_no_weight_decay = True``.
        mlp (torch.nn.Module): Position-wise MLP applied after ``mlp_norm``.
        ls_attn (LayerScale | torch.nn.Identity): Per-element learnable scale
            for the sequence mixer branch.  ``nn.Identity`` when
            ``layer_scale_init == 0``.
        ls_mlp (LayerScale | torch.nn.Identity): Per-element learnable scale
            for the MLP branch.  ``nn.Identity`` when ``layer_scale_init == 0``.
        drop_path (DropPath | torch.nn.Identity): Stochastic depth applied
            after both LayerScale modules.  ``nn.Identity`` when
            ``drop_path_rate == 0``.
        register_pooling (torch.nn.Module | None): Optional module that maps
            ``[B, num_registers, C]`` to a ``(B, C)`` conditioning vector.
            ``None`` when register-based conditioning is disabled.
        grn (torch.nn.Module | None): Optional Global Response Normalization
            (ConvNeXt V2) applied to the sequence mixer output before
            ``ls_attn``.  ``None`` when disabled.
        num_registers (int): Number of register tokens per sample.
        register_start_idx (int): Token index at which register tokens begin.
    """

    def __init__(
        self,
        sequence_mixer_cfg: LazyConfig,
        sequence_mixer_norm_cfg: LazyConfig,
        mlp_cfg: LazyConfig,
        mlp_norm_cfg: LazyConfig,
        hidden_dim: int,
        layer_scale_init: float = 1e-4,
        drop_path_rate: float = 0.0,
        register_pooling_cfg: LazyConfig | None = None,
        num_registers: int = 0,
        register_start_idx: int = 1,
        grn_cfg: LazyConfig | None = None,
    ):
        """Instantiate norms, sequence mixer, MLP, and optional register pooling.

        Args:
            sequence_mixer_cfg: LazyConfig for the sequence mixer.  Typical
                targets are
                :class:`~nvsubquadratic.modules.vit5_attention.ViT5Attention`
                or
                :class:`~nvsubquadratic.modules.vit5_hyena_adapter.ViT5HyenaAdapter`.
                Both include QKV and output projections internally (unlike the
                generic block where projections live in ``QKVSequenceMixer``).
            sequence_mixer_norm_cfg: LazyConfig for the pre-norm applied before
                the sequence mixer, e.g. ``RMSNorm(hidden_dim)``.
            mlp_cfg: LazyConfig for the position-wise MLP, e.g.
                :class:`~nvsubquadratic.modules.mlp.MLP`.
            mlp_norm_cfg: LazyConfig for the pre-norm applied before the MLP,
                e.g. ``RMSNorm(hidden_dim)``.
            hidden_dim: Channel dimension ``C`` shared by all sub-modules.
                Used to size :class:`~nvsubquadratic.modules.layer_scale.LayerScale`
                (one learnable scalar per channel).
            layer_scale_init: Initial value for both ``ls_attn`` and ``ls_mlp``
                LayerScale gammas.  Set to ``0`` to replace LayerScale with
                ``nn.Identity`` (disables per-channel learned scaling entirely).
                Typical values: ``1e-4`` (early training) or ``1.0`` (fine-tune
                from a strong checkpoint).
            drop_path_rate: Stochastic depth probability.  Set to ``0.0`` to
                replace ``DropPath`` with ``nn.Identity`` (no drop during
                training).  A single ``DropPath`` instance is shared between
                both branches.
            register_pooling_cfg: Optional LazyConfig for a register pooling
                module whose ``forward(regs)`` accepts ``[B, num_registers, C]``
                and returns ``(B, C)``.  When ``None`` or when
                ``num_registers == 0``, register conditioning is disabled and
                the sequence mixer is called without a ``conditioning`` kwarg.
            num_registers: Number of register tokens ``R`` in the sequence.
                Must be consistent with the token layout baked into
                ``sequence_mixer_cfg`` (e.g. ``ViT5Attention.num_registers``).
                Only used when ``register_pooling_cfg`` is not ``None``.
            register_start_idx: Zero-based token index at which the register
                block begins.  With the standard ViT-5 token layout
                ``[patches, CLS, registers]``, this equals
                ``num_patches_h * num_patches_w + 1`` for CLS-readout models
                and ``num_patches_h * num_patches_w`` for GAP-readout models.
                Typically injected by the network constructor.
            grn_cfg: Optional LazyConfig for a
                :class:`~nvsubquadratic.modules.grn.GlobalResponseNorm` module.
                When provided, GRN is applied to the sequence mixer output
                (``[B, T, C]``) before ``ls_attn``, promoting inter-channel
                feature competition (ConvNeXt V2 recipe).
        """
        super().__init__()

        self.input_norm = instantiate(sequence_mixer_norm_cfg)
        for param in self.input_norm.parameters():
            param._no_weight_decay = True

        self.sequence_mixer = instantiate(sequence_mixer_cfg)

        self.mlp_norm = instantiate(mlp_norm_cfg)
        for param in self.mlp_norm.parameters():
            param._no_weight_decay = True

        self.mlp = instantiate(mlp_cfg)

        self.ls_attn = LayerScale(hidden_dim, init_value=layer_scale_init) if layer_scale_init > 0 else nn.Identity()
        self.ls_mlp = LayerScale(hidden_dim, init_value=layer_scale_init) if layer_scale_init > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        # Optional register-based FiLM conditioning
        self.num_registers = num_registers
        self.register_start_idx = register_start_idx
        if register_pooling_cfg is not None and num_registers > 0:
            self.register_pooling = instantiate(register_pooling_cfg)
        else:
            self.register_pooling = None

        # Optional GRN (Global Response Normalization) after mixer
        self.grn = instantiate(grn_cfg) if grn_cfg is not None else None

    def flop_count(self, num_tokens: int, inference: bool = False) -> int:
        """Count FLOPs for one ViT-5 residual block.

        Counts MACs multiplied by 2 (multiply + add) for every sub-module that
        exposes a ``flop_count`` method, and falls back to 0 for modules that
        do not (e.g. GRN, when ``flop_count`` is absent).

        **Computation graph (one forward pass)**:

        .. code-block:: text

            input_norm        → flop_count(T)
            register_pooling  → flop_count(D)        [when enabled]
            sequence_mixer    → flop_count(T, inf)
            grn               → flop_count(T)        [when enabled; 0 if absent]
            ls_attn           → flop_count(T)        [when LayerScale, not Identity]
            mlp_norm          → flop_count(T)
            mlp               → flop_count(T)
            ls_mlp            → flop_count(T)        [when LayerScale, not Identity]

        ``drop_path`` contributes 0 FLOPs (stochastic identity — no arithmetic
        on active samples beyond the residual add, which is counted separately
        by the caller if desired).

        Args:
            num_tokens: Sequence length ``T`` passed to each sub-module.  This
                should equal ``num_patches + (1 if has_cls else 0) + num_registers``.
            inference: Passed through to ``self.sequence_mixer.flop_count``.
                Some sequence mixers (e.g. Hyena with precomputed kernels) have
                lower inference FLOPs than training FLOPs.

        Returns:
            Total FLOPs as an integer.  Does not include the two residual adds
            (``2 * T * C`` FLOPs each), which are typically negligible.
        """
        flops = 0

        # input_norm
        flops += self.input_norm.flop_count(num_tokens)

        # Register pooling (FiLM conditioning)
        if self.register_pooling is not None:
            D = self.input_norm.weight.shape[0]
            flops += self.register_pooling.flop_count(D)

        # Sequence mixer
        flops += self.sequence_mixer.flop_count(num_tokens, inference=inference)

        # GRN
        if self.grn is not None:
            # We assume GlobalResponseNorm implements a flop_count(num_tokens) property/method if present
            flops += getattr(self.grn, "flop_count", lambda t: 0)(num_tokens)

        # LayerScale (attention branch)
        if isinstance(self.ls_attn, LayerScale):
            flops += self.ls_attn.flop_count(num_tokens)

        # mlp_norm
        flops += self.mlp_norm.flop_count(num_tokens)

        # MLP
        flops += self.mlp.flop_count(num_tokens)

        # LayerScale (MLP branch)
        if isinstance(self.ls_mlp, LayerScale):
            flops += self.ls_mlp.flop_count(num_tokens)

        return flops

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """Apply the ViT-5 residual block.

        Executes two residual branches in sequence:

        1. **Sequence mixer branch** — normalise, optionally extract a register
           conditioning vector, run the sequence mixer (and optional GRN), scale
           with LayerScale, apply stochastic depth, add residual.
        2. **MLP branch** — normalise, run MLP, scale with LayerScale, apply
           stochastic depth, add residual.

        Register conditioning detail: when ``self.register_pooling`` is not
        ``None``, the slice ``x_normed[:, register_start_idx : register_start_idx
        + num_registers, :]`` is extracted from the *normalised* input and
        pooled to shape ``(B, C)``.  This vector is forwarded to the sequence
        mixer as ``conditioning=<vector>``, which the mixer uses for FiLM
        modulation (e.g. scaling SIREN kernel features).

        Args:
            x: Input token sequence of shape ``[B, T, C]``.  ``B`` is the
                batch size and ``C`` is the channel (hidden) dimension.
                ``T = num_patches + (1 if has_cls else 0) + num_registers
                (+ pad_size for Hyena blocks)`` is the total token count
                following the ViT-5 layout
                ``[patches, (CLS,) registers, (padding,)]``.  Attention
                blocks receive the unpadded sequence; Hyena blocks receive
                the zero-padded sequence so ``T % grid_w == 0``.
            condition: Accepted for API compatibility with
                :class:`~nvsubquadratic.modules.residual_block.ResidualBlock`
                but **always ignored** in this class.  ViT-5 conditioning is
                routed through register pooling, not through this argument.
                Pass ``None`` (the default) when calling directly.

        Returns:
            torch.Tensor: Output tensor of shape ``[B, T, C]``, the same shape
            as ``x``.
        """
        x_normed = self.input_norm(x)

        mixer_kwargs = {}
        if self.register_pooling is not None:
            s = self.register_start_idx
            regs = x_normed[:, s : s + self.num_registers, :]  # [B, num_registers, C]
            mixer_kwargs["conditioning"] = self.register_pooling(regs)  # [B, C]

        mixer_out = self.sequence_mixer(x_normed, **mixer_kwargs)
        if self.grn is not None:
            mixer_out = self.grn(mixer_out)
        x = x + self.drop_path(self.ls_attn(mixer_out))
        x = x + self.drop_path(self.ls_mlp(self.mlp(self.mlp_norm(x))))
        return x
