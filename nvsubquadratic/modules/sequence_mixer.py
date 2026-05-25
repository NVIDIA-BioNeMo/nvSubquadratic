# TODO: Add license header here


"""Sequence mixer abstraction layer for ND signals.

This module provides :class:`QKVSequenceMixer`, the **operator-agnostic dispatch
layer** that sits between the residual block and any concrete sequence-mixing
kernel (Hyena, attention, CKConv, Mamba, etc.).

Architecture role
-----------------
Every residual block in the network needs a *sequence mixer* — a module that
lets tokens (or spatial positions) exchange information over long ranges.
Rather than hard-coding a specific operator into each block, the network passes
a ``mixer_cfg`` :class:`~nvsubquadratic.lazy_config.LazyConfig` down to
:class:`QKVSequenceMixer`, which instantiates the concrete mixer and wraps it
with shared QKV input/output projections.  This means the rest of the network
(residual blocks, classifiers, diffusion heads) is **entirely agnostic** to
which sequence mixer is in use; swapping Hyena for attention, or attention for
CKConv, requires only a config change.

Dispatch pattern
----------------
The dispatch is performed by :func:`~nvsubquadratic.lazy_config.instantiate`
acting on ``mixer_cfg``.  Any class whose ``forward`` method accepts
``(q, k, v, cp_group, **kwargs)`` can be used as the inner mixer.  Note that
``cp_group`` is passed **positionally** as the fourth argument; an inner mixer
that captures it only via ``**kwargs`` would silently not receive it.

The currently supported inner mixers are:

* :class:`~nvsubquadratic.modules.hyena_nd.Hyena` — gated global-conv mixer
  (subquadratic in sequence length).
* :class:`~nvsubquadratic.modules.attention.Attention` — multi-head
  self-attention (quadratic in sequence length, but faster for short sequences
  and easy to compose with RoPE).
* :class:`~nvsubquadratic.modules.ckconv_nd.CKConvND` — continuous-kernel conv
  (any spatial rank, learned kernel parametrisation via an MLP).
* :class:`~nvsubquadratic.modules.mamba_nd.Mamba` — Mamba SSM variant for
  ND inputs.

Note:
    To add a new mixer type, implement a :class:`torch.nn.Module` whose
    ``forward(q, k, v, cp_group, **kwargs)`` method follows the channels-last
    convention ``[B, *spatial, C]`` and, optionally, implement
    ``flop_count(spatial_dims, inference) -> int``.  Then pass its
    :class:`~nvsubquadratic.lazy_config.LazyConfig` as ``mixer_cfg`` to
    :class:`QKVSequenceMixer` — no other changes are needed.

Input / output layout
---------------------
All tensors flowing through this module use **channels-last** layout::

    x: [B, *spatial, C]

where ``B`` is batch size, ``spatial`` is one or more spatial axes (e.g.
``(T,)`` for sequences, ``(H, W)`` for images, ``(D, H, W)`` for volumes),
and ``C = hidden_dim``.
"""

from typing import Callable

import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class QKVSequenceMixer(torch.nn.Module):
    """Operator-agnostic sequence mixer with shared QKV and output projections.

    :class:`QKVSequenceMixer` mirrors the structure of
    :class:`~nvsubquadratic.modules.attention.Attention` (and ViT5Attention)
    so that any inner mixer — Hyena, attention, CKConv, Mamba — can be dropped
    in without changing the surrounding residual block.

    The forward pass is:

    .. code-block:: text

        x  ─[Linear(C → 3C, + bias?)]──► split ──► Q, K, V
                                                         │
                                 inner_mixer(Q, K, V, cp_group, **kwargs)
                                                         │
                                 [Linear(C → C, + bias?)]──► y

    The QKV projection packs all three projections into a single
    ``Linear(C, 3·C)`` call for efficiency; the output projection maps back to
    ``C``.  Both projections optionally include a bias term (disabled by
    default; see ``qkv_bias`` and ``out_proj_bias``).

    Attributes:
        mixer (torch.nn.Module): The instantiated inner sequence-mixing
            operator (e.g. :class:`~nvsubquadratic.modules.hyena_nd.Hyena`).
        qkv_proj (torch.nn.Linear): Combined Q+K+V input projection;
            maps ``C`` → ``3·C`` (weight shape ``(3C, C)``).
        out_proj (torch.nn.Linear): Output projection; maps ``C`` → ``C``
            (weight shape ``(C, C)``).

    Example::

        from nvsubquadratic.lazy_config import LazyConfig
        from nvsubquadratic.modules.hyena_nd import Hyena

        mixer_cfg = LazyConfig(Hyena)(
            global_conv_cfg=...,
            short_conv_cfg=...,
            gate_nonlinear_cfg=...,
            pixelhyena_norm_cfg=...,
            qk_norm_cfg=None,
        )
        block = QKVSequenceMixer(hidden_dim=256, mixer_cfg=mixer_cfg)

        x = torch.randn(2, 32, 32, 256)   # [B, H, W, C]
        y = block(x)                       # [B, H, W, C]
    """

    def __init__(
        self,
        hidden_dim: int,
        mixer_cfg: LazyConfig,
        qkv_bias: bool = False,
        out_proj_bias: bool = False,
        init_method_in: Callable[[int], Callable[[torch.Tensor], torch.Tensor]] | None = None,
        init_method_out: Callable[[int], Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ):
        """Initialise the QKV sequence mixer.

        Args:
            hidden_dim: Channel dimension ``C`` of the input / output tensor.
                Both ``qkv_proj`` and ``out_proj`` are sized using this value.
            mixer_cfg: :class:`~nvsubquadratic.lazy_config.LazyConfig` for the
                inner sequence-mixing operator.  The target class's ``forward``
                method must accept ``(q, k, v, cp_group, **kwargs)`` where
                ``cp_group`` is the fourth positional argument.  Supported
                targets include
                :class:`~nvsubquadratic.modules.hyena_nd.Hyena`,
                :class:`~nvsubquadratic.modules.attention.Attention`,
                :class:`~nvsubquadratic.modules.ckconv_nd.CKConvND`, and
                :class:`~nvsubquadratic.modules.mamba_nd.Mamba`.
            qkv_bias: If ``True``, adds a learnable bias to the combined QKV
                projection.  The bias is zero-initialised when
                ``init_method_in`` is provided.  Defaults to ``False``.
            out_proj_bias: If ``True``, adds a learnable bias to the output
                projection.  Zero-initialised when ``init_method_out`` is
                provided.  Defaults to ``False``.
            init_method_in: Optional *curried* weight initialiser for
                ``qkv_proj``.  Must have the signature
                ``fn(dim: int) -> fn(tensor: Tensor) -> None``.  When provided,
                ``fn(hidden_dim)`` is called and the returned callable is applied
                to ``qkv_proj.weight.data``.  If ``qkv_bias`` is also ``True``,
                the bias is zero-initialised.  Pass ``None`` to use PyTorch's
                default (Kaiming uniform).
            init_method_out: Same as ``init_method_in`` but applied to
                ``out_proj.weight.data``.  Typically a scaled initialiser that
                controls residual-branch variance (GPT/Megatron style), e.g.::

                    import math
                    init_method_out = (
                        lambda dim: lambda w: torch.nn.init.normal_(
                            w, std=1 / math.sqrt(num_layers)
                        )
                    )

        Raises:
            Exception: Propagated from
                :func:`~nvsubquadratic.lazy_config.instantiate` if the target
                class cannot be constructed (e.g. missing required arguments or
                an invalid ``mixer_cfg``).  The exact exception type depends on
                the ``LazyConfig`` backend (typically an
                ``omegaconf.errors.InstantiationException`` or similar).
                Check ``mixer_cfg._target_`` and its keyword arguments if this
                is raised.
        """
        super().__init__()

        self.mixer = instantiate(mixer_cfg)

        self.qkv_proj = torch.nn.Linear(hidden_dim, 3 * hidden_dim, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=out_proj_bias)

        if init_method_in is not None:
            init_method_in(hidden_dim)(self.qkv_proj.weight.data)
            if qkv_bias:
                torch.nn.init.zeros_(self.qkv_proj.bias)
        if init_method_out is not None:
            init_method_out(hidden_dim)(self.out_proj.weight.data)
            if out_proj_bias:
                torch.nn.init.zeros_(self.out_proj.bias)

    def flop_count(self, spatial_dims: tuple[int, ...], inference: bool = False) -> int:
        """Count FLOPs for QKV projections + inner mixer + output projection.

        Uses the standard multiply-accumulate convention where one FLOP = one
        multiply + one add (i.e. the matrix-vector product ``y = Wx`` over
        ``T`` tokens costs ``2 · T · in_dim · out_dim`` FLOPs).  Bias
        additions are excluded, following the standard ML FLOP-counting
        convention.

        FLOPs breakdown (``D`` = ``hidden_dim``, ``T`` = ``prod(spatial_dims)``):

        1. **QKV projection** ``Linear(D, 3D)``:
           ``2 · T · D · 3D = 6 · T · D²``
        2. **Inner mixer** (e.g. Hyena, attention):
           Delegated to ``self.mixer.flop_count(spatial_dims, inference)``.
           For Hyena this is dominated by the FFT convolution
           ``O(T log T · D)``; for attention it is ``O(T² · D)``.
        3. **Output projection** ``Linear(D, D)``:
           ``2 · T · D²``

        Total (excluding inner mixer): ``8 · T · D²``.

        Args:
            spatial_dims: Spatial extents of the input signal, e.g. ``(H, W)``
                for images or ``(T,)`` for 1D sequences.  Linear projections
                treat the flattened token count ``T = prod(spatial_dims)`` as
                the sequence length.
            inference: Forwarded to ``self.mixer.flop_count``.  Some mixers
                (e.g. autoregressive Mamba) have different inference-time costs.

        Returns:
            Total FLOPs as a non-negative integer.

        Raises:
            AttributeError: If the inner mixer does not implement
                ``flop_count``.
        """
        D = self.qkv_proj.in_features
        T = 1
        for s in spatial_dims:
            T *= s

        flops = 0
        # QKV projection
        flops += 2 * T * D * self.qkv_proj.out_features
        # Inner mixer
        flops += self.mixer.flop_count(spatial_dims, inference=inference)
        # Output projection
        flops += 2 * T * self.out_proj.in_features * self.out_proj.out_features
        return flops

    def forward(
        self,
        x: torch.Tensor,
        cp_group: torch.distributed.ProcessGroup | None = None,
        **mixer_kwargs,
    ) -> torch.Tensor:
        """Run the QKV-project → mix → output-project forward pass.

        Args:
            x: Input tensor of shape ``(B, *spatial, C)`` where ``B`` is batch
                size, ``spatial`` is one or more spatial axes (e.g. ``(T,)``
                for 1D, ``(H, W)`` for 2D, ``(D, H, W)`` for 3D), and
                ``C = hidden_dim``.
            cp_group: Optional context-parallel process group
                (:class:`torch.distributed.ProcessGroup`).  When provided, the
                input is assumed to be already split across ranks along the
                spatial axis, and the inner mixer is responsible for the
                cross-rank communication (e.g. AllToAll for Hyena,
                ring-attention for :class:`~nvsubquadratic.modules.attention.Attention`).
                Pass ``None`` (default) for single-GPU / non-distributed runs.
            **mixer_kwargs: Additional keyword arguments forwarded verbatim to
                ``self.mixer.forward``.  Mixers that do not recognise a key
                must accept and ignore it via their own ``**kwargs``.  Common
                keys:

                * ``conditioning`` (:class:`torch.Tensor`, shape
                  ``(B, cond_dim)``): FiLM conditioning vector consumed by
                  :class:`~nvsubquadratic.modules.hyena_nd.Hyena` when a
                  ``condition_mixer`` is attached.  Ignored by mixers that do
                  not have a ``condition_mixer`` (it passes through
                  ``**mixer_kwargs`` and is discarded).

        Returns:
            Output tensor of shape ``(B, *spatial, C)`` — same layout as the
            input.
        """
        # Q, K, V projections via single linear
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Sequence mixer (e.g., self-attention, hyena, etc.)
        x = self.mixer(q, k, v, cp_group, **mixer_kwargs)
        # Output projection
        x = self.out_proj(x)
        return x
