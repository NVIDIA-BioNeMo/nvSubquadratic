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

r"""Hyena-ND: gated global convolutional mixer for 1D/2D/3D signals.

Background
----------
The Hyena operator (Poli et al., "Hyena Hierarchy: Towards Larger Convolutional
Language Models", ICML 2023, arXiv:2302.10866) replaces the quadratic attention
map with a
**subquadratic gated convolution**: two multiplicative gates sandwich a
long-range (global-kernel) depthwise convolution whose kernel is generated
implicitly by a small neural network (see ``kernels_nd.py``).  The operator
achieves O(N log N) time complexity in sequence length N rather than the O(N²)
of dense attention.

This module generalises the original 1D Hyena to **arbitrary spatial rank**
(1D sequences, 2D images, 3D volumes) by routing the global convolution
through ``CKConvND`` / ``fftconv{1,2,3}d`` primitives from
``nvsubquadratic.ops.fftconv``.

Computation graph (per forward call)
-------------------------------------
Given projections Q, K, V ∈ R^{B × C × *spatial} (channels-first internally):

.. code-block:: none

    short_conv([Q; K; V])         — optional depthwise short conv on concat QKV
         │
    QK-Norm(Q [, K])              — optional per-channel normalisation
         │                          (K is normalised only when gate_nonlinear = Identity)
    z = Q ⊙ σ(K)                  — first multiplicative gate
         │
    PixelHyena-Norm(z)            — optional normalisation (GroupNorm / RMSNorm / …)
         │
    h = GlobalConv(z)             — long-range FFT convolution via CKConvND
         │
    y = h ⊙ σ₂(V)                 — second multiplicative gate
         │
    Output-Norm(y)                — optional normalisation before projection
         │
    return y                      — [B, *spatial, C]  (channels-last on exit)

σ denotes ``gate_nonlinear`` (first gate) and σ₂ denotes ``gate_nonlinear_2``
(second gate).  By default σ₂ = σ.  When both are ``Identity`` the gates
reduce to plain element-wise products, recovering a linear variant closer
to Mamba's selective-scan formulation.  Setting σ = SiLU, σ₂ = Sigmoid follows
the gated attention formulation from Hyena.

ND generalisation
-----------------
The Hyena paper targets 1D autoregressive sequences with causal convolutions.
This implementation extends the design to spatial data:

* The short conv is a standard ``torch.nn.Conv{1,2,3}d`` (or a distributed
  equivalent) with a small kernel (e.g. 3×3 for images).
* The global conv is ``CKConvND``, which generates its kernel with a Random
  Fourier Feature MLP (``kernels_nd.py``) and convolves it via
  ``fftconv{1,2,3}d`` from ``nvsubquadratic.ops.fftconv``.  For 2D/3D signals
  the convolution is non-causal by default; causal 1D mode is preserved.
  By default the 2D/3D path uses zero-padded (linear) FFT convolution
  (``fftconv2d`` / ``fftconv3d``), matching ``torch.nn.ConvNd(padding='same')``
  semantics.  Set the ``circular`` flag on ``CKConvND`` to switch to periodic
  boundary conditions, or use ``mixed_fftconv`` for per-axis mixed BCs
  (see ``nvsubquadratic.ops.mixed_fftconv``).

Context parallelism
-------------------
When ``cp_group`` is supplied in ``forward``, the module uses
``AllToAllSingleFunction`` to shard along ``dim=2`` (the first spatial axis of
the channels-first ``[B, C, *spatial]`` tensor) while gathering the channel dim.
For 2D inputs ``[B, C, H, W]`` this means row-wise sharding (across H).  The
short conv is applied globally after the gather, and the result is sharded back.
The global conv receives only the local spatial slice (it must be
context-parallel-aware itself).

Related modules
---------------
* ``nvsubquadratic.modules.kernels_nd`` — implicit kernel parametrisation
  (``CKConvKernelND``, ``RandomFourierPositionalEmbeddingND``)
* ``nvsubquadratic.ops.fftconv`` — FFT convolution primitives consumed by the
  global conv
* ``nvsubquadratic.modules.ckconv_nd`` — ``CKConvND``, the usual choice for
  ``global_conv_cfg``
"""

from typing import Optional

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules._channels_first_utils import is_channels_first_norm
from nvsubquadratic.modules.distributed_depthwise_conv_nd import (
    DistributedDepthwiseConv1d,
    DistributedDepthwiseConv2d,
    DistributedDepthwiseConv3d,
)
from nvsubquadratic.parallel.a2a_comms import AllToAllSingleFunction


class Hyena(torch.nn.Module):
    r"""Gated global convolutional mixer for ND signals.

    The Hyena operator computes the following gated convolution (all tensors
    channels-first internally, channels-last on the public interface):

    .. math::

        z   &= Q \odot \sigma(K)                \\
        h   &= \mathrm{GlobalConv}(z)           \\
        y   &= h \odot \sigma_2(V)

    where :math:`\sigma` is ``gate_nonlinear``, :math:`\sigma_2` is
    ``gate_nonlinear_2`` (defaults to :math:`\sigma`), and
    :math:`\mathrm{GlobalConv}` is a depthwise FFT convolution whose kernel is
    generated on-the-fly by an implicit MLP (``CKConvND``).

    Setting both gates to ``Identity`` gives a **linear** gating variant
    (element-wise products only).  Setting :math:`\sigma = \mathrm{SiLU}` and
    :math:`\sigma_2 = \mathrm{Sigmoid}` matches the gated attention formulation
    used in the original Hyena paper.

    Paper references
    ----------------
    The two-gate structure follows the H3 block (Fu et al., "Hungry Hungry
    Hippos", ICLR 2023, arXiv:2212.14052, Section 3.2) and is generalised in
    Hyena (Poli et al., "Hyena Hierarchy: Towards Larger Convolutional Language
    Models", ICML 2023, arXiv:2302.10866, Section 3 "The Hyena Recurrence").
    The ND extension replaces the causal 1D FFT conv with a non-causal ND FFT
    conv (``CKConvND``).

    Optional components (each disabled by passing ``Identity`` or ``None``):
        - Short depthwise convolution on concatenated ``[Q, K, V]``
        - QK normalisation (Q always; K only when :math:`\sigma = \mathrm{Identity}`)
        - PixelHyena normalisation between first gate and global conv
        - Output normalisation after second gate
        - Context parallelism via AllToAll communication (``cp_group`` argument)

    Example::

        # Minimal 2D Hyena block (non-causal, no normalisation).
        # In practice global_conv_cfg wraps a fully-configured CKConvND.
        import torch
        from nvsubquadratic.lazy_config import LazyConfig
        from nvsubquadratic.modules.hyena_nd import Hyena

        hyena = Hyena(
            global_conv_cfg=...,          # LazyConfig wrapping CKConvND
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                192, 192, 3, padding=1, groups=192
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(torch.nn.Identity)(),
            qk_norm_cfg=None,
        )
        B, H, W, C = 2, 16, 16, 64
        q = k = v = torch.randn(B, H, W, C)
        y = hyena(q, k, v)  # [2, 16, 16, 64]

    Attributes:
        global_conv (torch.nn.Module): Long-range global convolution, typically
            ``CKConvND``.  Must expose ``hidden_dim`` and
            ``flop_count(spatial_dims, inference)`` for FLOP counting.
        short_conv (torch.nn.Module): Short depthwise convolution applied to
            the concatenated ``[Q, K, V]`` tensor (3·C input channels).
            Must be one of ``torch.nn.Conv{1,2,3}d``,
            ``DistributedDepthwiseConv{1,2,3}d``, or ``torch.nn.Identity``.
        gate_nonlinear (torch.nn.Module): Activation :math:`\sigma` for the
            first gate.  Applied to K before multiplying with Q.
        gate_nonlinear_2 (torch.nn.Module): Activation :math:`\sigma_2` for
            the second gate.  Applied to V before multiplying with h.
            Shares the same object as ``gate_nonlinear`` when
            ``gate_nonlinear_2_cfg`` is ``None``.
        pixelhyena_norm (torch.nn.Module): Normalisation layer applied to
            ``z = Q ⊙ σ(K)`` before the global conv.  Parameters are
            excluded from weight-decay via ``_no_weight_decay = True``.
        output_norm (torch.nn.Module): Normalisation layer applied to
            ``y = h ⊙ σ₂(V)`` before returning.  Parameters are excluded
            from weight-decay.
        q_norm (torch.nn.Module | None): Per-channel normalisation for Q.
            ``None`` when ``qk_norm_cfg`` is ``None``.
        k_norm (torch.nn.Module | None): Per-channel normalisation for K.
            ``None`` when ``qk_norm_cfg`` is ``None`` (QK-norm entirely
            disabled).  ``torch.nn.Identity`` when the gate is nonlinear
            (:math:`\sigma` already bounds K's magnitude); a fresh instance of
            ``qk_norm_cfg`` when the gate is ``Identity`` (linear gating).
    """

    def __init__(
        self,
        global_conv_cfg: LazyConfig,
        short_conv_cfg: LazyConfig,
        gate_nonlinear_cfg: LazyConfig,
        pixelhyena_norm_cfg: LazyConfig,
        qk_norm_cfg: Optional[LazyConfig] | None,
        output_norm_cfg: LazyConfig = LazyConfig(torch.nn.Identity)(),
        gate_nonlinear_2_cfg: Optional[LazyConfig] = None,
    ):
        r"""Construct a Hyena gated global convolutional mixer.

        All ``*_cfg`` arguments are ``LazyConfig`` objects that are
        instantiated inside ``__init__`` via ``nvsubquadratic.lazy_config.instantiate``.
        This pattern allows full Python configurability without importing
        module classes at config-definition time.

        Args:
            global_conv_cfg: ``LazyConfig`` for the long-range global
                convolution (e.g. ``CKConvND``).  The instantiated module must
                expose ``hidden_dim: int`` and
                ``flop_count(spatial_dims, inference) -> int``.
            short_conv_cfg: ``LazyConfig`` for the short depthwise conv applied
                to the concatenated ``[Q; K; V]`` tensor (3·C input channels).
                Must instantiate to one of ``torch.nn.Conv{1,2,3}d``,
                ``DistributedDepthwiseConv{1,2,3}d``, or ``torch.nn.Identity``.
                Use ``Identity`` to skip the short conv entirely.
            gate_nonlinear_cfg: ``LazyConfig`` for the first-gate activation
                :math:`\sigma(K)` (e.g. ``SiLU``).  Use ``Identity`` for
                linear gating.
            pixelhyena_norm_cfg: ``LazyConfig`` for the normalisation applied
                between the first gate and the global conv.  Use ``Identity``
                to disable.  Parameters receive ``_no_weight_decay = True``.
            qk_norm_cfg: ``LazyConfig`` for per-channel normalisation of Q (and
                K when the gate is ``Identity``).  Pass ``None`` to disable
                QK-norm entirely.  Two separate instances are created (one for
                Q, one for K) so that stateful norms (e.g. ``RMSNorm`` with a
                learnable scale) keep independent parameters.
            output_norm_cfg: ``LazyConfig`` for the normalisation applied after
                the second gate.  Defaults to a ``LazyConfig`` wrapping
                ``torch.nn.Identity`` (no normalisation).  Do **not** pass an
                already-instantiated module — pass a ``LazyConfig`` object that
                wraps the class.  Parameters receive ``_no_weight_decay = True``.
            gate_nonlinear_2_cfg: ``LazyConfig`` for the second-gate activation
                :math:`\sigma_2(V)`.  If ``None`` (default), both gates share
                the same activation object (``self.gate_nonlinear``).

        Raises:
            AssertionError: If the instantiated ``short_conv`` is not one of
                the supported Conv / DistributedDepthwiseConv / Identity types.
        """
        super().__init__()

        # Core global convs: feature and gate branches
        self.global_conv = instantiate(global_conv_cfg)
        self.short_conv = instantiate(short_conv_cfg)
        assert isinstance(
            self.short_conv,
            (
                torch.nn.Conv1d,
                torch.nn.Conv2d,
                torch.nn.Conv3d,
                torch.nn.Identity,
                DistributedDepthwiseConv1d,
                DistributedDepthwiseConv2d,
                DistributedDepthwiseConv3d,
            ),
        ), (
            f"Short conv must be an instance of torch.nn.ConvNd (1d, 2d, or 3d) or torch.nn.Identity. Current type: {type(self.short_conv)}"
        )

        # Nonlinear gates
        self.gate_nonlinear = instantiate(gate_nonlinear_cfg)
        if gate_nonlinear_2_cfg is not None:
            self.gate_nonlinear_2 = instantiate(gate_nonlinear_2_cfg)
        else:
            self.gate_nonlinear_2 = self.gate_nonlinear

        # Pixelhyena normalization (use torch.nn.Identity for no normalization)
        self.pixelhyena_norm = instantiate(pixelhyena_norm_cfg)
        # Exclude self.pixelhyena_norm from the parameter group with weight decay
        for param in self.pixelhyena_norm.parameters():
            param._no_weight_decay = True

        # Optional value normalization (use torch.nn.Identity for no normalization)
        self.output_norm = instantiate(output_norm_cfg)
        for param in self.output_norm.parameters():
            param._no_weight_decay = True

        # QK Normalization (separate instances for Q and K to support stateful norms like RMSNorm).
        # K-norm is only useful when gating is linear (Identity); a nonlinear gate
        # (e.g. SiLU) already bounds K's magnitude, so we use Identity for k_norm.
        if qk_norm_cfg is not None:
            self.q_norm = instantiate(qk_norm_cfg)
            if isinstance(self.gate_nonlinear, torch.nn.Identity):
                self.k_norm = instantiate(qk_norm_cfg)
            else:
                self.k_norm = torch.nn.Identity()
        else:
            self.q_norm = None
            self.k_norm = None

    def extra_repr(self) -> str:
        """Return a compact summary of key configuration choices.

        Included fields:
            - ``q_norm`` / ``k_norm`` class names (or ``"None"``).
              When QK-norm is disabled both are ``None``; the strings
              ``"q_norm=None"`` and ``"k_norm=None"`` are still emitted so
              the disabled state is explicit in ``repr(module)``.
            - ``gates=<σ>/<σ₂>`` when the two gate activations differ.
            - ``is_causal`` when the global conv exposes that attribute.

        Returns:
            Comma-separated string suitable for ``repr(module)`` output.
        """
        is_causal = getattr(self.global_conv, "is_causal", None)
        q_norm_str = self.q_norm.__class__.__name__ if self.q_norm is not None else "None"
        k_norm_str = self.k_norm.__class__.__name__ if self.k_norm is not None else "None"
        parts = [f"q_norm={q_norm_str}", f"k_norm={k_norm_str}"]
        if self.gate_nonlinear is not self.gate_nonlinear_2:
            g1 = self.gate_nonlinear.__class__.__name__
            g2 = self.gate_nonlinear_2.__class__.__name__
            parts.append(f"gates={g1}/{g2}")
        if is_causal is not None:
            parts.append(f"is_causal={is_causal}")
        return ", ".join(parts)

    def flop_count(self, spatial_dims: tuple[int, ...], inference: bool = False) -> int:
        r"""Count FLOPs for one forward pass of the Hyena mixer.

        Let ``C = self.global_conv.hidden_dim`` (the per-head channel count)
        and ``S = prod(spatial_dims)`` (total number of spatial positions).
        All counts use the **multiply-add = 1 FLOP** convention (i.e. a MAC
        counts as 1).

        FLOP breakdown:

        1. **Short depthwise conv** on concatenated ``[Q; K; V]``
           (``3·C`` input channels):

           .. math::

               2 \cdot \frac{in\_ch}{groups} \cdot out\_ch \cdot S \cdot k\_prod

           where :math:`k\_prod = \prod_d kernel\_size_d`.  Skipped when
           ``short_conv`` is ``Identity``.  For a pure depthwise conv
           (``groups == in_ch == out_ch``) this simplifies to
           ``2 · out_ch · S · k_prod``; the grouped formula is written here to
           handle partially-grouped convolutions (e.g.
           ``DistributedDepthwiseConvNd``).

        2. **QK-Norm** (when ``self.q_norm is not None``):
           ``3·C·S`` for Q; additional ``3·C·S`` for K only when
           ``gate_nonlinear`` is ``Identity`` (linear gating).
           The factor of 3 assumes an RMSNorm-like norm (sum-of-squares +
           rsqrt + elementwise scale).  Other norm types will differ; this is
           an approximation.

        3. **First gate** :math:`z = Q \odot \sigma(K)`:
           ``C·S`` for the elementwise multiply, plus ``C·S`` for the
           activation :math:`\sigma` when it is not ``Identity``.

        4. **PixelHyena norm** (when not ``Identity``): ``3·C·S``.

        5. **Global convolution**: delegated to
           ``self.global_conv.flop_count(spatial_dims, inference)``.

        6. **Second gate** :math:`y = h \odot \sigma_2(V)`:
           ``C·S`` for the multiply, plus ``C·S`` for :math:`\sigma_2` when
           not ``Identity``.

        7. **Output norm** (when not ``Identity``): ``3·C·S``.

        Args:
            spatial_dims: Spatial extent of the input per axis, e.g. ``(H, W)``
                for a 2-D feature map of shape ``[B, C, H, W]``.
            inference: Forwarded to ``self.global_conv.flop_count``; some
                implementations skip re-generating the kernel at inference time
                when it is cached.

        Returns:
            Total FLOP count as an integer (multiply-add = 1 FLOP convention).
        """
        C = self.global_conv.hidden_dim
        S = 1
        for s in spatial_dims:
            S *= s

        flops = 0

        # 1. Short depthwise conv
        if not isinstance(self.short_conv, torch.nn.Identity):
            k_prod = 1
            for k in self.short_conv.kernel_size:
                k_prod *= k
            in_ch = self.short_conv.in_channels  # 3 * C
            groups = self.short_conv.groups
            out_ch = self.short_conv.out_channels
            flops += 2 * (in_ch // groups) * out_ch * S * k_prod

        # 2. QK-Norm (k_norm is Identity when gate is non-linear, so no extra FLOPs)
        if self.q_norm is not None:
            flops += 3 * C * S  # Q norm
            if not isinstance(self.k_norm, torch.nn.Identity):
                flops += 3 * C * S  # K norm (only for linear gating)

        # 3. First gate: Q * σ(K)
        flops += C * S  # elementwise multiply
        if not isinstance(self.gate_nonlinear, torch.nn.Identity):
            flops += C * S  # activation on K

        # 4. PixelHyena norm
        if not isinstance(self.pixelhyena_norm, torch.nn.Identity):
            flops += 3 * C * S

        # 5. Global convolution
        flops += self.global_conv.flop_count(spatial_dims, inference=inference)

        # 6. Second gate: h * σ₂(V)
        flops += C * S
        if not isinstance(self.gate_nonlinear_2, torch.nn.Identity):
            flops += C * S  # activation on V

        # 7. Output norm
        if not isinstance(self.output_norm, torch.nn.Identity):
            flops += 3 * C * S

        return flops

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cp_group: torch.distributed.ProcessGroup = None,
        **mixer_kwargs,
    ) -> torch.Tensor:
        r"""Compute the Hyena gated global convolution.

        Implements:

        .. math::

            y = \mathrm{OutputNorm}\!\bigl(
                \mathrm{GlobalConv}\!\bigl(
                    \mathrm{Norm}(Q \odot \sigma(K))
                \bigr) \odot \sigma_2(V)
            \bigr)

        Tensors enter and leave in **channels-last** layout
        ``[B, *spatial, C]``.  Internally the module works channels-first
        ``[B, C, *spatial]`` for the short conv and global conv.

        Context parallelism (``cp_group``)
        -----------------------------------
        When ``cp_group`` is provided and has size > 1, the method applies two
        AllToAll communications around the short conv so that each device sees
        the full spatial extent during the convolution:

        1. Before short conv: ``split_to_full`` — gather spatial shards along
           ``dim=2`` (the first spatial axis), split along ``dim=1`` (channels).
        2. After short conv: ``full_to_split`` — scatter spatial, gather
           channels back.

        After step 1, each device holds the full spatial extent but only
        ``C / cp_size`` channels.  After step 2, the original ``C`` channels
        are restored and each device holds ``spatial_0 / cp_size`` positions
        along the first spatial axis.  The global conv receives only the local
        spatial slice and is expected to handle its own CP communication
        internally.

        Implementation note
        -------------------
        The ``query`` tensor is overwritten after the first gate to hold the
        gated intermediate ``z = Q ⊙ σ(K)``; the original Q tensor is no
        longer accessible after that point.  This is intentional to avoid an
        extra allocation.

        Args:
            query: ``[B, *spatial, C]`` — query tensor, typically the output
                of a linear projection ``W_Q · x``.
            key: ``[B, *spatial, C]`` — key tensor, typically ``W_K · x``.
            value: ``[B, *spatial, C]`` — value tensor, typically ``W_V · x``.
            cp_group: ``torch.distributed.ProcessGroup`` for context
                parallelism.  ``None`` disables CP (the default for single-GPU
                runs).
            **mixer_kwargs: Extra keyword arguments forwarded verbatim to
                ``self.global_conv`` (e.g. ``conditioning`` for FiLM-conditioned
                ``CKConvND``).

        Returns:
            ``[B, *spatial, C]`` — output tensor in channels-last layout,
            same shape as the inputs.
        """
        # Reshape query, key, and value to [B, C, * spatial_dims] (Required for short convolutional projections).
        query = rearrange(query, "b ... c -> b c ...")
        key = rearrange(key, "b ... c -> b c ...")
        value = rearrange(value, "b ... c -> b c ...")

        # Apply short convolutional projection
        if not isinstance(self.short_conv, torch.nn.Identity):
            # Concatenate query, key, and value, apply the short conv projection and split again
            x = torch.cat([query, key, value], dim=1)  # [B, 3 * hidden_dim, *spatial_dims]

            # CP communication - gather along first spatial dimension while splitting across channels/hidden dimension
            if cp_group is not None and cp_group.size() > 1:
                x = AllToAllSingleFunction.apply(x, cp_group, "split_to_full", True)

            # Always pass cp_group to distributed convolutions
            if hasattr(self.short_conv, "__class__") and "Distributed" in self.short_conv.__class__.__name__:
                x = self.short_conv(x, cp_group)
            else:
                # Standard PyTorch convolution doesn't support cp_group
                x = self.short_conv(x)

            # CP communication - scatter along first spatial dimension while gathering across channels/hidden dimension
            if cp_group is not None and cp_group.size() > 1:
                x = AllToAllSingleFunction.apply(x, cp_group, "full_to_split", True)

            # Split query, key, and value along channels/hidden dimension
            query, key, value = x.split(query.shape[1], dim=1)
            # Avoid in-place ops on views returned by split
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # QK normalization.
        # Tensors are BHL: [B, C, *spatial]. Channel-first norms can operate
        # directly; channel-last norms need movedim(1, -1) / movedim(-1, 1).
        # K is only normalized when gate_nonlinear is Identity (linear gating),
        # because a nonlinear σ(K) already bounds the magnitude.
        if self.q_norm is not None:
            if is_channels_first_norm(self.q_norm):
                query = self.q_norm(query)
            else:
                query = self.q_norm(query.movedim(1, -1)).movedim(-1, 1)
            if isinstance(self.gate_nonlinear, torch.nn.Identity):
                if is_channels_first_norm(self.k_norm):
                    key = self.k_norm(key)
                else:
                    key = self.k_norm(key.movedim(1, -1)).movedim(-1, 1)

        # First gate: z = Q ⊙ σ(K)
        query = query * self.gate_nonlinear(key)

        # Apply PixelHyena normalization (use torch.nn.Identity for no normalization)
        if not isinstance(self.pixelhyena_norm, torch.nn.Identity):
            if is_channels_first_norm(self.pixelhyena_norm):
                query = self.pixelhyena_norm(query)
            else:
                shape = query.shape  # [B, C, *spatial]
                query = query.movedim(1, -1).reshape(-1, shape[1])
                query = self.pixelhyena_norm(query)
                query = query.view(shape[0], *shape[2:], shape[1]).movedim(-1, 1)

        # CP communication - gather along first spatial dimension while splitting across channels/hidden dimension
        if cp_group is not None and cp_group.size() > 1:
            query = AllToAllSingleFunction.apply(query, cp_group, "split_to_full", True)

        # Apply global convolution
        y = self.global_conv(query, is_bhl_input=True, cp_group=cp_group, **mixer_kwargs)

        # CP communication - scatter along first spatial dimension while gathering across channels/hidden dimension
        if cp_group is not None and cp_group.size() > 1:
            y = AllToAllSingleFunction.apply(y, cp_group, "full_to_split", True)

        # Second gate: y = h ⊙ σ₂(V)
        y = y * self.gate_nonlinear_2(value)

        # Output normalization (after the second gate).
        if not isinstance(self.output_norm, torch.nn.Identity):
            if is_channels_first_norm(self.output_norm):
                y = self.output_norm(y)
            else:
                shape = y.shape  # [B, C, *spatial]
                y = y.movedim(1, -1).reshape(-1, shape[1])
                y = self.output_norm(y)
                y = y.view(shape[0], *shape[2:], shape[1]).movedim(-1, 1)

        # Reshape back to [B, * spatial_dims, C]
        return rearrange(y, "b c ... -> b ... c")
