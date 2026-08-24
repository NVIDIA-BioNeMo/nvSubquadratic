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

"""Gated Delta Product (GDP) — a standalone port of Megatron's mixer, for benchmarking.

GDP is the mixer Nemotron is evaluating as a replacement for its Mamba-2 layers, so it
is the operator HyenaND actually has to beat. This is a **benchmark baseline**, not an
nvSubquadratic contribution: it exists so ``gdp`` can appear alongside ``mamba`` in the
forward-time sweeps.

Ported from, and kept faithful to::

    megatron-lm @ dev-arch-mar2026-v2 (daf96b8e)
      megatron/core/ssm/gated_delta_product_original_v4.py      (num_householder=3)
      megatron/core/ssm/gated_delta_product_original_v4_nh2.py  (num_householder=2)

The two upstream files differ only in ``num_householder``; here that is a constructor
argument, defaulting to **3** to match ``gated_delta_product_original_v4.py`` — the
actively maintained file, and the one nemotron_workspace's HANDOFF cites as the
reference. Pass ``num_householder=2`` for the ``_nh2`` variant.

**Why a port rather than importing Megatron.** The operator being measured is
``fla.ops.gated_delta_product.chunk_gated_delta_product``, an external library kernel
that is identical either way. Importing the Megatron module instead would require
TransformerEngine (its spec builds ``in_proj``/``out_proj`` from
``TELayerNormColumnParallelLinear``/``TERowParallelLinear``), a megatron-core built from
the ADLR branch (the released 0.18.2 has no GDP at all), and an initialised
``ProcessGroupCollection`` — all of which are inert at TP=1/CP=1 on one GPU.

**What is deliberately dropped**, none of which does arithmetic at TP=1/CP=1 on a single
device: tensor/context parallelism (``GDPContextParallel``), inference contexts (static
and dynamic batching, conv/ssm state caches), sequence packing (``cu_seqlens``,
``seq_idx``), and sharded-checkpoint plumbing.

**What differs numerically from upstream**: ``in_proj``/``out_proj`` are
``torch.nn.Linear`` rather than TE fused linears. At TP=1 both are plain GEMMs, but TE's
fused epilogues can be faster, so absolute timings here are a slight *under*-estimate of
GDP's projection speed and a correspondingly conservative comparison for HyenaND. The
FLA kernel — which dominates at long sequence — is the same code upstream calls.

Requires ``flash-linear-attention`` (``fla``), ``mamba-ssm`` (for ``RMSNormGated`` and
``causal_conv1d``) and ``einops``.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def _shim_tilelang() -> None:
    """Make ``import tilelang`` fail cleanly, before anything imports ``mamba_ssm``.

    ``mamba_ssm``'s package ``__init__`` eagerly imports ``modules.mamba3`` ->
    ``tilelang`` -> a bundled TVM that collides with the ``apache-tvm-ffi`` that
    subquadratic-ops-torch >= 0.2.2 installs, raising
    ``AttributeError: attribute '__dict__' of 'type' objects is not writable``.
    Setting the module to ``None`` turns that into an ImportError, which mamba_ssm's
    optional import skips — the same shim ``benchmark_patch_size_2d._mamba_mixer_cfg``
    uses for Mamba2.

    This must run before the FIRST ``mamba_ssm`` or ``fla`` import in the process, not
    merely before the ``fla`` one: importing *any* ``mamba_ssm`` submodule (e.g.
    ``ops.triton.layernorm_gated`` for ``RMSNormGated``) executes that package
    ``__init__`` and trips the same chain.
    """
    sys.modules.setdefault("tilelang", None)


def _import_fla():
    """Return the FLA chunked GDP kernel."""
    _shim_tilelang()
    from fla.ops.gated_delta_product import chunk_gated_delta_product

    return chunk_gated_delta_product


class GatedDeltaProduct(nn.Module):
    """Single GDP mixer layer, ``[B, L, D] -> [B, L, D]``.

    Upstream is sequence-first (``[L, B, D]``); this takes batch-first so it drops into
    the benchmark harness beside the other mixers, and transposes internally.

    Args:
        d_model: Model hidden size.
        num_householder: Householder products per token (``M``). 3 matches
            ``gated_delta_product_original_v4.py``, 2 the ``_nh2`` variant. Drives both
            the ``in_proj`` width and the kernel's effective sequence length (``L*M``),
            so it is the dominant cost knob.
        d_state: SSM state dim (Megatron ``mamba_state_dim``).
        headdim: Per-head width (Megatron ``mamba_head_dim``).
        ngroups: Key/query groups (Megatron ``mamba_num_groups``). When
            ``nheads > ngroups`` the kernel's q/k are repeat-interleaved, GQA-style.
        nheads: Value heads. Defaults to ``d_model * expand // headdim``, matching
            ``TransformerConfig.mamba_num_heads``'s documented fallback.
        expand: Inner expansion factor, used only for the ``nheads`` default.
        d_conv: Causal short-conv width.
        conv_bias / bias: Bias on the short conv / the projections.
        rmsnorm: Apply the gated RMSNorm before ``out_proj``.
        dt_min, dt_max, dt_init_floor: ``dt_bias`` init range.
        A_init_range: ``A`` init range; ``A_log = log(A)`` is kept in fp32.
        device, dtype: Passed to the submodules.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_householder: int = 3,
        d_state: int = 128,
        headdim: int = 64,
        ngroups: int = 8,
        nheads: int | None = None,
        expand: int = 2,
        d_conv: int = 4,
        conv_bias: bool = False,
        bias: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_init_range: tuple[float, float] = (0.0, 16.0),
        conv_init: float | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if num_householder < 1:
            raise ValueError(f"num_householder must be >= 1, got {num_householder}.")

        self.d_model = d_model
        self.num_householder = num_householder
        self.d_state = d_state
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = nheads if nheads is not None else (d_model * expand) // headdim
        self.d_inner = self.nheads * self.headdim
        self.d_conv = d_conv
        self.rmsnorm = rmsnorm
        self.activation = "silu"

        if self.nheads % self.ngroups != 0:
            raise ValueError(
                f"nheads ({self.nheads}) must be divisible by ngroups ({self.ngroups}) "
                "for the q/k repeat_interleave."
            )

        M = num_householder
        factory = {"device": device, "dtype": dtype}

        # z | V (M copies) | K (M copies) | Q | b (M per head) | a (1 per head).
        # Width formula copied verbatim from the upstream in_proj construction.
        in_width = (
            self.d_inner * (1 + M) + self.ngroups * self.d_state * (M + 1) + self.nheads * (M + 1)
        )
        self.in_proj = nn.Linear(d_model, in_width, bias=bias, **factory)

        # Short conv spans V and K (M copies each) plus one Q — not z, and not b/a.
        conv_dim = self.d_inner * M + (M + 1) * self.ngroups * self.d_state
        self.conv_dim = conv_dim
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory,
        )
        if conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -conv_init, conv_init)

        # dt_bias init so softplus(dt_bias) lands in [dt_min, dt_max]; A_log stays fp32
        # or A can underflow to -inf in fp16 (upstream's comment).
        dt = torch.exp(
            torch.rand(self.nheads, device=device) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))

        if rmsnorm:
            _shim_tilelang()  # must precede the first mamba_ssm import in the process
            from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

            self.norm = RMSNormGated(
                self.d_inner,
                eps=1e-5,
                group_size=self.d_inner // self.ngroups,
                norm_before_gate=norm_before_gate,
                **factory,
            )

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory)

        self._chunk_gated_delta_product = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the mixer. ``hidden_states`` is ``[B, L, D]``; returns the same shape."""
        from causal_conv1d import causal_conv1d_fn
        from einops import rearrange

        if self._chunk_gated_delta_product is None:
            self._chunk_gated_delta_product = _import_fla()
        chunk_gated_delta_product = self._chunk_gated_delta_product

        M, H, P, G, S = self.num_householder, self.nheads, self.headdim, self.ngroups, self.d_state

        zVKQba = self.in_proj(hidden_states)
        z, VKQ, ba = torch.split(
            zVKQba,
            [self.d_inner, self.conv_dim, H * (M + 1)],
            dim=-1,
        )

        # causal_conv1d_fn wants [B, D, L], channels-first contiguous.
        VKQ = rearrange(VKQ, "b l d -> b d l").contiguous()
        VKQ = causal_conv1d_fn(
            x=VKQ,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
        )
        VKQ = rearrange(VKQ, "b d l -> b l d").contiguous()

        value, key, query = torch.split(
            VKQ, [self.d_inner * M, G * S * M, G * S], dim=-1
        )
        b, a = torch.split(ba, [H * M, H], dim=-1)

        # The M householder copies are folded into the sequence axis: the kernel sees
        # L*M rows for v/k/beta while q and g stay at L, one query per real token.
        z = rearrange(z, "b l (h p) -> b l h p", p=P).contiguous()
        value = rearrange(value, "b l (m h p) -> b (l m) h p", m=M, p=P).contiguous()
        key = rearrange(key, "b l (m g n) -> b (l m) g n", m=M, n=S).contiguous()
        query = rearrange(query, "b l (g n) -> b l g n", n=S).contiguous()
        beta = rearrange(b.contiguous().sigmoid(), "b l (m h) -> b (l m) h", m=M).contiguous()

        # fp32 for A: in fp16 exp(A_log) can underflow and give -inf decay.
        g = -self.A_log.float().exp() * F.softplus(a.float().contiguous() + self.dt_bias)

        if H // G > 1:
            query = query.repeat_interleave(H // G, dim=2)
            key = key.repeat_interleave(H // G, dim=2)

        core_attn_out, _ = chunk_gated_delta_product(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            num_householder=M,
            use_qk_l2norm_in_kernel=True,
        )

        y = rearrange(core_attn_out, "b l h p -> b l (h p)").contiguous()
        if self.rmsnorm:
            y = self.norm(y, rearrange(z, "b l h p -> b l (h p)").contiguous())
        return self.out_proj(y)


class GatedDeltaProductNDMixer(nn.Module):
    """Adapter giving :class:`GatedDeltaProduct` the benchmark's ND tensor convention.

    The harness feeds ``[B, R, ...(N axes)..., C]``; GDP is a causal 1D sequence mixer,
    so the spatial axes are flattened to one scan and restored afterwards — the same
    rasterisation the Mamba2 baseline uses, which is what makes the two comparable.

    Args:
        mixer_cfg: A ``LazyConfig`` for the inner :class:`GatedDeltaProduct`, following
            the repo's nested-config convention (as ``MambaNDMixer`` takes
            ``mamba_layer_cfg``): the config is instantiated here rather than by the
            caller, so ``instantiate`` does not have to recurse into the kwarg.
    """

    def __init__(self, mixer_cfg) -> None:
        super().__init__()
        from nvsubquadratic.lazy_config import instantiate

        self.mixer = instantiate(mixer_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the spatial axes, mix, restore. ``x`` is ``[B, ...spatial, C]``."""
        shape = x.shape
        y = self.mixer(x.reshape(shape[0], -1, shape[-1]))
        return y.reshape(shape)
