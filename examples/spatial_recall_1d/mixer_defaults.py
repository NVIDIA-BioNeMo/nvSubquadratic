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

"""Default mixer configurations for 1D spatial recall experiments.

Provides pre-configured LazyConfig mixers for common architectures:
- Hyena: CKConv with SIREN kernel (causal or non-causal)
- Mamba: Bidirectional Mamba2
- Attention: Multi-head self-attention

Key difference from 2D:
- L_cache uses canvas_length (e.g., 4096) not canvas_size (e.g., 64)
- Short conv uses Conv1d not Conv2d (or CausalConv1D for causal mode)

Usage:
    from examples.spatial_recall_1d.mixer_defaults import get_hyena_mixer_cfg

    mixer_cfg = get_hyena_mixer_cfg(is_causal=True)
"""

from typing import Optional

import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.causal_conv1d import CausalConv1D
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init


# =============================================================================
# Hyena Mixer (with CKConvND + SIREN kernel)
# =============================================================================


def get_hyena_mixer_cfg(
    # SIREN kernel params
    kernel_mlp_hidden_dim: int = 32,
    kernel_num_layers: int = 3,
    kernel_embedding_dim: int = 32,
    kernel_omega_0: float = 10.0,
    kernel_hidden_omega_0: float = 1.0,
    # CKConv params
    grid_type: str = "double",
    fft_padding: str = "zero",
    # Hyena params
    qk_norm_cfg: Optional[LazyConfig] = None,
    use_rope: bool = False,
    rope_base: float = 10000.0,
    is_causal: bool = False,
    # Kernel cache size (for 1D, use canvas_length by default)
    L_cache: str | int = "${dataset.canvas_size} * ${dataset.canvas_size}",
    short_conv_kernel_size: int = 3,
) -> LazyConfig:
    """Get Hyena mixer config with SIREN kernel for 1D sequences.

    Key difference from 2D: L_cache uses canvas_length (4096) not canvas_size (64).

    Args:
        kernel_mlp_hidden_dim: Hidden dim for SIREN MLP.
        kernel_num_layers: Number of SIREN layers.
        kernel_embedding_dim: Embedding dimension for SIREN.
        kernel_omega_0: First layer omega for SIREN.
        kernel_hidden_omega_0: Hidden layer omega for SIREN.
        grid_type: Grid type for CKConv ("single" or "double").
        fft_padding: FFT padding mode ("zero" or "circular").
        qk_norm_cfg: QK normalization config (e.g., LazyConfig(L2Norm)()). None to disable.
        use_rope: Use rotary position embeddings.
        rope_base: Base for RoPE.
        is_causal: Whether to use causal convolutions (for autoregressive tasks).
        L_cache: Kernel cache size (should be canvas_length for 1D).
        short_conv_kernel_size: Kernel size for short conv (default: 3).

    Returns:
        LazyConfig for QKVSequenceMixer with Hyena.
    """
    # Short conv: CausalConv1D for causal mode, else standard Conv1d with symmetric padding
    if is_causal:
        short_conv_cfg = LazyConfig(CausalConv1D)(
            in_channels="3 * ${net.hidden_dim}",
            out_channels="3 * ${net.hidden_dim}",
            kernel_size=short_conv_kernel_size,
            groups="3 * ${net.hidden_dim}",
            bias=False,
        )
    else:
        short_conv_cfg = LazyConfig(torch.nn.Conv1d)(
            in_channels="3 * ${net.hidden_dim}",
            out_channels="3 * ${net.hidden_dim}",
            kernel_size=short_conv_kernel_size,
            groups="3 * ${net.hidden_dim}",
            padding=short_conv_kernel_size // 2,
            bias=False,
        )

    return LazyConfig(QKVSequenceMixer)(
        hidden_dim="${net.hidden_dim}",
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim="${net.data_dim}",
                hidden_dim="${net.hidden_dim}",
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim="${net.data_dim}",
                    out_dim="${net.hidden_dim}",
                    mlp_hidden_dim=kernel_mlp_hidden_dim,
                    num_layers=kernel_num_layers,
                    embedding_dim=kernel_embedding_dim,
                    omega_0=kernel_omega_0,
                    L_cache=L_cache,
                    use_bias=True,
                    hidden_omega_0=kernel_hidden_omega_0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type=grid_type,
                fft_padding=fft_padding,
                is_causal=is_causal,
            ),
            short_conv_cfg=short_conv_cfg,
            gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
            pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
            qk_norm_cfg=qk_norm_cfg,
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
    )


# =============================================================================
# Mamba2 Mixer (bidirectional)
# =============================================================================


def get_mamba_mixer_cfg(
    headdim: int = 64,
    expand: int = 2,
    bidirectional: bool = True,
    # SSM state and frequency parameters
    d_state: int = 128,
    A_init_range: tuple[float, float] = (1, 16),
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init_floor: float = 1e-4,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
) -> LazyConfig:
    """Get Mamba2 mixer configuration.

    Note: Mamba is NOT wrapped in QKVSequenceMixer - it handles its own projections.

    Args:
        headdim: Mamba2 head dimension.
        expand: Expansion factor for inner dimension.
        bidirectional: Whether to use bidirectional Mamba.
        d_state: SSM state dimension. Larger = more capacity for long-range patterns.
        A_init_range: Range for A matrix initialization (controls decay rate).
            Smaller values = slower decay = longer memory.
            Larger values = faster decay = shorter memory.
            Default (1, 16) is the original Mamba2 setting.
        dt_min: Minimum time step for discretization initialization.
            Smaller = finer temporal resolution = slower effective decay.
        dt_max: Maximum time step for discretization initialization.
            Larger = coarser temporal resolution = faster effective decay.
        dt_init_floor: Floor for dt initialization (numerical stability).
        dt_limit: Runtime limits on dt values (min, max).

    Returns:
        LazyConfig for MambaNDMixer.

    Example - Longer memory (analogous to larger rope_base/L_cache):
        get_mamba_mixer_cfg(A_init_range=(0.5, 4), dt_min=0.0001, dt_max=0.01)

    Example - Shorter memory (faster dynamics):
        get_mamba_mixer_cfg(A_init_range=(4, 32), dt_min=0.01, dt_max=0.5)
    """
    # Import here to avoid requiring mamba-ssm if not using Mamba
    from mamba_ssm import Mamba2

    from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer

    return LazyConfig(MambaNDMixer)(
        mamba_layer_cfg=LazyConfig(Mamba2)(
            d_model="${net.hidden_dim}",
            headdim=headdim,
            expand=expand,
            d_state=d_state,
            A_init_range=A_init_range,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            dt_limit=dt_limit,
        ),
        bidirectional=bidirectional,
    )


# =============================================================================
# Attention Mixer (multi-head self-attention)
# =============================================================================


def get_attention_mixer_cfg(
    num_heads: int = 8,
    apply_qk_norm: bool = True,
    use_rope: bool = True,
    is_causal: bool = False,
    rope_base: float = 10000.0,
    attn_dropout: float = 0.0,
) -> LazyConfig:
    """Get Attention mixer configuration.

    Args:
        num_heads: Number of attention heads.
        apply_qk_norm: Whether to apply QK normalization.
        use_rope: Whether to use rotary position embeddings.
        is_causal: Whether to use causal attention.
        rope_base: Base for rotary position embeddings.
        attn_dropout: Attention dropout rate.

    Returns:
        LazyConfig for QKVSequenceMixer with Attention.
    """
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim="${net.hidden_dim}",
        mixer_cfg=LazyConfig(Attention)(
            hidden_dim="${net.hidden_dim}",
            num_heads=num_heads,
            apply_qk_norm=apply_qk_norm,
            use_rope=use_rope,
            is_causal=is_causal,
            rope_base=rope_base,
            attn_dropout=attn_dropout,
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
    )
