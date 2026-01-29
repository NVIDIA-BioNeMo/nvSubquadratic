# TODO: Add license header here

"""Default mixer configurations for spatial recall 2D experiments.

This module provides pre-configured LazyConfigs for commonly used sequence mixers:
- Hyena (with CKConvND and SIREN kernel)
- Mamba2 (bidirectional)
- Attention (multi-head self-attention)

All mixers use interpolators to automatically pick up values from the config:
- "${net.hidden_dim}" for hidden dimension
- "${net.data_dim}" for spatial dimensionality
- "${net.num_blocks}" for number of blocks (used in init scaling)
- "${dataset.canvas_size}" for input spatial size

Usage:
    from examples.spatial_recall_2d.mixer_defaults import get_hyena_mixer_cfg

    config = get_base_config(
        in_channels=1,
        out_channels=1,
        mixer_cfg=get_hyena_mixer_cfg(),
    )
"""

import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.delta_hyena import DeltaHyena, ReasoningDeltaHyena
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer


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
    apply_qk_norm: bool = True,
    use_rope: bool = False,
    rope_base: float = 10000.0,
) -> LazyConfig:
    """Get Hyena mixer configuration with CKConvND and SIREN kernel."""
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
                    L_cache="${dataset.canvas_size}",
                    use_bias=True,
                    hidden_omega_0=kernel_hidden_omega_0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type=grid_type,
                fft_padding=fft_padding,
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels="3 * ${net.hidden_dim}",
                out_channels="3 * ${net.hidden_dim}",
                kernel_size=3,
                groups="3 * ${net.hidden_dim}",
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
            pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
            apply_qk_norm=apply_qk_norm,
            use_rope=use_rope,
            rope_base=rope_base,
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
    )


# =============================================================================
# Associative Delta-Hyena Mixer
# =============================================================================


def get_delta_hyena_mixer_cfg(
    num_heads: int = 8,
    gamma_init: float = 0.1,
    # SIREN kernel params for the value filter
    kernel_mlp_hidden_dim: int = 32,
    kernel_num_layers: int = 3,
    kernel_embedding_dim: int = 32,
    kernel_omega_0: float = 10.0,
    kernel_hidden_omega_0: float = 1.0,
    # CKConv params
    grid_type: str = "double",
    fft_padding: str = "zero",
    # Hyena params
    apply_qk_norm: bool = True,
    use_rope: bool = False,
    rope_base: float = 10000.0,
) -> LazyConfig:
    """Get Associative Delta-Hyena mixer configuration."""
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim="${net.hidden_dim}",
        mixer_cfg=LazyConfig(DeltaHyena)(
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
                    L_cache="${dataset.canvas_size}",
                    use_bias=True,
                    hidden_omega_0=kernel_hidden_omega_0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type=grid_type,
                fft_padding=fft_padding,
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels="3 * ${net.hidden_dim}",
                out_channels="3 * ${net.hidden_dim}",
                kernel_size=3,
                groups="3 * ${net.hidden_dim}",
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
            pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
            apply_qk_norm=apply_qk_norm,
            use_rope=use_rope,
            rope_base=rope_base,
            hidden_dim="${net.hidden_dim}",
            num_heads=num_heads,
            gamma_init=gamma_init,
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
    )


# =============================================================================
# Reasoning Delta-Hyena Mixer
# =============================================================================


def get_reasoning_delta_hyena_mixer_cfg(
    num_heads: int = 8,
    gamma_init: float = 0.1,
    num_recurrence: int = 3,
    # SIREN kernel params for the value filter
    kernel_mlp_hidden_dim: int = 32,
    kernel_num_layers: int = 3,
    kernel_embedding_dim: int = 32,
    kernel_omega_0: float = 10.0,
    kernel_hidden_omega_0: float = 1.0,
    # CKConv params
    grid_type: str = "double",
    fft_padding: str = "zero",
    # Hyena params
    apply_qk_norm: bool = True,
    use_rope: bool = False,
    rope_base: float = 10000.0,
) -> LazyConfig:
    """Get Reasoning Delta-Hyena mixer configuration."""
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim="${net.hidden_dim}",
        mixer_cfg=LazyConfig(ReasoningDeltaHyena)(
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
                    L_cache="${dataset.canvas_size}",
                    use_bias=True,
                    hidden_omega_0=kernel_hidden_omega_0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type=grid_type,
                fft_padding=fft_padding,
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels="3 * ${net.hidden_dim}",
                out_channels="3 * ${net.hidden_dim}",
                kernel_size=3,
                groups="3 * ${net.hidden_dim}",
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
            pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
            apply_qk_norm=apply_qk_norm,
            use_rope=use_rope,
            rope_base=rope_base,
            hidden_dim="${net.hidden_dim}",
            num_heads=num_heads,
            gamma_init=gamma_init,
            num_recurrence=num_recurrence,
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
) -> LazyConfig:
    """Get Mamba2 mixer configuration."""
    # Import here to avoid requiring mamba-ssm if not using Mamba
    from mamba_ssm import Mamba2

    from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer

    return LazyConfig(MambaNDMixer)(
        mamba_layer_cfg=LazyConfig(Mamba2)(
            d_model="${net.hidden_dim}",
            headdim=headdim,
            expand=expand,
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
    """Get Attention mixer configuration."""
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
