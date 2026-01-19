# TODO: Add license header here

"""Default mixer configurations for 1D spatial recall experiments.

Provides pre-configured LazyConfig mixers for common architectures:
- Hyena: CKConv with SIREN kernel
- Mamba: Bidirectional Mamba2
- Attention: Multi-head self-attention

Key difference from 2D:
- L_cache uses canvas_length (e.g., 4096) not canvas_size (e.g., 64)
- Short conv uses Conv1d not Conv2d

Usage:
    from examples.spatial_recall_1d.mixer_defaults import get_hyena_mixer_cfg

    mixer_cfg = get_hyena_mixer_cfg()
"""

import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.ckconv_nd import CKConvND
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
    # Kernel cache size (for 1D, use canvas_length)
    L_cache: str | int = "${dataset.canvas_length}",
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
        apply_qk_norm: Apply QK normalization.
        use_rope: Use rotary position embeddings.
        rope_base: Base for RoPE.
        L_cache: Kernel cache size (should be canvas_length for 1D).

    Returns:
        LazyConfig for QKVSequenceMixer with Hyena.
    """
    raise NotImplementedError("Hyena is not supported for 1D sequences.")
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
            ),
            # Short conv: Conv1d for 1D sequences
            short_conv_cfg=LazyConfig(torch.nn.Conv1d)(
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
# Mamba2 Mixer (bidirectional)
# =============================================================================


def get_mamba_mixer_cfg(
    headdim: int = 64,
    expand: int = 2,
    bidirectional: bool = True,
) -> LazyConfig:
    """Get Mamba2 mixer configuration.

    Note: Mamba is NOT wrapped in QKVSequenceMixer - it handles its own projections.

    Args:
        headdim: Mamba2 head dimension.
        expand: Expansion factor for inner dimension.
        bidirectional: Whether to use bidirectional Mamba.

    Returns:
        LazyConfig for MambaNDMixer.
    """
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
