# TODO: Add license header here

"""Shared mixer defaults for spatial recall v2 experiments.

Modernised mixer factories shared across all spatial recall v2 tasks and dimensions.
All mixers use config interpolation (e.g. ``"${net.hidden_dim}"``) to stay in sync
with the experiment config.

v2 changes relative to v1:
- Hyena: RMSNorm (was LayerNorm), SiLU+Sigmoid gates, L2 QK-norm,
  output_norm, fft_backend="subq_ops".
- Attention: unchanged (already modern in v1).
- Mamba: unchanged (handles its own projections).
"""

import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.rms_norm_channel_first import RMSNormChannelFirst
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# Default SIREN kernel hyperparameters (same as v5 ImageNet)
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0


# =============================================================================
# Hyena Mixer (with CKConvND + SIREN kernel) -- v2 architecture
# =============================================================================


def short_conv_cfg(data_dim: int, kernel_size: int = 3) -> LazyConfig:
    """Build the appropriate short convolution for the given dimensionality.

    For 1D, uses :class:`CausalConv1D` whose ``is_causal`` is wired via
    interpolator to the sibling global_conv's ``is_causal`` so both stay in sync.
    """
    if data_dim == 1:
        from nvsubquadratic.modules.causal_conv1d import CausalConv1D

        return LazyConfig(CausalConv1D)(
            in_channels="3 * ${net.hidden_dim}",
            out_channels="3 * ${net.hidden_dim}",
            kernel_size=kernel_size,
            groups="3 * ${net.hidden_dim}",
            bias=False,
            is_causal="${net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.is_causal}",
        )
    elif data_dim == 2:
        conv_cls = torch.nn.Conv2d
    elif data_dim == 3:
        conv_cls = torch.nn.Conv3d
    else:
        raise ValueError(f"Unsupported data_dim={data_dim}")

    return LazyConfig(conv_cls)(
        in_channels="3 * ${net.hidden_dim}",
        out_channels="3 * ${net.hidden_dim}",
        kernel_size=kernel_size,
        groups="3 * ${net.hidden_dim}",
        padding=kernel_size // 2,
        bias=False,
    )


def get_hyena_mixer_cfg(
    short_conv_cfg: LazyConfig,
    L_cache: str | int = "${dataset.canvas_size}",
    is_causal: bool = False,
    fft_backend: str = "subq_ops",
) -> LazyConfig:
    """Get modernised Hyena mixer config (v2).

    v2 architecture (ported from ImageNet v5):
    - SiLU first gate + Sigmoid second gate
    - L2 QK-norm (channel-first)
    - RMSNormChannelFirst for pixelhyena and output norms

    All dimension-dependent fields (CKConvND, SIRENKernelND) use the
    ``"${net.data_dim}"`` interpolator.  The two values that require a
    concrete Python type — the short convolution module and the FFT backend
    — are passed explicitly by the caller.

    Args:
        short_conv_cfg: LazyConfig for the depthwise short convolution.
            Build with :func:`short_conv_cfg`.
        L_cache: Kernel cache size. Default ``"${dataset.canvas_size}"``.
        is_causal: Causal convolutions (for 1D autoregressive tasks).
        fft_backend: ``"subq_ops"`` (2D-only CUDA kernel) or ``"torch_fft"``.

    Returns:
        LazyConfig for QKVSequenceMixer wrapping the Hyena mixer.
    """
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim="${net.hidden_dim}",
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim="${net.data_dim}",
                hidden_dim="${net.hidden_dim}",
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim="${net.data_dim}",
                    out_dim="${net.hidden_dim}",
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache=L_cache,
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
                fft_backend=fft_backend,
                is_causal=is_causal,
            ),
            short_conv_cfg=short_conv_cfg,
            # v2: SiLU + Sigmoid gating (was Identity in v1)
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
            # v2: channel-first RMSNorm — Hyena works in [B, C, *spatial] so
            # channel-first norms avoid movedim transposes and are compile-friendly.
            pixelhyena_norm_cfg=LazyConfig(RMSNormChannelFirst)(dim="${net.hidden_dim}", eps=1e-6, use_quack=False),
            output_norm_cfg=LazyConfig(RMSNormChannelFirst)(dim="${net.hidden_dim}", eps=1e-6, use_quack=False),
            # v2: L2 QK-norm on channel dim (dim=1 = channel-first)
            qk_norm_cfg=LazyConfig(L2Norm)(dim=1),
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
    )


# =============================================================================
# Mamba2 Mixer (bidirectional)
# =============================================================================


def get_mamba_mixer_cfg(
    headdim: int = 32,
    expand: int = 2,
    bidirectional: bool = True,
) -> LazyConfig:
    """Get Mamba2 mixer config.

    Note: Mamba handles its own projections -- NOT wrapped in QKVSequenceMixer.

    Args:
        headdim: Head dimension.
        expand: Inner dimension expansion factor.
        bidirectional: Bidirectional processing.

    Returns:
        LazyConfig for MambaNDMixer.
    """
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
    rope_spatial_dims: tuple[int, ...] | None = None,
) -> LazyConfig:
    """Get Attention mixer config.

    Args:
        num_heads: Number of attention heads.
        apply_qk_norm: Apply QK normalization.
        use_rope: Rotary position embeddings.
        is_causal: Causal attention mask.
        rope_spatial_dims: Spatial dimensions for precomputed RoPE buffers.
            Required when use_rope=True. E.g. (4096,) for 1D, (64, 64) for 2D.

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
            rope_spatial_dims=rope_spatial_dims,
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
    )
