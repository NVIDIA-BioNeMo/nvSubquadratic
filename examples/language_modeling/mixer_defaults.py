"""Causal mixer configurations for autoregressive language modeling tasks.

Provides causal variants of Hyena and Attention mixers, shared between
MQAR and WikiText-103 experiments.
"""

import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.causal_conv1d import CausalConv1D
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer


def get_causal_hyena_mixer_cfg(
    L_cache: int | str = 256,
    kernel_mlp_hidden_dim: int = 32,
    kernel_num_layers: int = 3,
    kernel_embedding_dim: int = 32,
    kernel_omega_0: float = 10.0,
    kernel_hidden_omega_0: float = 1.0,
    grid_type: str = "double",
    fft_padding: str = "zero",
    apply_qk_norm: bool = True,
    use_rope: bool = False,
    rope_base: float = 10000.0,
    short_conv_kernel_size: int = 3,
    mask_cfg: LazyConfig | None = None,
) -> LazyConfig:
    """Get causal Hyena mixer config for autoregressive tasks.

    Args:
        L_cache: Kernel cache size (should match seq_len).
        kernel_mlp_hidden_dim: Hidden dim for SIREN MLP.
        kernel_num_layers: Number of SIREN layers.
        kernel_embedding_dim: Embedding dimension for SIREN.
        kernel_omega_0: First layer omega for SIREN.
        kernel_hidden_omega_0: Hidden layer omega for SIREN.
        grid_type: Grid type for CKConv.
        fft_padding: FFT padding mode.
        apply_qk_norm: Apply QK normalization.
        use_rope: Use rotary position embeddings.
        rope_base: Base for RoPE.
        short_conv_kernel_size: Kernel size for causal short conv.
        mask_cfg: Optional mask config for filter decay (default: Identity).

    Returns:
        LazyConfig for QKVSequenceMixer with causal Hyena.
    """
    if mask_cfg is None:
        mask_cfg = LazyConfig(torch.nn.Identity)()

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
                mask_cfg=mask_cfg,
                grid_type=grid_type,
                fft_padding=fft_padding,
                is_causal=True,
            ),
            short_conv_cfg=LazyConfig(CausalConv1D)(
                in_channels="3 * ${net.hidden_dim}",
                out_channels="3 * ${net.hidden_dim}",
                kernel_size=short_conv_kernel_size,
                groups="3 * ${net.hidden_dim}",
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


def get_causal_attention_mixer_cfg(
    num_heads: int = 8,
    apply_qk_norm: bool = True,
    use_rope: bool = True,
    rope_base: float = 10000.0,
    attn_dropout: float = 0.0,
) -> LazyConfig:
    """Get causal Attention mixer config for autoregressive tasks.

    Args:
        num_heads: Number of attention heads.
        apply_qk_norm: Whether to apply QK normalization.
        use_rope: Whether to use rotary position embeddings.
        rope_base: Base for rotary position embeddings.
        attn_dropout: Attention dropout rate.

    Returns:
        LazyConfig for QKVSequenceMixer with causal Attention.
    """
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim="${net.hidden_dim}",
        mixer_cfg=LazyConfig(Attention)(
            hidden_dim="${net.hidden_dim}",
            num_heads=num_heads,
            apply_qk_norm=apply_qk_norm,
            use_rope=use_rope,
            is_causal=True,
            rope_base=rope_base,
            attn_dropout=attn_dropout,
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
    )
