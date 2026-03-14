"""ViT-5-Small + Hyena ImageNet-1k pretrain — CLS-row, gated, FiLM + pos-embed conditioning, EMA.

Extends the FiLM config with input-dependent spatial warping: the first
FiLM (gamma, beta) pair modulates the SIREN positional embedding output
*before* it enters the hidden layers.  This lets the model learn
input-dependent coordinate transforms — effectively warping the kernel's
spatial structure based on the input.

The remaining FiLM pairs modulate hidden activations as before, so this
is strictly a superset of standard FiLM conditioning.

Only the network definition lives here; everything else (dataset, optimizer,
scheduler, EMA, etc.) comes from ``_pretrain_base``.

CLS-row architecture: CLS + 13 registers as extra row -> 15x14 grid.
Gated variant: SiLU first gate, Sigmoid second gate.
"""

import torch

from examples.vit5_imagenet.v3._pretrain_base import (
    FINAL_IMAGE_SIZE,
    HIDDEN_DIM,
    INIT_FN_FACTORY,
    INPUT_CHANNELS,
    NUM_BLOCKS,
    NUM_CLASSES,
    NUM_PATCHES_H,
    NUM_PATCHES_W,
    PATCH_SIZE,
    get_base_config,
    make_block_cfg,
)
from experiments.callbacks.film_monitor import FiLMMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.qk_norm import L2Norm


NUM_REGISTERS = NUM_PATCHES_W - 1  # 13 — fills the extra row: [CLS, regs, patches] -> 15x14 grid

# ─── Hyena / SIREN kernel hyperparameters ────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# ─── FiLM conditioning ──────────────────────────────────────────────────────────
FILM_HIDDEN_DIM = 64
FILM_PARAMETERIZATION = "residual"  # "residual" or "direct"
FILM_NO_WEIGHT_DECAY = 1e-3
FILM_INIT_TYPE = "identity"  # "identity" or "small_random"
FILM_INIT_STD = 1e-4


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Hyena CLS-row gated + FiLM + pos-embed pretrain config with EMA."""
    config = get_base_config()
    config.compile_compatible_fftconv = True

    film_cfg = LazyConfig(KernelFiLMGenerator)(
        cond_dim=HIDDEN_DIM,
        kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_film_layers=KERNEL_NUM_LAYERS,  # +1 vs standard FiLM: extra pair for pos-embed
        film_hidden_dim=FILM_HIDDEN_DIM,
        film_parameterization=FILM_PARAMETERIZATION,
        no_weight_decay=FILM_NO_WEIGHT_DECAY,
        init_type=FILM_INIT_TYPE,
        init_std=FILM_INIT_STD,
        gamma_max=4.0,
    )

    hyena_mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2,
                    out_dim=HIDDEN_DIM,
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache=NUM_PATCHES_H + 1,
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                    film_cfg=film_cfg,
                    film_on_pos_embed=True,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * HIDDEN_DIM,
                out_channels=3 * HIDDEN_DIM,
                kernel_size=3,
                groups=3 * HIDDEN_DIM,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            use_rope=False,
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )

    register_pooling_cfg = LazyConfig(RegisterPooling)(num_registers=NUM_REGISTERS)

    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=FINAL_IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        prepend_registers=True,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=make_block_cfg(
            sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
                inner_mixer_cfg=hyena_mixer_cfg,
                grid_w=NUM_PATCHES_W,
            ),
            register_pooling_cfg=register_pooling_cfg,
            num_registers=NUM_REGISTERS,
        ),
    )

    config.callbacks.append(
        LazyConfig(FiLMMonitorCallback)(
            log_every_n_steps=50,
            num_film_layers=KERNEL_NUM_LAYERS,
            film_on_pos_embed=True,
        )
    )

    return config
