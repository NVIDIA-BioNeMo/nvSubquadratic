"""Hyena ResNet baseline for ARC-AGI with FiLM conditioning.

This is identical to cfg_hyena_rearc.py, except the task token is not broadcast-added
to the spatial colour embeddings. Instead, it is routed as a conditioning vector into
the Hyena sequence mixer via FiLM to modulate the SIREN implicit convolution kernels
(gamma*h + beta).

Initialization is set to "identity", so step 0 is equivalent to a network with no task
conditioning (i.e. identical to current Hyena without the broadcast add).
"""

import math

import torch

from examples.arc._base import LEARNING_RATE, NUM_EPOCHS, NUM_GPUS, PLACEHOLDER
from experiments.datamodules.arc import ARCDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.arc_wrapper import ARCWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import KernelFiLMGenerator
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.arc_resnet import ARCResNet
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# ── Architecture ──────────────────────────────────────────────────────────────
EMBED_DIM = 384
NUM_BLOCKS = 12
PATCH_SIZE = 2
MAX_SIZE = 32  # 32×32 canvas → seq length 256 at patch_size=2
NUM_COLORS = 12  # 10 ARC colours + IGNORE(10) + PAD(11)
NUM_TASKS = 400  # training tasks only (same as reference)

# ── Hyena ─────────────────────────────────────────────────────────────────────
DATA_DIM = 2
PATCHED_RESOLUTION = MAX_SIZE // PATCH_SIZE  # 16
FFT_PADDING = "zero"  # ARC grids are non-periodic
GRID_TYPE = "single"
OMEGA_0 = 30.0
DROPOUT = 0.1

# ── FiLM ──────────────────────────────────────────────────────────────────────
# Matches the SIRENKernelND dimensions: embedding=64, num_layers=3 -> 2 hidden layers
COND_DIM = EMBED_DIM
KERNEL_HIDDEN_DIM = 64
NUM_FILM_LAYERS = 2
FILM_HIDDEN_DIM = 64

# ── Training ──────────────────────────────────────────────────────────────────
# 128 per GPU × 2 GPUs = 256 effective global batch size (matches VARC's global BS).
BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1
NUM_TRAINING_SAMPLES_REARC = 413_020


def get_config() -> ExperimentConfig:
    """Hyena ResNet trained on ARC + RE-ARC with FiLM task-token routing."""
    training_iterations = math.ceil(NUM_EPOCHS * NUM_TRAINING_SAMPLES_REARC / (BATCH_SIZE * NUM_GPUS))

    config = ExperimentConfig()
    config.debug = False
    config.seed = 42

    config.dataset = LazyConfig(ARCDataModule)(
        data_dir="data/arc/data",
        rearc_dir="/home/dwessel/code/VARC_info/raw_data/re_arc",
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        seed=config.seed,
        max_size=MAX_SIZE,
        num_color_permutations=9,
        rearc_num_color_permutations=0,
        val_task_split="training",
        val_subset="test",
    )

    config.lightning_wrapper_class = LazyConfig(ARCWrapper)()

    config.optimizer = LazyConfig(torch.optim.AdamW)(params=PLACEHOLDER, lr=LEARNING_RATE, weight_decay=0.0)

    config.train = TrainConfig(batch_size="${dataset.batch_size}", iterations=training_iterations, grad_clip=1.0)

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
        mode="max",
    )
    config.trainer.checkpoint_monitor = "val/exact_match"
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.compile_compatible_fftconv = True  # Use real-valued FFT path to avoid complex64 Inductor error
    config.trainer.precision = "bf16-mixed"

    norm_cfg = LazyConfig(RMSNorm)(dim=EMBED_DIM)

    film_cfg = LazyConfig(KernelFiLMGenerator)(
        cond_dim=COND_DIM,
        kernel_hidden_dim=KERNEL_HIDDEN_DIM,
        num_film_layers=NUM_FILM_LAYERS,
        film_hidden_dim=FILM_HIDDEN_DIM,
        init_type="identity",
    )

    resnet_cfg = LazyConfig(ResidualNetwork)(
        in_channels=EMBED_DIM,
        out_channels=NUM_COLORS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=EMBED_DIM,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(Patchify)(
            in_features=EMBED_DIM,
            out_features=EMBED_DIM,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        out_proj_cfg=LazyConfig(Unpatchify)(
            in_features=EMBED_DIM,
            out_features=NUM_COLORS,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        norm_cfg=norm_cfg,
        block_cfg=LazyConfig(ResidualBlock)(
            pass_condition_to_sequence_mixer=True,  # Key difference vs Broadcast: route condition through mixer
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim=EMBED_DIM,
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim=DATA_DIM,
                        hidden_dim=EMBED_DIM,
                        fft_padding=FFT_PADDING,
                        use_fp16_fft=False,
                        kernel_cfg=LazyConfig(SIRENKernelND)(
                            data_dim=DATA_DIM,
                            out_dim=EMBED_DIM,
                            mlp_hidden_dim=64,
                            num_layers=3,
                            embedding_dim=64,
                            omega_0=OMEGA_0,
                            L_cache=PATCHED_RESOLUTION,
                            use_bias=True,
                            hidden_omega_0=1.0,
                            film_cfg=film_cfg,  # Inject FiLM generator into SIREN kernels
                        ),
                        mask_cfg=LazyConfig(torch.nn.Identity)(),
                        grid_type=GRID_TYPE,
                    ),
                    short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                        in_channels=3 * EMBED_DIM,
                        out_channels=3 * EMBED_DIM,
                        kernel_size=3,
                        groups=3 * EMBED_DIM,
                        padding=1,
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                    gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
                    pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=EMBED_DIM),
                    output_norm_cfg=LazyConfig(RMSNorm)(dim=EMBED_DIM),
                    qk_norm_cfg=LazyConfig(L2Norm)(),
                    use_rope=False,
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            sequence_mixer_norm_cfg=norm_cfg,
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim=EMBED_DIM,
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=norm_cfg,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
    )

    config.net = LazyConfig(ARCResNet)(
        num_tasks=NUM_TASKS,
        num_colors=NUM_COLORS,
        hidden_dim=EMBED_DIM,
        resnet_cfg=resnet_cfg,
        task_injection="film",  # Tells ARCResNet wrapper how to route task_tok
    )

    config.wandb = WandbConfig(entity="implicit-long-convs", project="nvsubquadratic", job_group="arc_film")

    return config
