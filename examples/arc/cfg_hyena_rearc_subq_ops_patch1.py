"""Hyena ResNet for ARC-AGI — pixel-space (patch_size=1) ablation.

Identical to cfg_hyena_rearc_subq_ops.py except patch_size is changed from 2
to 1, so the model operates directly in pixel space.

patch_size=2 (baseline):
  - Patchify projects 2×2 pixel blocks → EMBED_DIM tokens.
  - Sequence length: (32/2)² = 256 tokens per sample.
  - CKConv spatial dims: 16×16.

patch_size=1 (this config):
  - Patchify projects each pixel individually → EMBED_DIM tokens.
  - Sequence length: 32² = 1024 tokens per sample (4× longer).
  - CKConv spatial dims: 32×32.
  - L_cache=32 → CKConvND halves to 16 internally (grid_type='single').

Motivation: ARC grids are at most 30×30 pixels with 10 discrete colours.
Working in pixel space avoids any information loss from patch aggregation and
lets the convolutional kernel attend at single-pixel granularity. The
trade-off is 4× longer sequences.

Run on 1× A6000 GPU with batch_size=64 + accumulate_grad_steps=4 to achieve
effective global BS=256 while staying within 48 GB VRAM.
"""

import math

import torch

from experiments.datamodules.arc import ARCDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.arc_wrapper import ARCWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
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
PATCH_SIZE = 1  # pixel space — no spatial downsampling
MAX_SIZE = 32
NUM_COLORS = 12
NUM_TASKS = 400

# ── Hyena ─────────────────────────────────────────────────────────────────────
DATA_DIM = 2
PATCHED_RESOLUTION = MAX_SIZE // PATCH_SIZE  # 32 (full pixel grid)
FFT_PADDING = "zero"
OMEGA_0 = 30.0
DROPOUT = 0.1

# ── Training (1× A6000, effective global BS=256 via grad accum) ───────────────
NUM_EPOCHS = 500
LEARNING_RATE = 3e-4
PLACEHOLDER = None
BATCH_SIZE = 64  # per-step batch; 4× accum → 256 effective
GRAD_ACCUM = 4
NUM_GPUS = 1
NUM_TRAINING_SAMPLES_REARC = 413_020


def get_config() -> ExperimentConfig:
    """Hyena ResNet ARC ablation: pixel-space (patch_size=1, seq_len=1024)."""
    # training_iterations uses effective BS = BATCH_SIZE × GRAD_ACCUM × NUM_GPUS
    training_iterations = math.ceil(NUM_EPOCHS * NUM_TRAINING_SAMPLES_REARC / (BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

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
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=training_iterations,
        grad_clip=1.0,
        accumulate_grad_steps=GRAD_ACCUM,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
        mode="max",
    )
    config.trainer.checkpoint_monitor = "val/exact_match"
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.trainer.precision = "bf16-mixed"

    norm_cfg = LazyConfig(RMSNorm)(dim=EMBED_DIM)

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
            patch_size=PATCH_SIZE,  # 1 → per-pixel linear projection
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
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim=EMBED_DIM,
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim=DATA_DIM,
                        hidden_dim=EMBED_DIM,
                        fft_padding=FFT_PADDING,
                        use_fp16_fft=False,
                        fft_backend="subq_ops",
                        kernel_cfg=LazyConfig(SIRENKernelND)(
                            data_dim=DATA_DIM,
                            out_dim=EMBED_DIM,
                            mlp_hidden_dim=64,
                            num_layers=3,
                            embedding_dim=64,
                            omega_0=OMEGA_0,
                            L_cache=PATCHED_RESOLUTION,  # 32; halved to 16 internally (grid_type='single')
                            use_bias=True,
                            hidden_omega_0=1.0,
                        ),
                        mask_cfg=LazyConfig(torch.nn.Identity)(),
                        grid_type="single",
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
    )

    config.wandb = WandbConfig(entity="implicit-long-convs", project="nvsubquadratic", job_group="arc_patch_ablation")

    return config
