"""CIFAR-10 quick-eval: flat Hyena (10 blocks, no patch merging).

Architecture
    initial p=2 → 16×16 = 256 tokens, dim=384 (matches hier stage-2 dim)
    10 blocks, no PatchMerging; GAP readout; pure layout.
    SIREN kernel: BlockDiagonalLearnableOmegaSIRENKernelND.

Training recipe
    Identical to cifar10_hyena_hier: ~100 epochs, AdamW, cosine, bf16-mixed.

The flat run is the no-merging baseline: same epoch budget, same kernel,
same compute per block — only the hierarchical structure differs.
"""

import os

import torch
from torch.optim import AdamW

from experiments.datamodules.cifar10 import CIFAR10DataModule
from experiments.default_cfg import (
    AutoResumeConfig,
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.grn import GlobalResponseNorm
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import BlockDiagonalLearnableOmegaSIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import trunc_normal_init_factory
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Dataset ──────────────────────────────────────────────────────────────────
CIFAR10_DIR = os.environ.get("CIFAR10_DIR", "./data")
IMAGE_SIZE = 32
NUM_CLASSES = 10
# dim=384 × 16×16 grid × large batch causes OOM on 22GB L4; use batch=64
# with accum=4 to keep effective batch = 256 matching the hier run.
BATCH_SIZE = 64
ACCUM_STEPS = 4

# ─── Architecture ─────────────────────────────────────────────────────────────
PATCH_SIZE = 2
NUM_PATCHES_H = IMAGE_SIZE // PATCH_SIZE  # 16
NUM_PATCHES_W = IMAGE_SIZE // PATCH_SIZE  # 16
HIDDEN_DIM = 384  # matches the last stage of the hierarchical model
NUM_BLOCKS = 10

LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4

# ─── SIREN kernel (same as hier) ───────────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_NUM_BLOCKS = 8
KERNEL_OMEGA_0_MIN = 1.0
KERNEL_OMEGA_0_MAX = 12.0
KERNEL_HIDDEN_OMEGA_0 = 1.0
KERNEL_OFF_BLOCK_SCALE = 0.1

# ─── Training (identical to hier) ─────────────────────────────────────────────
TARGET_EPOCHS = 100
CIFAR10_TRAIN_SIZE = 50_000
ITERS_PER_EPOCH = CIFAR10_TRAIN_SIZE // BATCH_SIZE
TOTAL_ITERATIONS = TARGET_EPOCHS * ITERS_PER_EPOCH
WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / TARGET_EPOCHS

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"

INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)


def get_config() -> ExperimentConfig:
    """Return the CIFAR-10 flat (no patch-merging) Hyena config."""
    kernel_cfg = LazyConfig(BlockDiagonalLearnableOmegaSIRENKernelND)(
        data_dim=2,
        out_dim=HIDDEN_DIM,
        mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_layers=KERNEL_NUM_LAYERS,
        embedding_dim=KERNEL_EMBEDDING_DIM,
        L_cache=NUM_PATCHES_H,
        use_bias=True,
        num_blocks=KERNEL_NUM_BLOCKS,
        omega_0_min=KERNEL_OMEGA_0_MIN,
        omega_0_max=KERNEL_OMEGA_0_MAX,
        schedule="linear",
        off_block_scale=KERNEL_OFF_BLOCK_SCALE,
        hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
        film_cfg=None,
    )
    mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=kernel_cfg,
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
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )
    block_cfg = LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
            inner_mixer_cfg=mixer_cfg,
            grid_w=NUM_PATCHES_W,
        ),
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim=HIDDEN_DIM,
            activation="gelu",
            expansion_factor=float(MLP_RATIO),
            bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=INIT_FN_FACTORY,
            init_method_out=INIT_FN_FACTORY,
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        hidden_dim=HIDDEN_DIM,
        layer_scale_init=LAYER_SCALE_INIT,
        drop_path_rate=DROP_PATH_RATE,
        grn_cfg=LazyConfig(GlobalResponseNorm)(dim=HIDDEN_DIM),
    )

    net = LazyConfig(ViT5ClassificationNet)(
        in_channels=3,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=0,
        block_cfg=block_cfg,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        readout="gap",
        dropout_rate=0.0,
    )

    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = False
    config.net = net

    config.dataset = LazyConfig(CIFAR10DataModule)(
        data_dir=CIFAR10_DIR,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        image_size=IMAGE_SIZE,
        mixup=0.8,
        cutmix=1.0,
    )

    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

    config.optimizer = LazyConfig(AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TOTAL_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
        accumulate_grad_steps=ACCUM_STEPS,
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=5,
        checkpoint_every_n_steps=None,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    config.wandb = WandbConfig(
        job_group="cifar10_hier_vs_flat",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    config.autoresume = AutoResumeConfig(enabled=False)
    return config
