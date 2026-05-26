"""CIFAR-10 quick-eval: hierarchical Hyena (3 stages, with patch merging).

Architecture
    initial p=2 → 16×16 grid → dims [96, 192, 384], depths [2, 2, 6]
    PatchMerging between stages (16→8→4); GAP readout; pure layout (no FiLM).
    SIREN kernel: BlockDiagonalLearnableOmegaSIRENKernelND.

Training recipe
    ~100 epochs  (50 000 × 100 / 256 ≈ 19 500 steps)
    AdamW lr=3e-4, cosine schedule, 5-epoch warmup
    bf16-mixed, single GPU
    Standard augmentation: RandomCrop(32,pad=4) + RandomHorizontalFlip

This is a diagnostic run to compare classification performance against the
flat (no patch-merging) baseline at the same compute and epoch budget.
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
from nvsubquadratic.modules.patch_merging import PatchMerging
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_hierarchical_classification import (
    ViT5HierarchicalClassificationNet,
)
from nvsubquadratic.utils.init import trunc_normal_init_factory
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Dataset ──────────────────────────────────────────────────────────────────
CIFAR10_DIR = os.environ.get("CIFAR10_DIR", "./data")
IMAGE_SIZE = 32
NUM_CLASSES = 10
BATCH_SIZE = 256

# ─── Architecture ─────────────────────────────────────────────────────────────
INITIAL_PATCH_SIZE = 2
STAGE_DIMS = [96, 192, 384]
STAGE_DEPTHS = [2, 2, 6]
NUM_STAGES = len(STAGE_DIMS)
INITIAL_GRID = IMAGE_SIZE // INITIAL_PATCH_SIZE  # 16
STAGE_GRIDS = [INITIAL_GRID // (2**i) for i in range(NUM_STAGES)]  # 16, 8, 4

LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4

# ─── SIREN kernel ──────────────────────────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_NUM_BLOCKS = 8
KERNEL_OMEGA_0_MIN = 1.0
KERNEL_OMEGA_0_MAX = 12.0
KERNEL_HIDDEN_OMEGA_0 = 1.0
KERNEL_OFF_BLOCK_SCALE = 0.1

# ─── Training ──────────────────────────────────────────────────────────────────
TARGET_EPOCHS = 100
CIFAR10_TRAIN_SIZE = 50_000
ITERS_PER_EPOCH = CIFAR10_TRAIN_SIZE // BATCH_SIZE  # 195
TOTAL_ITERATIONS = TARGET_EPOCHS * ITERS_PER_EPOCH  # ~19 500
WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / TARGET_EPOCHS

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"

INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)


def _build_block(hidden_dim: int, grid_w: int) -> LazyConfig:
    L_cache = grid_w
    kernel_cfg = LazyConfig(BlockDiagonalLearnableOmegaSIRENKernelND)(
        data_dim=2,
        out_dim=hidden_dim,
        mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_layers=KERNEL_NUM_LAYERS,
        embedding_dim=KERNEL_EMBEDDING_DIM,
        L_cache=L_cache,
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
        hidden_dim=hidden_dim,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=hidden_dim,
                kernel_cfg=kernel_cfg,
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * hidden_dim,
                out_channels=3 * hidden_dim,
                kernel_size=3,
                groups=3 * hidden_dim,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            output_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )
    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
            inner_mixer_cfg=mixer_cfg,
            grid_w=grid_w,
        ),
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim=hidden_dim,
            activation="gelu",
            expansion_factor=float(MLP_RATIO),
            bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=INIT_FN_FACTORY,
            init_method_out=INIT_FN_FACTORY,
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
        hidden_dim=hidden_dim,
        layer_scale_init=LAYER_SCALE_INIT,
        drop_path_rate=DROP_PATH_RATE,
        grn_cfg=LazyConfig(GlobalResponseNorm)(dim=hidden_dim),
    )


def get_config() -> ExperimentConfig:
    """Return the CIFAR-10 hierarchical Hyena config."""
    stage_block_cfgs = [_build_block(hidden_dim=STAGE_DIMS[i], grid_w=STAGE_GRIDS[i]) for i in range(NUM_STAGES)]
    patch_merge_cfgs = [
        LazyConfig(PatchMerging)(
            in_dim=STAGE_DIMS[i],
            out_dim=STAGE_DIMS[i + 1],
            grid_h=STAGE_GRIDS[i],
            grid_w=STAGE_GRIDS[i],
            norm_cfg=LazyConfig(RMSNorm)(dim=4 * STAGE_DIMS[i], eps=1e-6),
            has_register_row=False,
        )
        for i in range(NUM_STAGES - 1)
    ]

    net = LazyConfig(ViT5HierarchicalClassificationNet)(
        in_channels=3,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        initial_patch_size=INITIAL_PATCH_SIZE,
        stage_dims=STAGE_DIMS,
        stage_depths=STAGE_DEPTHS,
        stage_block_cfgs=stage_block_cfgs,
        patch_merge_cfgs=patch_merge_cfgs,
        norm_cfg=LazyConfig(RMSNorm)(dim=STAGE_DIMS[-1], eps=1e-6),
        layout="pure",
        num_registers=0,
        dropout_rate=0.0,
    )

    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = False  # keep off for quick iteration
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
        accumulate_grad_steps=1,
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
