"""Shared builders for the CIFAR-10 patch-size × hierarchy ablation.

Images are upscaled to 64×64 so that patch_size=4 yields a 16×16 initial
grid, which is deep enough to support all 4 Swin-T / VMamba-T stages.

Runs
----
flat  × {p4, p8, p16} — ViT5ClassificationNet, 10 blocks, dim=384, no merging
hier  × {p4, p8, p16} — ViT5HierarchicalClassificationNet, patch merging

Stage layout (Swin-T / VMamba-T: depths=[2,2,6,2], dims=[96,192,384,768]):

  patch_size=4  → initial grid 16×16
    flat : 10 blocks, dim=384
    hier : 4 stages 16→8→4→2  depths=[2,2,6,2]  dims=[96,192,384,768]  ← full Swin-T

  patch_size=8  → initial grid 8×8
    flat : 10 blocks, dim=384
    hier : 3 stages 8→4→2     depths=[2,2,6]    dims=[96,192,384]      ← Swin stages 1-3
           (4th stage would yield 1×1, hitting L_cache≥2 floor)

  patch_size=16 → initial grid 4×4
    flat : 10 blocks, dim=384
    hier : 2 stages 4→2       depths=[6,2]      dims=[384,768]         ← Swin stages 3-4
           (3rd stage would yield 1×1)

Training (all runs)
-------------------
  200 epochs, batch=64, no grad-accum, AdamW lr=3e-4, cosine, bf16-mixed.
  W&B job_group: "cifar10_patch_ablation_64px"
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
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.networks.vit5_hierarchical_classification import (
    ViT5HierarchicalClassificationNet,
)
from nvsubquadratic.utils.init import trunc_normal_init_factory
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Dataset ──────────────────────────────────────────────────────────────────
CIFAR10_DIR = os.environ.get("CIFAR10_DIR", "./data")
IMAGE_SIZE = 64  # upscaled from 32 to allow 4 Swin stages with patch_size=4
NUM_CLASSES = 10

# ─── Shared training hyper-parameters ─────────────────────────────────────────
BATCH_SIZE = 64
TARGET_EPOCHS = 200
CIFAR10_TRAIN_SIZE = 50_000
ITERS_PER_EPOCH = CIFAR10_TRAIN_SIZE // BATCH_SIZE  # 781
TOTAL_ITERATIONS = TARGET_EPOCHS * ITERS_PER_EPOCH  # 156_200
WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / TARGET_EPOCHS  # 0.025

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"

LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4

# ─── SIREN kernel ──────────────────────────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_NUM_BLOCKS = 8  # divides 96, 192, 384, 768 cleanly
KERNEL_OMEGA_0_MIN = 1.0
KERNEL_OMEGA_0_MAX = 12.0
KERNEL_HIDDEN_OMEGA_0 = 1.0
KERNEL_OFF_BLOCK_SCALE = 0.1

# ─── Flat architecture ────────────────────────────────────────────────────────
FLAT_HIDDEN_DIM = 384
FLAT_NUM_BLOCKS = 10

# ─── Hier architecture — Swin-T / VMamba-T stage maps ────────────────────────
# p=4: 4 stages 16→8→4→2 — full Swin-T
HIER_P4_STAGE_DIMS = [96, 192, 384, 768]
HIER_P4_STAGE_DEPTHS = [2, 2, 6, 2]

# p=8: 3 stages 8→4→2 — Swin stages 1-3
HIER_P8_STAGE_DIMS = [96, 192, 384]
HIER_P8_STAGE_DEPTHS = [2, 2, 6]

# p=16: 2 stages 4→2 — Swin stages 3-4
# A 4×4 initial grid corresponds to the output of Swin stage 2 in the p=4 hierarchy.
HIER_P16_STAGE_DIMS = [384, 768]
HIER_P16_STAGE_DEPTHS = [6, 2]

INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)


# ─── Block builder ────────────────────────────────────────────────────────────


def _build_block(hidden_dim: int, grid_w: int) -> LazyConfig:
    """Return a ViT5ResidualBlock LazyConfig for the given (hidden_dim, grid_w)."""
    kernel_cfg = LazyConfig(BlockDiagonalLearnableOmegaSIRENKernelND)(
        data_dim=2,
        out_dim=hidden_dim,
        mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_layers=KERNEL_NUM_LAYERS,
        embedding_dim=KERNEL_EMBEDDING_DIM,
        L_cache=grid_w,
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


# ─── Shared training config skeleton ──────────────────────────────────────────


def _base_config() -> ExperimentConfig:
    """Return an ExperimentConfig pre-filled with the shared training settings."""
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = False

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
        job_group="cifar10_patch_ablation_64px",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    config.autoresume = AutoResumeConfig(enabled=False)
    return config


# ─── Public config builders ───────────────────────────────────────────────────


def build_flat_config(patch_size: int) -> ExperimentConfig:
    """Flat baseline (no patch merging) for the given patch_size ∈ {4, 8, 16}."""
    assert patch_size in {4, 8, 16}, f"Unsupported patch_size={patch_size}"
    grid = IMAGE_SIZE // patch_size  # 16, 8, or 4

    block_cfg = _build_block(hidden_dim=FLAT_HIDDEN_DIM, grid_w=grid)

    net = LazyConfig(ViT5ClassificationNet)(
        in_channels=3,
        num_classes=NUM_CLASSES,
        hidden_dim=FLAT_HIDDEN_DIM,
        num_blocks=FLAT_NUM_BLOCKS,
        patch_size=patch_size,
        image_size=IMAGE_SIZE,
        num_registers=0,
        block_cfg=block_cfg,
        norm_cfg=LazyConfig(RMSNorm)(dim=FLAT_HIDDEN_DIM, eps=1e-6),
        readout="gap",
        dropout_rate=0.0,
    )

    config = _base_config()
    config.net = net
    return config


def build_hier_config(patch_size: int) -> ExperimentConfig:
    """Hierarchical config (with patch merging) for the given patch_size ∈ {4, 8, 16}.

    Stage layout (Swin-T / VMamba-T consistent, images upscaled to 64×64):
      p=4 : 4 stages 16→8→4→2  depths=[2,2,6,2]  dims=[96,192,384,768]  ← full Swin-T
      p=8 : 3 stages 8→4→2     depths=[2,2,6]    dims=[96,192,384]      ← Swin stages 1-3
      p=16: 2 stages 4→2       depths=[6,2]      dims=[384,768]         ← Swin stages 3-4
    """
    assert patch_size in {4, 8, 16}, f"Unsupported patch_size={patch_size}"

    if patch_size == 4:
        stage_dims = HIER_P4_STAGE_DIMS
        stage_depths = HIER_P4_STAGE_DEPTHS
    elif patch_size == 8:
        stage_dims = HIER_P8_STAGE_DIMS
        stage_depths = HIER_P8_STAGE_DEPTHS
    else:  # patch_size == 16
        stage_dims = HIER_P16_STAGE_DIMS
        stage_depths = HIER_P16_STAGE_DEPTHS

    num_stages = len(stage_dims)
    initial_grid = IMAGE_SIZE // patch_size
    stage_grids = [initial_grid // (2**i) for i in range(num_stages)]

    stage_block_cfgs = [_build_block(hidden_dim=stage_dims[i], grid_w=stage_grids[i]) for i in range(num_stages)]
    patch_merge_cfgs = [
        LazyConfig(PatchMerging)(
            in_dim=stage_dims[i],
            out_dim=stage_dims[i + 1],
            grid_h=stage_grids[i],
            grid_w=stage_grids[i],
            norm_cfg=LazyConfig(RMSNorm)(dim=4 * stage_dims[i], eps=1e-6),
            has_register_row=False,
        )
        for i in range(num_stages - 1)
    ]

    net = LazyConfig(ViT5HierarchicalClassificationNet)(
        in_channels=3,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        initial_patch_size=patch_size,
        stage_dims=stage_dims,
        stage_depths=stage_depths,
        stage_block_cfgs=stage_block_cfgs,
        patch_merge_cfgs=patch_merge_cfgs,
        norm_cfg=LazyConfig(RMSNorm)(dim=stage_dims[-1], eps=1e-6),
        layout="pure",
        num_registers=0,
        dropout_rate=0.0,
    )

    config = _base_config()
    config.net = net
    return config
