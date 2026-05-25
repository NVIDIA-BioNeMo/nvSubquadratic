"""Shared base config for v6_hierarchical: 4-stage Swin-style Hyena on ImageNet.

Architecture: Swin-T-like stage layout (depths [2,2,6,2], dims [96,192,384,768])
on top of the ViT-5 building blocks (ViT5ResidualBlock + ViT5HyenaAdapter).
Both pure and FiLM variants share this base; the per-stage block builder is
selected by the leaf config.

Per stage, the Hyena mixer uses ``BlockDiagonalLearnableOmegaSIRENKernelND``
(block-diagonal MLP init + learnable per-row ω₀ schedule) as the SIREN
kernel.  All other Hyena ingredients (short conv, RMSNorm, L2 QK-norm) match
``v5_patch``.

Patch sizes per stage (image 224, initial p=4):
    stage 0 : grid 56x56, dim 96
    stage 1 : grid 28x28, dim 192
    stage 2 : grid 14x14, dim 384
    stage 3 : grid  7x 7, dim 768

Token counts (T fed into each stage's mixer):
    pure          : grid_h * grid_w
    register_row  : grid_w + grid_h * grid_w   (registers as first row)

Effective batch on 8 GPUs = 2048 (same as v5_patch).
"""

from __future__ import annotations

import os
from typing import Literal

import torch

# ── Model-only imports (always safe; no GPU / cluster deps) ─────────────────
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling
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
from nvsubquadratic.utils.init import trunc_normal_init, trunc_normal_init_factory
from nvsubquadratic.utils.qk_norm import L2Norm


# ── Training-infrastructure imports (apex / DALI / Lightning) ───────────────
# These are intentionally deferred to inside get_base_config() so that
# build_hyena_hier_net() can be imported and instantiated in environments
# that don't have apex / DALI installed (CI, interactive testing, etc.).


# ─── Dataset ─────────────────────────────────────────────────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
IMAGENET_PATH = os.environ.get("IMAGENET_PATH", "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageNet/")
IMAGENET_FOLDER_PATH = os.environ.get(
    "IMAGENET_FOLDER_PATH", "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder"
)
LOCAL_STAGING_DIR = os.environ.get("LOCAL_STAGING_DIR", "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder")

# ─── Hierarchical layout (Swin-T-like) ───────────────────────────────────────
INITIAL_PATCH_SIZE = 4
STAGE_DIMS = [96, 192, 384, 768]
STAGE_DEPTHS = [2, 2, 6, 2]
NUM_STAGES = len(STAGE_DIMS)
INITIAL_GRID = IMAGE_SIZE // INITIAL_PATCH_SIZE  # 56
STAGE_GRIDS = [INITIAL_GRID // (2**i) for i in range(NUM_STAGES)]  # 56, 28, 14, 7

NUM_HEADS = 6  # for QKVSequenceMixer in_proj split
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4
NUM_REGISTERS = 4  # Used by the FiLM variant; ignored by pure variant.

# ─── SIREN kernel (BD + learnable ω₀) ────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_NUM_BLOCKS = 8
KERNEL_OMEGA_0_MIN = 1.0
KERNEL_OMEGA_0_MAX = 12.0
KERNEL_HIDDEN_OMEGA_0 = 1.0
KERNEL_OFF_BLOCK_SCALE = 0.1

# ─── FiLM conditioning (variant-only) ────────────────────────────────────────
FILM_HIDDEN_DIM = 64

# ─── Training recipe (matches v5_patch) ──────────────────────────────────────
EPOCHS = 800
IMAGENET_TRAIN_SIZE = 1_281_167
EFFECTIVE_BATCH_SIZE = 2048
NUM_GPUS = 8
ITERS_PER_EPOCH = IMAGENET_TRAIN_SIZE // EFFECTIVE_BATCH_SIZE
TOTAL_ITERATIONS = EPOCHS * ITERS_PER_EPOCH
WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / EPOCHS

LEARNING_RATE = 4e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"

NUM_WORKERS = 12

# Effective batch = NUM_GPUS * batch_per_gpu * accum_steps = 2048
# Patch-4 initial grid (3136 spatial tokens) matches v5_patch hyena_patch4's footprint.
BATCH_PER_GPU = 16
ACCUM_STEPS = 16

INIT_FN = trunc_normal_init(std=0.02)
INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)


# ─── Builders ────────────────────────────────────────────────────────────────


def _siren_kernel_cfg(hidden_dim: int, L_cache: int, film_cfg: LazyConfig | None) -> LazyConfig:
    """SIREN kernel for one stage: BlockDiagonal + learnable per-row ω₀."""
    return LazyConfig(BlockDiagonalLearnableOmegaSIRENKernelND)(
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
        film_cfg=film_cfg,
    )


def _hyena_mixer_cfg(hidden_dim: int, L_cache: int, film_cfg: LazyConfig | None) -> LazyConfig:
    """Hyena mixer for one stage (CKConvND + short conv + bilinear gates)."""
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=hidden_dim,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=hidden_dim,
                kernel_cfg=_siren_kernel_cfg(hidden_dim=hidden_dim, L_cache=L_cache, film_cfg=film_cfg),
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


def _stage_block_cfg(
    hidden_dim: int,
    grid_w: int,
    L_cache: int,
    layout: Literal["pure", "register_row"],
) -> LazyConfig:
    """Build a ViT5ResidualBlock config for one stage.

    For ``layout="register_row"``, the block owns a ``RegisterPooling`` whose
    output feeds the SIREN kernel via FiLM.  For ``layout="pure"``, no
    register pooling is attached and ``film_cfg`` is None.
    """
    if layout == "register_row":
        film_cfg = LazyConfig(KernelFiLMGenerator)(
            cond_dim=hidden_dim,
            kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
            num_film_layers=KERNEL_NUM_LAYERS - 1,
            film_hidden_dim=FILM_HIDDEN_DIM,
        )
        register_pooling_cfg = LazyConfig(RegisterPooling)(num_registers=NUM_REGISTERS)
        num_registers = NUM_REGISTERS
    else:
        film_cfg = None
        register_pooling_cfg = None
        num_registers = 0

    mixer_cfg = _hyena_mixer_cfg(hidden_dim=hidden_dim, L_cache=L_cache, film_cfg=film_cfg)
    grn_cfg = LazyConfig(GlobalResponseNorm)(dim=hidden_dim)

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
        register_pooling_cfg=register_pooling_cfg,
        num_registers=num_registers,
        register_start_idx=0,  # registers are at the head of the sequence (no CLS)
        grn_cfg=grn_cfg,
    )


def build_hyena_hier_net(layout: Literal["pure", "register_row"]) -> LazyConfig:
    """Build the full hierarchical Hyena network config.

    Args:
        layout: ``"pure"`` (no registers, no FiLM) or ``"register_row"`` (4
            registers prepended as the first 2D-grid row, conditioning every
            stage's SIREN kernel via FiLM).
    """
    # Per-stage block configs. L_cache covers both spatial dims; for the
    # register-row layout the height direction is grid + 1 (register row added).
    stage_block_cfgs = []
    for i in range(NUM_STAGES):
        grid_i = STAGE_GRIDS[i]
        L_cache_i = (grid_i + 1) if layout == "register_row" else grid_i
        stage_block_cfgs.append(
            _stage_block_cfg(
                hidden_dim=STAGE_DIMS[i],
                grid_w=grid_i,
                L_cache=L_cache_i,
                layout=layout,
            )
        )

    # Patch-merging configs between consecutive stages.
    patch_merge_cfgs = []
    for i in range(NUM_STAGES - 1):
        in_dim = STAGE_DIMS[i]
        out_dim = STAGE_DIMS[i + 1]
        grid_i = STAGE_GRIDS[i]
        patch_merge_cfgs.append(
            LazyConfig(PatchMerging)(
                in_dim=in_dim,
                out_dim=out_dim,
                grid_h=grid_i,
                grid_w=grid_i,
                norm_cfg=LazyConfig(RMSNorm)(dim=4 * in_dim, eps=1e-6),
                num_registers=NUM_REGISTERS if layout == "register_row" else 0,
                has_register_row=(layout == "register_row"),
            )
        )

    return LazyConfig(ViT5HierarchicalClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        initial_patch_size=INITIAL_PATCH_SIZE,
        stage_dims=STAGE_DIMS,
        stage_depths=STAGE_DEPTHS,
        stage_block_cfgs=stage_block_cfgs,
        patch_merge_cfgs=patch_merge_cfgs,
        norm_cfg=LazyConfig(RMSNorm)(dim=STAGE_DIMS[-1], eps=1e-6),
        layout=layout,
        num_registers=NUM_REGISTERS if layout == "register_row" else 0,
        dropout_rate=0.0,
    )


def get_base_config():
    """Return the shared base experiment config (dataset, optim, scheduler, etc.).

    The caller must set ``config.net`` (use :func:`build_hyena_hier_net`).

    Training-infrastructure imports (apex, DALI, Lightning) are deferred to
    here so that :func:`build_hyena_hier_net` is importable in environments
    without those packages (CI, interactive testing, etc.).
    """
    # ── deferred training-infra imports ──────────────────────────────────────
    from apex.optimizers import FusedLAMB as Lamb

    from experiments.callbacks.model_ema import LabeledEMAWeightAveraging
    from experiments.datamodules.dali_imagenet_fused import (
        AugmentConfig,
        DALIImageNetFusedDataModule,
        MixupConfig,
    )
    from experiments.default_cfg import (
        AutoResumeConfig,
        ExperimentConfig,
        SchedulerConfig,
        TrainConfig,
        TrainerConfig,
        WandbConfig,
    )
    from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper

    # ─────────────────────────────────────────────────────────────────────────
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.compile_compatible_fftconv = True

    config.dataset = LazyConfig(DALIImageNetFusedDataModule)(
        data_dir=IMAGENET_PATH,
        imagefolder_dir=IMAGENET_FOLDER_PATH,
        prefetch_factor=3,
        batch_size=BATCH_PER_GPU,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        seed=config.seed,
        image_size=IMAGE_SIZE,
        final_image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        drop_labels=False,
        task="classification",
        mixup_cfg=LazyConfig(MixupConfig)(
            mixup=0.8,
            cutmix=1.0,
            mixup_prob=1.0,
            mixup_switch_prob=0.5,
            mixup_mode="batch",
            smoothing=0.0,
        ),
        augment_cfg=LazyConfig(AugmentConfig)(
            use_three_augment=True,
            color_jitter=0.3,
        ),
        device_id=0,
        local_staging_dir=LOCAL_STAGING_DIR,
    )

    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

    config.optimizer = LazyConfig(Lamb)(
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
        check_val_every_n_epoch=4,
        checkpoint_every_n_steps=5000,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    config.callbacks = [LazyConfig(LabeledEMAWeightAveraging)(decay=0.99996)]
    config.trainer.checkpoint_monitor = "val/acc_ema"

    config.wandb = WandbConfig(
        job_group="v6_hierarchical",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    config.autoresume = AutoResumeConfig(enabled=False)
    return config
