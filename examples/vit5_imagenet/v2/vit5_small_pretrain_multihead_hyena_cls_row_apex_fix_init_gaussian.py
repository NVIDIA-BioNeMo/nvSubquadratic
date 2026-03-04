"""ViT-5-Small + Multi-Head Hyena ImageNet-1k — CLS-row variant, Apex FusedLAMB.

Based on v2/vit5_small_pretrain_hyena_cls_row_apex.py but replaces the depthwise
CKConvND with CKConvMultiheadND (dense within-head channel mixing).

Key differences from v2/vit5_small_pretrain_hyena_cls_row_apex.py:
- CKConvMultiheadND replaces CKConvND: each head performs dense [head_dim x
  head_dim] channel mixing within heads while keeping heads isolated.
- SIREN kernel out_dim = NUM_HEADS * HEAD_DIM * HEAD_DIM (dense kernel per head)
  instead of HIDDEN_DIM (depthwise).
- NUM_HEADS=6, HEAD_DIM=64 (consistent with the ViT-5-Small attention variant).
- PerHeadRMSNorm for QK normalization (each head normalized independently).
- SiLU gate nonlinearity + output RMSNorm (same as other v2 configs).
- CLS-row architecture: CLS + 13 registers as extra row → 15×14 grid.
- DALI fused data pipeline with local NVMe staging.
"""

import os

import torch

from experiments.datamodules.dali_imagenet_fused import DALIImageNetFusedDataModule
from experiments.datamodules.imagenet import AugmentConfig, MixupConfig
from experiments.default_cfg import AutoResumeConfig, ExperimentConfig, SchedulerConfig, TrainConfig, TrainerConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig

from apex.optimizers import FusedLAMB as Lamb

from nvsubquadratic.modules.ckconv_multihead_nd import CKConvMultiheadND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import PerHeadRMSNorm, RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet

# ─── Dataset ────────────────────────────────────────────────────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
FINAL_IMAGE_SIZE = 224
IMAGENET_PATH = os.environ.get("IMAGENET_PATH", "/ivi/zfs/s0/original_homes/dknigge/imagenet_folder")
IMAGENET_FOLDER_PATH = os.environ.get("IMAGENET_FOLDER_PATH", "/ivi/zfs/s0/original_homes/dknigge/imagenet_folder")
LOCAL_STAGING_DIR = "/local_scratch/dknigge/imagenet_dataset"

# ─── Model (ViT-5-Small + Multi-Head Hyena, CLS-row) ────────────────────────────
HIDDEN_DIM = 384
NUM_BLOCKS = 12
NUM_HEADS = 6
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 64
PATCH_SIZE = 16
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4
NUM_PATCHES_H = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES_W = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14
NUM_REGISTERS = NUM_PATCHES_W - 1  # 13 — fills the extra row: [CLS, regs, patches] → (H'+1)×W' grid

# ─── Multi-Head Hyena / SIREN kernel hyperparameters ─────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0
KERNEL_OUT_DIM = NUM_HEADS * HEAD_DIM * HEAD_DIM  # dense kernel per head

# ─── Training recipe ────────────────────────────────────────────────────────────
EFFECTIVE_BATCH_SIZE = 2048
BATCH_SIZE = 128
NUM_GPUS = 8
NUM_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // (BATCH_SIZE * NUM_GPUS)  # 2: gradient accumulation to achieve effective batch size of 2048
EPOCHS = 800
IMAGENET_TRAIN_SIZE = 1_281_167
ITERS_PER_EPOCH = IMAGENET_TRAIN_SIZE // EFFECTIVE_BATCH_SIZE
TOTAL_ITERATIONS = EPOCHS * ITERS_PER_EPOCH
WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / EPOCHS

LEARNING_RATE = 4e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"

NUM_WORKERS = 12


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Multi-Head Hyena CLS-row config with Apex FusedLAMB."""
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"  # "max-autotune" can cause OOM due to large memory usage of some fused kernels (e.g., FusedLAMB) during autotuning; this mode disables CUDA graph capture during autotuning to reduce memory usage and avoid OOM.
    # ─── Dataset (fused DALI + local NVMe staging) ────────────────────────
    config.dataset = LazyConfig(DALIImageNetFusedDataModule)(
        data_dir=IMAGENET_PATH,
        imagefolder_dir=IMAGENET_FOLDER_PATH,
        prefetch_factor=3,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        seed=config.seed,
        image_size=IMAGE_SIZE,
        final_image_size=FINAL_IMAGE_SIZE,
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

    # ─── Network ────────────────────────────────────────────────────────────
    hyena_mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvMultiheadND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2,
                    out_dim=KERNEL_OUT_DIM,
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache=NUM_PATCHES_H + 1,  # 15: grid is (H'+1)×W' due to the extra CLS row.
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                ),
                mask_cfg=LazyConfig(GaussianModulationND)(
                    data_dim=2,
                    num_channels=KERNEL_OUT_DIM,
                    min_std=0.02,
                    max_std=1.5,
                    init_std_low=0.05,
                    init_std_high=1.2,
                    parametrization="direct",
                ),
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
            qk_norm_cfg=LazyConfig(PerHeadRMSNorm)(num_heads=NUM_HEADS, head_dim=HEAD_DIM, eps=1e-6),
            use_rope=False,
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        ),
    )

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
        block_cfg=LazyConfig(ViT5ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
                inner_mixer_cfg=hyena_mixer_cfg,
                grid_w=NUM_PATCHES_W,
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="gelu",
                expansion_factor=float(MLP_RATIO),
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            hidden_dim=HIDDEN_DIM,
            layer_scale_init=LAYER_SCALE_INIT,
            drop_path_rate=DROP_PATH_RATE,
        ),
    )

    # ─── Lightning wrapper ──────────────────────────────────────────────────
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(use_bce_loss=True)

    # ─── Optimizer (Apex FusedLAMB) ─────────────────────────────────────────
    config.optimizer = LazyConfig(Lamb)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ─── Training ───────────────────────────────────────────────────────────
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TOTAL_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
        accumulate_grad_steps=NUM_ACCUM_STEPS,
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=4,
        checkpoint_every_n_steps=5000,
    )

    # ─── Scheduler ──────────────────────────────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ─── Wandb ──────────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        job_group="vit5_imagenet_pretrain",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    # ─── Auto-resume ────────────────────────────────────────────────────────
    config.autoresume = AutoResumeConfig(
        enabled=False,
    )

    return config
