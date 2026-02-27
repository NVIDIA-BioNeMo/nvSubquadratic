"""ViT-5-Small + Hyena ImageNet-1k — Apex FusedLAMB variant.

Same architecture as vit5_small_pretrain_apex.py except every ViT5Attention
block is replaced by a 2D Hyena layer (CKConvND + SIREN kernel) wrapped in
a ViT5HyenaAdapter that handles the CLS-token / register-token bookkeeping.

Key differences from the attention baseline:
- Sequence mixer: QKVSequenceMixer(Hyena) instead of ViT5Attention
- ViT5HyenaAdapter reshapes [B,T,C] ↔ [B,H',W',C] around the 2D mixer
- CLS token is updated via mean-pool of mixed patches each layer
- num_registers=0 (register tokens are an attention-specific concept)
- Positional info comes from absolute PE + SIREN kernel (no RoPE inside Hyena)
"""

import os

import torch

from experiments.datamodules.imagenet import AugmentConfig, ImageNetDataModule, MixupConfig
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


try:
    from apex.optimizers import FusedLAMB as Lamb
except ImportError:
    import warnings

    warnings.warn(
        "apex.optimizers.FusedLAMB not found — falling back to torch_optimizer.Lamb. "
        "Install Apex for fused multi-tensor LAMB (significant optimizer step speedup).",
        stacklevel=2,
    )
    from torch_optimizer import Lamb

from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Dataset ────────────────────────────────────────────────────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
FINAL_IMAGE_SIZE = 224
IMAGENET_WDS_PATH = os.environ.get("IMAGENET_WDS_PATH", "data/imagenet-wds")
HF_DATASET_NAME = "ILSVRC/imagenet-1k"

# ─── Model (ViT-5-Small + Hyena) ────────────────────────────────────────────────
HIDDEN_DIM = 384
NUM_BLOCKS = 12
PATCH_SIZE = 16
NUM_REGISTERS = 0  # Registers are attention-specific; not used with Hyena
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4
NUM_PATCHES_H = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES_W = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14

# ─── Hyena / SIREN kernel hyperparameters ────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# ─── Training recipe ────────────────────────────────────────────────────────────
BATCH_SIZE = 128  # Per-GPU batch size (reduced from 256 to fit in 24GB)
ACCUMULATE_GRAD_STEPS = 2  # Gradient accumulation steps
EPOCHS = 800
IMAGENET_TRAIN_SIZE = 1_281_167
EFFECTIVE_BATCH_SIZE = 2048  # 8 GPUs × 128 batch/GPU × 2 accum steps = 2048
ITERS_PER_EPOCH = IMAGENET_TRAIN_SIZE // EFFECTIVE_BATCH_SIZE
TOTAL_ITERATIONS = EPOCHS * ITERS_PER_EPOCH
WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / EPOCHS

LEARNING_RATE = 4e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"

NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Hyena config with Apex FusedLAMB."""
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = False
    config.compile_mode = (
        "reduce-overhead"  # Using reduce-overhead instead of max-autotune due to complex ops in Hyena FFT
    )

    # ─── Dataset ────────────────────────────────────────────────────────────
    config.dataset = LazyConfig(ImageNetDataModule)(
        data_dir=IMAGENET_WDS_PATH,
        imagefolder_dir=os.environ.get("IMAGENET_FOLDER_PATH", None),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        seed=config.seed,
        image_size=IMAGE_SIZE,
        final_image_size=FINAL_IMAGE_SIZE,
        center_crop=True,
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
    )

    # ─── Network ────────────────────────────────────────────────────────────
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
                    L_cache=NUM_PATCHES_H,
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
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
            gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            use_rope=False,
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
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
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=LazyConfig(ViT5ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
                inner_mixer_cfg=hyena_mixer_cfg,
                num_patches_h=NUM_PATCHES_H,
                num_patches_w=NUM_PATCHES_W,
                num_registers=NUM_REGISTERS,
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="gelu",
                expansion_factor=float(MLP_RATIO),
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
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
        accumulate_grad_steps=ACCUMULATE_GRAD_STEPS,
    )

    config.trainer = TrainerConfig(
        val_check_interval=1.0,
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
