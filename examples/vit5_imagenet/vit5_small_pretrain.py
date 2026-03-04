"""ViT-5-Small ImageNet-1k Pretraining Configuration.

Architecture (from ViT-5 paper, Table 4):
- 12 layers, dim 384, 6 heads, 4 registers, ~22M params
- RMSNorm, LayerScale (init 1e-4), QK-Norm (RMSNorm)
- APE + 2D RoPE, register tokens with high-frequency RoPE (theta=100)
- GeLU MLP (ratio 4), no QKV bias
- Patch size 16 -> 224/16 = 14x14 = 196 tokens

Pretraining recipe (from ViT-5 paper, Table 12):
- LAMB optimizer, lr 4e-3, weight decay 0.05, grad clip 1.0
- Batch 2048, 800 epochs, cosine schedule, 5 warmup epochs
- 3-Augment, Color Jitter 0.3, Mixup 0.8, CutMix 1.0
- BCE loss, no label smoothing, stochastic depth 0.05
- Input 224x224
"""

import os

import torch

# NOTE: Apex FusedLAMB is faster but its optimizer state is incompatible with
# torch_optimizer.Lamb for checkpoint resume.  Use Lamb for runs that need to
# resume from a Lamb checkpoint; switch to FusedLAMB only for fresh runs.
#
# try:
#     from apex.optimizers import FusedLAMB as Lamb
# except ImportError:
#     import warnings
#     warnings.warn(
#         "apex.optimizers.FusedLAMB not found — falling back to torch_optimizer.Lamb. "
#         "Install Apex for fused multi-tensor LAMB (significant optimizer step speedup).",
#         stacklevel=2,
#     )
#     from torch_optimizer import Lamb
from torch_optimizer import Lamb

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
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


# ─── Dataset ────────────────────────────────────────────────────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
FINAL_IMAGE_SIZE = 224
IMAGENET_PATH = os.environ.get("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
IMAGENET_FOLDER_PATH = os.environ.get("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")
HF_DATASET_NAME = "ILSVRC/imagenet-1k"

# ─── Model (ViT-5-Small) ────────────────────────────────────────────────────────
HIDDEN_DIM = 384
NUM_BLOCKS = 12
NUM_HEADS = 6
PATCH_SIZE = 16
NUM_REGISTERS = 4
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4
NUM_PATCHES_H = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES_W = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14

# ─── Training recipe ────────────────────────────────────────────────────────────
BATCH_SIZE = 256  # per-GPU; 256 * 8 GPUs = 2048 effective
EPOCHS = 800
IMAGENET_TRAIN_SIZE = 1_281_167
EFFECTIVE_BATCH_SIZE = 2048
ITERS_PER_EPOCH = IMAGENET_TRAIN_SIZE // EFFECTIVE_BATCH_SIZE
TOTAL_ITERATIONS = EPOCHS * ITERS_PER_EPOCH  # ~500,000
WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / EPOCHS  # 5/800 ≈ 0.00625

LEARNING_RATE = 4e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"

NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small ImageNet-1k pretraining configuration."""
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = True
    config.compile_mode = "max-autotune"
    hf_token = os.environ.get("HF_TOKEN")

    # ─── Dataset ────────────────────────────────────────────────────────────
    config.dataset = LazyConfig(ImageNetDataModule)(
        data_dir=IMAGENET_PATH,
        imagefolder_dir=IMAGENET_FOLDER_PATH,
        prefetch_factor=2,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        seed=config.seed,
        image_size=IMAGE_SIZE,
        final_image_size=FINAL_IMAGE_SIZE,
        center_crop=True,
        num_classes=NUM_CLASSES,
        drop_labels=False,
        hf_dataset_name=HF_DATASET_NAME,
        hf_dataset_config=None,
        hf_auth_token=hf_token,
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
            sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                num_patches_h=NUM_PATCHES_H,
                num_patches_w=NUM_PATCHES_W,
                num_registers=NUM_REGISTERS,
                qk_norm=LazyConfig(RMSNorm)(dim=HIDDEN_DIM // NUM_HEADS, eps=1e-6),
                rope_base=10000.0,
                reg_rope_base=100.0,
                attn_dropout=0.0,
                proj_dropout=0.0,
                qkv_bias=False,
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
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

    # ─── Optimizer ────────────────────────────────────────────────────────
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
    )

    config.trainer = TrainerConfig(
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

    # ─── Auto-resume ─────────────────────────────────────────────────────────
    config.autoresume = AutoResumeConfig(
        enabled=False,
    )

    return config
