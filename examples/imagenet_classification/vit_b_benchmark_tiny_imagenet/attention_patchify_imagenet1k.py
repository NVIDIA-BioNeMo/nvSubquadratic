# TODO: Add license header here

"""ImageNet-1K Classification - Attention with Patchification (ViT-B/16).

Model Size: ViT-B/16 — canonical configuration
- Hidden dim: 768
- Num blocks: 12
- Num heads: 12 (head_dim = 64)
- Patchification: patch_size=16 → 224/16 = 14×14 = 196 tokens

Purpose: Phase 0 sanity check of the full ImageNet-1K training pipeline with a
standard ViT-B/16 attention + patchify baseline.  Expected top-1 val acc after
~300k iterations (effective BS 1024): ≥ 70% (DeiT-B literature: ~81% at full
training; 300k iters is ~240 epochs which should comfortably exceed 70%).
"""

import os

import torch

from experiments.datamodules.imagenet import AugmentConfig, ImageNetDataModule, MixupConfig
from experiments.default_cfg import EMAConfig, ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.classification_resnet import ClassificationResNet


# Dataset parameters
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = NUM_CLASSES = 1_000  # ImageNet-1K classes
DATA_DIM = 2

# Training parameters
BATCH_SIZE = 128  # per GPU; 8 GPUs × 128 = 1024 effective BS (DeiT standard)
IMAGENET_PATH = os.environ.get("IMAGENET_CACHE", "data/imagenet")
HF_DATASET_NAME = "ILSVRC/imagenet-1k"
HF_DATASET_CONFIG = None
IMAGE_SIZE = 224           # standard ViT input resolution (Resize(256) -> Crop(224))
FINAL_IMAGE_SIZE = 224     # same as IMAGE_SIZE to avoid double-resize
PRECISION = "bf16-mixed"

# Model parameters — ViT-B scale (identical to TinyImageNet ablation runs)
NUM_HIDDEN_CHANNELS = 768
NUM_BLOCKS = 12
NUM_HEADS = 12  # head_dim = 768/12 = 64
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0  # DeiT-B uses 0.0 dropout; regularization via DropPath + Mixup/CutMix

# Patchification parameters — ViT-B/16
PATCH_SIZE = 16   # 224/16 = 14×14 = 196 tokens
STRIDE = 16       # non-overlapping (ViT-style)

# Optimisation parameters
TRAINING_ITERATIONS = 300_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
LEARNING_RATE = 1e-3   # DeiT-B standard for BS=1024
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
DROP_PATH_RATE = 0.1  # Stochastic depth rate (linearly increasing across blocks)


def get_config() -> ExperimentConfig:
    """Return the ImageNet-1K ViT-B/16 attention + patchify configuration."""
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    hf_token = os.environ.get("HF_TOKEN")

    config.dataset = LazyConfig(ImageNetDataModule)(
        data_dir=IMAGENET_PATH,
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
        hf_dataset_config=HF_DATASET_CONFIG,
        hf_auth_token=hf_token,
        task="classification",
        mixup_cfg=LazyConfig(MixupConfig)(
            mixup=0.8,
            cutmix=1.0,
            mixup_prob=1.0,
            mixup_switch_prob=0.5,
            mixup_mode="batch",
        ),
        # RandAugment — matching TinyImageNet benchmark strategy
        augment_cfg=LazyConfig(AugmentConfig)(
            use_three_augment=False,
            color_jitter=0.0,
            rand_augment="rand-m9-n3-mstd0.5",
        ),
    )

    config.net = LazyConfig(ClassificationResNet)(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(Patchify)(
            in_features="${net.in_channels}",
            out_features="${net.hidden_dim}",
            data_dim="${net.data_dim}",
            patch_size=PATCH_SIZE,
            stride=STRIDE,
        ),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(
            in_features="${net.hidden_dim}", out_features="${net.out_channels}"
        ),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Attention)(
                    hidden_dim="${net.hidden_dim}",
                    num_heads=NUM_HEADS,
                    apply_qk_norm=True,
                    use_rope=True,
                    is_causal=False,
                    rope_base=10000.0,
                    attn_dropout=0.0,
                ),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(
                    num_layers="${net.num_blocks}"
                ),
            ),
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="gelu",
                expansion_factor=4.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p="${net.block_cfg.dropout_cfg.p}"),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(
                    num_layers="${net.num_blocks}"
                ),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
        drop_path_rate=DROP_PATH_RATE,
    )

    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)()

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    config.ema = EMAConfig(
        enabled=True,
        decay=0.9999,
        warmup_steps=5_000,
    )

    config.wandb = WandbConfig(
        job_group="imagenet1k_vit_b_benchmark",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    return config