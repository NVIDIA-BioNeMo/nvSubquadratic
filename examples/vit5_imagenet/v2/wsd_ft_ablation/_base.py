"""WSD finetuning ablation — shared base config.

All 16 ablation configs import get_config() from here and override
only the hyperparameters being ablated.  Fixed settings:

- Model: ViT-5-Small attention (12 blocks, dim 384, patch 16, 224x224)
- Pretrained: run qyjyx58f (val/acc_ema 81.81%), alias "best"
- Scheduler: WSD — 10% warmup, 70% stable, 20% linear decay
- 20 epochs, batch 256/GPU x 2 GPUs = effective 512
- EMA decay 0.99996, SoftTargetCE loss, bf16-mixed
- DALI fused data pipeline with local NVMe staging
"""

import os

import torch

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
    StartFromCheckpointConfig,
    TrainConfig,
    TrainerConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from experiments.utils.checkpointing import StripCompiledPrefix
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import trunc_normal_init, trunc_normal_init_factory


# ─── Constants ───────────────────────────────────────────────────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
FINAL_IMAGE_SIZE = 224
IMAGENET_PATH = os.environ.get("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
IMAGENET_FOLDER_PATH = os.environ.get("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")

HIDDEN_DIM = 384
NUM_BLOCKS = 12
NUM_HEADS = 6
PATCH_SIZE = 16
NUM_REGISTERS = 4
LAYER_SCALE_INIT = 1e-4
MLP_RATIO = 4
NUM_PATCHES_H = FINAL_IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES_W = FINAL_IMAGE_SIZE // PATCH_SIZE

# Weight init matching the ViT-5 reference: trunc_normal(std=0.02) for all Linear layers.
_INIT_FN = trunc_normal_init(std=0.02)
_INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)

BATCH_SIZE = 256
EPOCHS = 20
IMAGENET_TRAIN_SIZE = 1_281_167
NUM_GPUS = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * NUM_GPUS
ITERS_PER_EPOCH = IMAGENET_TRAIN_SIZE // EFFECTIVE_BATCH_SIZE
TOTAL_ITERATIONS = EPOCHS * ITERS_PER_EPOCH

PRECISION = "bf16-mixed"
NUM_WORKERS = 12

PRETRAINED_RUN_PATH = "implicit-long-convs/nvsubquadratic/qyjyx58f"  # pragma: allowlist secret


def _augment_kwargs(
    *,
    use_three_augment: bool,
    color_jitter: float,
    rand_augment: str | None,
    random_erasing_prob: float,
) -> dict:
    """Build AugmentConfig kwargs, omitting None values to avoid lazy_config placeholder confusion."""
    kwargs: dict = {
        "use_three_augment": use_three_augment,
        "color_jitter": color_jitter,
        "random_erasing_prob": random_erasing_prob,
        "random_erasing_mode": "pixel",
    }
    if rand_augment is not None:
        kwargs["rand_augment"] = rand_augment
    return kwargs


def get_config(
    *,
    lr: float = 1e-5,
    wd: float = 0.1,
    drop_path_rate: float = 0.05,
    use_three_augment: bool = False,
    rand_augment: str | None = "rand-m9-mstd0.5-inc1",
    color_jitter: float = 0.3,
    random_erasing_prob: float = 0.0,
    mixup: float = 0.8,
    cutmix: float = 1.0,
    smoothing: float = 0.1,
    epochs: int = EPOCHS,
    scheduler_name: str = "wsd",
    warmup_pct: float = 0.10,
    stable_pct: float = 0.70,
) -> ExperimentConfig:
    """Return a WSD finetuning config with the given hyperparameters."""
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = True

    # ─── Dataset ─────────────────────────────────────────────────────────
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
        eval_crop_ratio=1.0,
        mixup_cfg=LazyConfig(MixupConfig)(
            mixup=mixup,
            cutmix=cutmix,
            mixup_prob=1.0,
            mixup_switch_prob=0.5,
            mixup_mode="batch",
            smoothing=smoothing,
        ),
        augment_cfg=LazyConfig(AugmentConfig)(
            **_augment_kwargs(
                use_three_augment=use_three_augment,
                color_jitter=color_jitter,
                rand_augment=rand_augment,
                random_erasing_prob=random_erasing_prob,
            ),
        ),
        device_id=0,
        local_staging_dir=f"/scratch/{os.environ.get('USER', 'unknown')}/imagenet_dataset",
    )

    # ─── Network ─────────────────────────────────────────────────────────
    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=FINAL_IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        readout="cls",
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
                out_proj_bias=False,
                init_fn_qkv_proj=_INIT_FN,
                init_fn_out_proj=_INIT_FN,
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="gelu",
                expansion_factor=float(MLP_RATIO),
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
                init_method_in=_INIT_FN_FACTORY,
                init_method_out=_INIT_FN_FACTORY,
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            hidden_dim=HIDDEN_DIM,
            layer_scale_init=LAYER_SCALE_INIT,
            drop_path_rate=drop_path_rate,
        ),
    )

    # ─── Lightning wrapper ───────────────────────────────────────────────
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

    # ─── Optimizer ───────────────────────────────────────────────────────
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=lr,
        weight_decay=wd,
    )

    # ─── Training ────────────────────────────────────────────────────────
    total_iters = epochs * ITERS_PER_EPOCH
    config.train = TrainConfig(
        do=True,
        batch_size="${dataset.batch_size}",
        iterations=total_iters,
        grad_clip=0.0,
        precision=PRECISION,
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=1,
        checkpoint_monitor="val/acc_ema",
    )

    # ─── Scheduler ───────────────────────────────────────────────────────
    config.scheduler = SchedulerConfig(
        name=scheduler_name,
        warmup_iterations_percentage=warmup_pct,
        stable_iterations_percentage=stable_pct,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ─── Load pretrained weights (qyjyx58f, 81.81% val/acc_ema) ─────────
    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        run_path=PRETRAINED_RUN_PATH,
        alias="best",
        strict=True,
        callbacks=[LazyConfig(StripCompiledPrefix)()],
    )

    # ─── Wandb ───────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        job_group="wsd_ft_ablation",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    # ─── Auto-resume ─────────────────────────────────────────────────────
    config.autoresume = AutoResumeConfig(enabled=False)

    # ─── EMA ─────────────────────────────────────────────────────────────
    config.callbacks = [
        LazyConfig(LabeledEMAWeightAveraging)(decay=0.99996),
    ]

    return config
