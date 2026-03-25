"""Shared base config for the v5_patch ablation: Hyena vs Attention across patch sizes.

Provides parameterized builder functions that derive all patch-size-dependent
quantities (num_patches, registers, L_cache, grid_w, batch_size, accum_steps)
from a single ``patch_size`` argument.

Patch sizes: 16, 8, 4, 2, 1  (image 224x224)
Effective batch size is always 2048 on 8 GPUs.
"""

import os

import torch
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
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling
from nvsubquadratic.modules.grn import GlobalResponseNorm
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import trunc_normal_init, trunc_normal_init_factory
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Dataset ─────────────────────────────────────────────────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
IMAGENET_PATH = os.environ.get("IMAGENET_PATH", "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageNet/")
IMAGENET_FOLDER_PATH = os.environ.get(
    "IMAGENET_FOLDER_PATH", "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder"
)
LOCAL_STAGING_DIR = os.environ.get(
    "LOCAL_STAGING_DIR", "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder"
)

# ─── Model (ViT-5-Small) ─────────────────────────────────────────────────────
HIDDEN_DIM = 384
NUM_BLOCKS = 12
NUM_HEADS = 6
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 64
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4

# ─── SIREN kernel hyperparameters ─────────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# ─── FiLM conditioning ────────────────────────────────────────────────────────
FILM_HIDDEN_DIM = 64

# ─── Training recipe ─────────────────────────────────────────────────────────
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

INIT_FN = trunc_normal_init(std=0.02)
INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)

# ─── Per-patch-size batch configuration ───────────────────────────────────────
# Effective batch = NUM_GPUS * batch_per_gpu * accum_steps = 2048
# Smaller patches → more tokens → smaller per-GPU batch.
PATCH_BATCH_CONFIG = {
    16: {"batch_per_gpu": 256, "accum_steps": 1},
    8: {"batch_per_gpu": 64, "accum_steps": 4},
    4: {"batch_per_gpu": 16, "accum_steps": 16},
    2: {"batch_per_gpu": 4, "accum_steps": 64},
    1: {"batch_per_gpu": 1, "accum_steps": 256},
}


# ─── Builder functions ────────────────────────────────────────────────────────


def _make_block_cfg(sequence_mixer_cfg: LazyConfig, **kwargs) -> LazyConfig:
    """Build a ViT5ResidualBlock config with the given sequence mixer."""
    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=sequence_mixer_cfg,
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
        **kwargs,
    )


def get_base_config(patch_size: int) -> ExperimentConfig:
    """Return the shared base config parameterized by patch_size.

    Sets batch_size and accumulate_grad_steps to maintain effective batch = 2048
    on 8 GPUs. Callers must set ``config.net``.
    """
    batch_cfg = PATCH_BATCH_CONFIG[patch_size]
    batch_per_gpu = batch_cfg["batch_per_gpu"]
    accum_steps = batch_cfg["accum_steps"]

    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = True
    # CUDA graphs are incompatible with gradient accumulation
    config.compile_mode = "max-autotune" if accum_steps == 1 else "max-autotune-no-cudagraphs"
    config.compile_compatible_fftconv = True

    # ─── Dataset (fused DALI + local staging) ─────────────────────────────
    config.dataset = LazyConfig(DALIImageNetFusedDataModule)(
        data_dir=IMAGENET_PATH,
        imagefolder_dir=IMAGENET_FOLDER_PATH,
        prefetch_factor=3,
        batch_size=batch_per_gpu,
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

    # ─── Lightning wrapper ────────────────────────────────────────────────
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

    # ─── Optimizer (Apex FusedLAMB) ───────────────────────────────────────
    config.optimizer = LazyConfig(Lamb)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ─── Training ─────────────────────────────────────────────────────────
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TOTAL_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
        accumulate_grad_steps=accum_steps,
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=4,
        checkpoint_every_n_steps=5000,
    )

    # ─── Scheduler ────────────────────────────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ─── EMA ──────────────────────────────────────────────────────────────
    config.callbacks = [LazyConfig(LabeledEMAWeightAveraging)(decay=0.99996)]
    config.trainer.checkpoint_monitor = "val/acc_ema"

    # ─── Wandb ────────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        job_group="v5_patch_ablation",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    # ─── Auto-resume ──────────────────────────────────────────────────────
    config.autoresume = AutoResumeConfig(enabled=False)

    return config


def build_attention_net(patch_size: int) -> LazyConfig:
    """Build ViT5ClassificationNet with standard attention for a given patch_size."""
    num_patches_h = IMAGE_SIZE // patch_size
    num_patches_w = IMAGE_SIZE // patch_size
    num_registers = num_patches_w - 1  # Scale registers with grid (matches Hyena CLS-row)

    return LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=patch_size,
        image_size=IMAGE_SIZE,
        num_registers=num_registers,
        dropout_rate=0.0,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=_make_block_cfg(
            sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                num_patches_h=num_patches_h,
                num_patches_w=num_patches_w,
                num_registers=num_registers,
                qk_norm=LazyConfig(RMSNorm)(dim=HEAD_DIM, eps=1e-6),
                rope_base=10000.0,
                reg_rope_base=100.0,
                attn_dropout=0.0,
                proj_dropout=0.0,
                qkv_bias=False,
                out_proj_bias=False,
                init_fn_qkv_proj=INIT_FN,
                init_fn_out_proj=INIT_FN,
            ),
        ),
    )


def build_hyena_net(patch_size: int) -> LazyConfig:
    """Build ViT5ClassificationNet with Hyena CLS-row + FiLM + GRN for a given patch_size."""
    num_patches_h = IMAGE_SIZE // patch_size
    num_patches_w = IMAGE_SIZE // patch_size
    num_registers = num_patches_w - 1  # CLS-row: fill one row with [CLS, regs]

    film_cfg = LazyConfig(KernelFiLMGenerator)(
        cond_dim=HIDDEN_DIM,
        kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_film_layers=KERNEL_NUM_LAYERS - 1,
        film_hidden_dim=FILM_HIDDEN_DIM,
    )

    mixer_cfg = LazyConfig(QKVSequenceMixer)(
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
                    L_cache=num_patches_h + 1,  # CLS-row adds 1 row
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                    film_cfg=film_cfg,
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
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            use_rope=False,
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )

    register_pooling_cfg = LazyConfig(RegisterPooling)(num_registers=num_registers)
    grn_cfg = LazyConfig(GlobalResponseNorm)(dim=HIDDEN_DIM)

    block_cfg = _make_block_cfg(
        sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
            inner_mixer_cfg=mixer_cfg,
            grid_w=num_patches_w,
        ),
        register_pooling_cfg=register_pooling_cfg,
        num_registers=num_registers,
        register_start_idx=1,
        grn_cfg=grn_cfg,
    )

    return LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=patch_size,
        image_size=IMAGE_SIZE,
        num_registers=num_registers,
        dropout_rate=0.0,
        use_cls_token=True,
        prepend_registers=True,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=block_cfg,
    )
