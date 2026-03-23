"""ViT-5-Small + Hyena + GAP pretraining — v4 base with register FiLM conditioning.

Pretraining from scratch on ImageNet-1k with register-based FiLM
conditioning and compress-concat register pooling.  Based on the v2
GAP pretraining recipe (``vit5_small_pretrain_hyena_gap_apex.py``).

Key additions over v2:
- Learnable register tokens with FiLM conditioning for SIREN kernels
- ``RegisterCompressConcat`` (default) or ``RegisterPooling`` for
  aggregating register tokens into a FiLM conditioning vector
- Repeated augmentation support (DeiT-style, default ``num_repeats=3``)

Training recipe:
- Model: ViT-5-Small Hyena GAP gated (12 blocks, dim 384, patch 16, 224x224)
- 800 epochs, batch 256/GPU, effective batch 2048 (8 GPUs)
- Optimizer: Apex FusedLAMB, lr=4e-3, wd=0.05, grad_clip=1.0
- Scheduler: Cosine with 5-epoch warmup
- EMA, SoftTargetCE loss, bf16-mixed
- DALI fused data pipeline with local NVMe staging
"""

import os
from typing import Literal

import torch
from apex.optimizers import FusedLAMB as Lamb

from experiments.callbacks.film_monitor import FiLMMonitorCallback
from experiments.callbacks.iteration_speed import IterationSpeedCallback
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
from nvsubquadratic.modules.film import (
    KernelFiLMGenerator,
    RegisterCompressConcat,
    RegisterPooling,
)
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Constants ───────────────────────────────────────────────────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
FINAL_IMAGE_SIZE = 224
IMAGENET_PATH = os.environ.get("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
IMAGENET_FOLDER_PATH = os.environ.get("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")

HIDDEN_DIM = 384
NUM_BLOCKS = 12
PATCH_SIZE = 16
NUM_PATCHES_H = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES_W = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14
LAYER_SCALE_INIT = 1e-4
MLP_RATIO = 4

# ─── Hyena / SIREN kernel hyperparameters ────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# ─── FiLM conditioning defaults ─────────────────────────────────────────────────
FILM_HIDDEN_DIM = 64

# ─── Training recipe ────────────────────────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS = 800
IMAGENET_TRAIN_SIZE = 1_281_167
EFFECTIVE_BATCH_SIZE = 2048
ITERS_PER_EPOCH = IMAGENET_TRAIN_SIZE // EFFECTIVE_BATCH_SIZE
TOTAL_ITERATIONS = EPOCHS * ITERS_PER_EPOCH
WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / EPOCHS

LEARNING_RATE = 4e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"
NUM_WORKERS = 12


def get_config(
    *,
    lr: float = LEARNING_RATE,
    wd: float = WEIGHT_DECAY,
    drop_path_rate: float = 0.05,
    epochs: int = EPOCHS,
    grad_clip: float = GRAD_CLIP,
    ema_decay: float = 0.99996,
    # ─── Augmentation ────────────────────────────────────────────────
    mixup: float = 0.8,
    cutmix: float = 1.0,
    smoothing: float = 0.0,
    use_three_augment: bool = True,
    color_jitter: float = 0.3,
    rand_augment: str = "rand-m9-mstd0.5-inc1",
    random_erasing_prob: float = 0.0,
    num_repeats: int = 3,
    # ─── Register + FiLM parameters ──────────────────────────────────
    num_registers: int = 0,
    num_film_layers: int | None = None,
    film_after_pos_embed: bool = True,
    film_hidden_dim: int = FILM_HIDDEN_DIM,
    film_wd: float | bool = False,
    film_init_type: Literal["identity", "small_random"] = "identity",
    reg_init: Literal["trunc_normal", "zeros"] = "zeros",
    # ─── Register pooling mode ───────────────────────────────────────
    register_pooling_mode: Literal["weighted_avg", "compress_concat"] = "compress_concat",
    film_compression_ratio: int = 4,
    # ─── Classification readout ──────────────────────────────────────
    readout: Literal["cls", "gap", "register_concat"] = "gap",
    neck_compression_ratio: int | None = None,
) -> ExperimentConfig:
    """Return a pretraining config for ViT-5-Small Hyena GAP with FiLM registers.

    Args:
        lr: Learning rate for LAMB.
        wd: Global weight decay.
        drop_path_rate: Stochastic depth rate.
        epochs: Number of pretraining epochs.
        grad_clip: Gradient clipping norm.
        ema_decay: EMA decay rate.
        mixup: Mixup alpha (0.0 = disabled).
        cutmix: CutMix alpha (0.0 = disabled).
        smoothing: Label smoothing (0.0 for pretraining with soft targets).
        use_three_augment: If True, use three-augment instead of RandAugment.
        color_jitter: Color jitter factor.
        rand_augment: RandAugment config string (used when ``use_three_augment=False``).
        random_erasing_prob: Random erasing probability (0.0 = disabled).
        num_repeats: Repeated augmentation factor (1 = disabled, 3 = DeiT default).
        num_registers: Number of learnable register tokens. 0 = no FiLM.
        num_film_layers: FiLM depth (None = no FiLM). 3 = full, 2 = hidden-only.
        film_after_pos_embed: If True, first FiLM pair modulates pos_embed sine output.
        film_hidden_dim: FiLM generator MLP bottleneck dimension.
        film_wd: Weight decay for FiLM params. False = global, True = no WD, float = custom.
        film_init_type: FiLM output layer init: "identity" (weights=0) or "small_random".
        reg_init: Register token initialization: "trunc_normal" or "zeros".
        register_pooling_mode: How to aggregate register tokens for FiLM conditioning.
            "weighted_avg": learnable softmax-weighted average (``RegisterPooling``).
            "compress_concat": compress each register via shared linear and concatenate
            (``RegisterCompressConcat``).
        film_compression_ratio: Compression ratio for "compress_concat" register pooling.
            Each register is projected to ``HIDDEN_DIM // film_compression_ratio`` dims.
            The FiLM generator receives
            ``num_registers * (HIDDEN_DIM // film_compression_ratio)`` as ``cond_dim``.
        readout: Classification readout strategy.
            "cls": CLS token readout. "gap": global average pooling over patch tokens.
            "register_concat": compress and concatenate register tokens for classification.
        neck_compression_ratio: Compression ratio for "register_concat" readout.
            Each register is projected to ``HIDDEN_DIM // neck_compression_ratio`` dims.
            Required when ``readout="register_concat"``.
    """
    has_film = num_registers > 0 and num_film_layers is not None

    # Determine FiLM cond_dim based on pooling mode
    film_compressed_dim = HIDDEN_DIM // film_compression_ratio
    if has_film and register_pooling_mode == "compress_concat":
        film_cond_dim = num_registers * film_compressed_dim
    else:
        film_cond_dim = HIDDEN_DIM

    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = True
    config.compile_mode = "max-autotune"
    config.compile_compatible_fftconv = True
    config.num_blocks = NUM_BLOCKS

    # ─── Dataset (fused DALI + local NVMe staging) ───────────────────────
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
            mixup=mixup,
            cutmix=cutmix,
            mixup_prob=1.0,
            mixup_switch_prob=0.5,
            mixup_mode="batch",
            smoothing=smoothing,
        ),
        augment_cfg=LazyConfig(AugmentConfig)(
            use_three_augment=use_three_augment,
            color_jitter=color_jitter,
            rand_augment=rand_augment,
            random_erasing_prob=random_erasing_prob,
            random_erasing_mode="pixel",
            num_repeats=num_repeats,
        ),
        device_id=0,
        local_staging_dir=f"/scratch/{os.environ.get('USER', 'unknown')}/imagenet_dataset",
    )

    # ─── Network ─────────────────────────────────────────────────────────
    # FiLM cfg: only when registers and film_layers are configured
    film_cfg = None
    if has_film:
        film_cfg = LazyConfig(KernelFiLMGenerator)(
            cond_dim=film_cond_dim,
            kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
            num_film_layers=num_film_layers,
            film_hidden_dim=film_hidden_dim,
            no_weight_decay=film_wd,
            init_type=film_init_type,
        )

    l_cache = NUM_PATCHES_H

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
                    L_cache=l_cache,
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                    **({"film_cfg": film_cfg, "film_after_pos_embed": film_after_pos_embed} if has_film else {}),
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
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
    )

    # Block config: add register pooling when FiLM is enabled
    block_kwargs = {}
    if has_film:
        if register_pooling_mode == "compress_concat":
            block_kwargs["register_pooling_cfg"] = LazyConfig(RegisterCompressConcat)(
                num_registers=num_registers,
                hidden_dim=HIDDEN_DIM,
                compressed_dim=film_compressed_dim,
            )
        else:
            block_kwargs["register_pooling_cfg"] = LazyConfig(RegisterPooling)(num_registers=num_registers)
        block_kwargs["num_registers"] = num_registers
        block_kwargs["register_start_idx"] = 0  # GAP model: no CLS, registers at position 0

    # Compute output norm / head dimension based on readout mode
    if readout == "register_concat":
        assert neck_compression_ratio is not None, "neck_compression_ratio required for register_concat readout"
        out_norm_dim = num_registers * (HIDDEN_DIM // neck_compression_ratio)
    else:
        out_norm_dim = HIDDEN_DIM

    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=FINAL_IMAGE_SIZE,
        num_registers=num_registers if has_film else 0,
        dropout_rate=0.0,
        readout=readout,
        neck_compression_ratio=neck_compression_ratio,
        prepend_registers=True if has_film else False,
        norm_cfg=LazyConfig(RMSNorm)(dim=out_norm_dim, eps=1e-6),
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
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            hidden_dim=HIDDEN_DIM,
            layer_scale_init=LAYER_SCALE_INIT,
            drop_path_rate=drop_path_rate,
            **block_kwargs,
        ),
    )

    # Pass register init mode to network
    if has_film:
        config.net.reg_init = reg_init

    # ─── Lightning wrapper ───────────────────────────────────────────────
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

    # ─── Optimizer (Apex FusedLAMB) ──────────────────────────────────────
    config.optimizer = LazyConfig(Lamb)(
        params=PLACEHOLDER,
        lr=lr,
        weight_decay=wd,
    )

    # ─── Training ────────────────────────────────────────────────────────
    total_iters = epochs * ITERS_PER_EPOCH
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=total_iters,
        grad_clip=grad_clip,
        precision=PRECISION,
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=4,
        checkpoint_every_n_steps=5000,
        checkpoint_monitor="val/acc_ema",
    )

    # ─── Scheduler (cosine, warmup) ──────────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ─── Wandb ───────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        job_group="v4_pretrain",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    # ─── Auto-resume ─────────────────────────────────────────────────────
    config.autoresume = AutoResumeConfig(enabled=False)

    # ─── Callbacks ────────────────────────────────────────────────────────
    config.callbacks = [
        LazyConfig(LabeledEMAWeightAveraging)(decay=ema_decay),
        LazyConfig(IterationSpeedCallback)(
            log_every_n_steps=50,
            batch_size_per_gpu=BATCH_SIZE,
        ),
    ]
    if has_film:
        config.callbacks.append(
            LazyConfig(FiLMMonitorCallback)(
                log_every_n_steps=50,
                num_film_layers=num_film_layers,
                film_on_pos_embed=film_after_pos_embed,
                film_after_pos_embed=film_after_pos_embed,
            )
        )

    return config
