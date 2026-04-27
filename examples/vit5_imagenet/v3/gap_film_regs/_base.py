"""GAP Hyena finetuning — shared base config for register + FiLM experiments.

All configs in this folder import get_config() from here and override only
the hyperparameters being ablated.  Fixed settings:

- Model: ViT-5-Small Hyena GAP gated (12 blocks, dim 384, patch 16, 224x224)
- Pretrained: run tcji9tfx (81.5% val/acc_ema, GAP, no FiLM, no registers)
- Scheduler: Cosine — 5-epoch warmup (20% of 25 epochs)
- 25 epochs, batch 256/GPU x 2 GPUs = effective 512
- EMA decay 0.99996, SoftTargetCE loss, bf16-mixed
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
    StartFromCheckpointConfig,
    TrainConfig,
    TrainerConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from experiments.utils.checkpointing import RemapSirenSequentialKeys, StripCompiledPrefix
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling
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

# ─── Hyena / SIREN kernel hyperparameters (must match pretrained model) ──────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# ─── FiLM conditioning defaults ─────────────────────────────────────────────────
FILM_HIDDEN_DIM = 64

# ─── Finetuning recipe ──────────────────────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS = 25
IMAGENET_TRAIN_SIZE = 1_281_167
NUM_GPUS = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * NUM_GPUS  # 512
ITERS_PER_EPOCH = IMAGENET_TRAIN_SIZE // EFFECTIVE_BATCH_SIZE
TOTAL_ITERATIONS = EPOCHS * ITERS_PER_EPOCH

WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / EPOCHS  # 0.20

PRECISION = "bf16-mixed"
NUM_WORKERS = 12

# Best GAP Hyena pretrained checkpoint (no FiLM, no registers, no CLS)
PRETRAINED_RUN_PATH = "implicit-long-convs/nvsubquadratic/tcji9tfx"  # pragma: allowlist secret


def get_config(
    *,
    lr: float = 3e-5,
    wd: float = 0.05,
    drop_path_rate: float = 0.15,
    smoothing: float = 0.1,
    fft_backend: str = "torch_fft",
    epochs: int = EPOCHS,
    run_path: str = PRETRAINED_RUN_PATH,
    mixup: float = 0.0,
    cutmix: float = 0.0,
    use_three_augment: bool = False,
    rand_augment: str = "rand-m9-mstd0.5-inc1",
    random_erasing_prob: float = 0.0,
    num_repeats: int = 1,
    layer_decay: float | None = None,
    ema_decay: float = 0.99996,
    train_do: bool = True,
    # ─── Register + FiLM parameters ──────────────────────────────────
    num_registers: int = 0,
    num_film_layers: int | None = None,
    film_after_pos_embed: bool = True,
    film_hidden_dim: int = FILM_HIDDEN_DIM,
    film_wd: float | bool = False,
    film_init_type: Literal["identity", "small_random"] = "identity",
    reg_init: Literal["trunc_normal", "zeros"] = "zeros",
    optimizer_type: Literal["adamw", "lamb"] = "adamw",
) -> ExperimentConfig:
    """Return a cosine finetuning config for the GAP Hyena model.

    Args:
        lr: Learning rate for AdamW.
        wd: Global weight decay.
        drop_path_rate: Stochastic depth rate.
        smoothing: Label smoothing.
        fft_backend: ``"torch_fft"`` or ``"subq_ops"``.
        epochs: Number of finetuning epochs.
        run_path: W&B run path for the pretrained checkpoint.
        mixup: Mixup alpha (0.0 = disabled).
        cutmix: CutMix alpha (0.0 = disabled).
        use_three_augment: If True, use three-augment instead of RandAugment.
        rand_augment: RandAugment config string.
        random_erasing_prob: Random erasing probability (0.0 = disabled).
        num_repeats: Repeated augmentation factor (1 = disabled, 3 = DeiT default).
        layer_decay: Removed. Layer-wise LR decay (LLRD) is no longer
            supported; passing a non-``None`` value raises ``RuntimeError``.
            The kwarg is kept only so legacy configs fail loudly with a
            clear message instead of silently being ignored.
        ema_decay: EMA decay rate.
        train_do: If False, skip training (validation only).
        num_registers: Number of learnable register tokens. 0 = no FiLM.
        num_film_layers: FiLM depth (None = no FiLM). 3 = full, 2 = hidden-only.
        film_after_pos_embed: If True, first FiLM pair modulates pos_embed sine output.
        film_hidden_dim: FiLM generator MLP bottleneck dimension.
        film_wd: Weight decay for FiLM params. False = global, True = no WD, float = custom.
        film_init_type: FiLM output layer init: "identity" (weights=0) or "small_random".
        reg_init: Register token initialization: "trunc_normal" or "zeros".
        optimizer_type: "adamw" for torch.optim.AdamW, "lamb" for apex FusedLAMB.
    """
    if layer_decay is not None:
        raise RuntimeError(
            "Layer-wise learning rate decay (LLRD) has been removed from "
            "BaseLightningWrapper / construct_optimizer. This config "
            f"requested layer_decay={layer_decay!r}, which is no longer "
            "supported. Drop the layer_decay argument (or use a non-LLRD "
            "config) to proceed."
        )

    has_film = num_registers > 0 and num_film_layers is not None

    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.compile_compatible_fftconv = True

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
            use_three_augment=use_three_augment,
            color_jitter=0.3,
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
            cond_dim=HIDDEN_DIM,
            kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
            num_film_layers=num_film_layers,
            film_hidden_dim=film_hidden_dim,
            no_weight_decay=film_wd,
            init_type=film_init_type,
        )

    # Keep L_cache=14 (matching pretrained). The SIREN auto-extends the grid
    # when it sees a 15-row input, preserving the original grid coordinates
    # for the patch region and extrapolating for the register row.
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
                fft_backend=fft_backend,
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
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
    )

    # Block config: add register pooling when FiLM is enabled.
    # register_start_idx is auto-computed by ViT5ClassificationNet from the token layout.
    block_kwargs = {}
    if has_film:
        block_kwargs["register_pooling_cfg"] = LazyConfig(RegisterPooling)(num_registers=num_registers)
        block_kwargs["num_registers"] = num_registers

    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=FINAL_IMAGE_SIZE,
        num_registers=num_registers if has_film else 0,
        dropout_rate=0.0,
        readout="gap",
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

    # ─── Optimizer ───────────────────────────────────────────────────────
    if optimizer_type == "lamb":
        config.optimizer = LazyConfig(Lamb)(
            params=PLACEHOLDER,
            lr=lr,
            weight_decay=wd,
        )
    else:
        config.optimizer = LazyConfig(torch.optim.AdamW)(
            params=PLACEHOLDER,
            lr=lr,
            weight_decay=wd,
        )

    # ─── Training ────────────────────────────────────────────────────────
    total_iters = epochs * ITERS_PER_EPOCH
    config.train = TrainConfig(
        do=train_do,
        batch_size="${dataset.batch_size}",
        iterations=total_iters,
        grad_clip=0.0,
        precision=PRECISION,
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=1,
        checkpoint_monitor="val/acc_ema",
    )

    # ─── Scheduler (cosine, 20% warmup) ─────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ─── Load pretrained weights ─────────────────────────────────────────
    # tcji9tfx was trained before the SIREN refactor — needs key remapping.
    # When FiLM/registers are added, new params won't be in checkpoint → strict=False.
    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        run_path=run_path,
        alias="best",
        strict=not has_film,
        callbacks=[
            LazyConfig(RemapSirenSequentialKeys)(),
            LazyConfig(StripCompiledPrefix)(),
        ],
    )

    # ─── Wandb ───────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        job_group="v3_gap_film_regs",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    # ─── Auto-resume ─────────────────────────────────────────────────────
    config.autoresume = AutoResumeConfig(enabled=False)

    # ─── Callbacks ────────────────────────────────────────────────────────
    config.callbacks = [
        LazyConfig(LabeledEMAWeightAveraging)(decay=ema_decay),
        LazyConfig(IterationSpeedCallback)(
            log_every_n_steps=10,
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
