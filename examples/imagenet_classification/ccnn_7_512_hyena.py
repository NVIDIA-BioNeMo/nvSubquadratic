# TODO: Add license header here

"""Config file for ImageNet classification using the shared ResNet backbone."""

import os

import torch

from experiments.datamodules._deprecated.ref_imagenet import ImageNetDataModule
from experiments.datamodules.dali_imagenet_fused import AugmentConfig, MixupConfig
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import RandomFourierKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.classification_resnet import ClassificationResNet
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# Dataset parameters
INPUT_CHANNELS = 3  # RGB images
OUTPUT_CHANNELS = NUM_CLASSES = 1_000  # ImageNet classes
DATA_DIM = 2

# Training parameters
BATCH_SIZE = 32
IMAGENET_PATH = os.environ.get("IMAGENET_CACHE", "/projects/0/prjs1161/imagenet")
HF_DATASET_NAME = "ILSVRC/imagenet-1k"
HF_DATASET_CONFIG = None
IMAGE_SIZE = 256
FINAL_IMAGE_SIZE = 64
PRECISION = "bf16-mixed"  # Tested options: "32-true", "bf16-mixed"

# Model parameters
NUM_HIDDEN_CHANNELS = 512
NUM_BLOCKS = 7
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1
GRID_TYPE = "double"
FFT_PADDING = "zero"

# Optimisation parameters
TRAINING_ITERATIONS = 600_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0


def get_config() -> ExperimentConfig:
    """Return the ImageNet classification configuration."""
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
        # Enable Augmentations
        mixup_cfg=LazyConfig(MixupConfig)(
            mixup=0.8,
            cutmix=1.0,  # Enable both mixup and cutmix
            mixup_prob=1.0,
            mixup_switch_prob=0.5,
            mixup_mode="batch",
        ),
        augment_cfg=LazyConfig(AugmentConfig)(
            use_three_augment=True,
            color_jitter=0.4,
        ),
    )

    config.net = LazyConfig(ClassificationResNet)(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.in_channels}", out_features="${net.hidden_dim}"),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.hidden_dim}", out_features="${net.out_channels}"),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim="${net.data_dim}",
                        hidden_dim="${net.hidden_dim}",
                        kernel_cfg=LazyConfig(RandomFourierKernelND)(
                            data_dim="${net.data_dim}",
                            out_dim="${net.hidden_dim}",
                            mlp_hidden_dim=64,
                            num_layers=3,
                            embedding_dim=64,
                            omega_0=100.0,
                            L_cache=32,
                            use_bias=True,
                            nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                        ),
                        mask_cfg=LazyConfig(GaussianModulationND)(
                            data_dim="${net.data_dim}",
                            num_channels="${net.hidden_dim}",
                            min_attenuation_at_step=0.1,
                            max_attenuation_at_limit=0.95,
                            init_extent=1.0,
                            parametrization="direct",
                        ),
                        grid_type=GRID_TYPE,
                        fft_padding=FFT_PADDING,
                    ),
                    short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                        in_channels="3 * ${net.hidden_dim}",
                        out_channels="3 * ${net.hidden_dim}",
                        kernel_size=3,
                        groups="3 * ${net.hidden_dim}",
                        padding=1,
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
                    pixelhyena_norm_cfg=LazyConfig(torch.nn.GroupNorm)(
                        num_groups=1,
                        num_channels="${net.hidden_dim}",
                    ),
                    qk_norm_cfg=LazyConfig(L2Norm)(),
                    use_rope=True,
                    rope_base=10000.0,
                ),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=2.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p="${net.block_cfg.dropout_cfg.p}"),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
    )

    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)()

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}", iterations=TRAINING_ITERATIONS, grad_clip=GRAD_CLIP, precision=PRECISION
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    config.wandb = WandbConfig(
        job_group="imagenet_classification",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    return config
