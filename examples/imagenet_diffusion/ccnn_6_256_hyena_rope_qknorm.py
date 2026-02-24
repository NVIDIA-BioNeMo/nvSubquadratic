# TODO: Add license header here

"""Config file for ImageNet diffusion using the shared ResNet backbone."""

import os

import torch

from experiments.datamodules.imagenet import ImageNetDataModule
from experiments.default_cfg import (
    DiffusionConfig,
    DiffusionExperimentConfig,
    EMAConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.diffusion_wrapper import DiffusionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import RandomFourierKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import AdaLNZeroResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.qk_norm import L2Norm


# Dataset parameters
INPUT_CHANNELS = 3  # RGB images
OUTPUT_CHANNELS = 3  # Reconstruct RGB
NUM_CLASSES = 1_000
DATA_DIM = 2

# Training parameters
BATCH_SIZE = 64
IMAGENET_PATH = os.environ.get("IMAGENET_CACHE", "/projects/0/prjs1161/imagenet")
HF_DATASET_NAME = "imagenet-1k"
HF_DATASET_CONFIG = None
IMAGE_SIZE = 256
FINAL_IMAGE_SIZE = 64
PRECISION = "bf16-mixed"  # Tested options: "32-true", "bf16-mixed"

# Model parameters
NUM_HIDDEN_CHANNELS = 256
NUM_BLOCKS = 6
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1
GRID_TYPE = "single"
FFT_PADDING = "circular"

# Optimisation parameters
TRAINING_ITERATIONS = 800_000
WARMUP_ITERATIONS_PERCENTAGE = 0.02
NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 2e-4
GRAD_CLIP = 1.0
ACCUMULATE_GRAD_STEPS = 1

# Diffusion parameters
NUM_TRAIN_TIMESTEPS = 1_000
CLASSIFIER_FREE_GUIDANCE = True
BETA_START = 1e-4
BETA_END = 2e-2
BETA_SCHEDULE = "linear"
TIME_EMBED_DIM = NUM_HIDDEN_CHANNELS
MAX_PERIOD = 10_000.0
NUM_INFERENCE_STEPS = 50
NUM_SAMPLES = 8
LOG_SAMPLES = True
COSINE_SCHEDULE_LOGSNR_MIN = -10.0
COSINE_SCHEDULE_LOGSNR_MAX = 10.0
PREDICTION_TYPE = "v_prediction"
DDIM_ETA = 0.0
EMA_ENABLED = True
EMA_DECAY = 0.999
EMA_WARMUP_STEPS = 1_000
GUIDANCE_SCALE = 3.5
CONDITION_DROPOUT_PROB = 0.1
USE_SIGMOID_LOSS_WEIGHTING = True
SIGMOID_LOSS_BIAS = -3.0
FID_ENABLED = True
FID_NUM_BATCHES = 8
FID_NUM_INFERENCE_STEPS = NUM_INFERENCE_STEPS


def get_config() -> DiffusionExperimentConfig:
    """Return the ImageNet diffusion configuration."""
    config = DiffusionExperimentConfig()
    config.debug = False
    config.seed = 42
    hf_token = os.environ.get("HF_TOKEN")
    config.classifier_free_guidance = CLASSIFIER_FREE_GUIDANCE

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
        hf_dataset_name="imagenet-1k",
        hf_dataset_config=None,
        hf_auth_token=hf_token,
        task="generation",
    )

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.in_channels}", out_features="${net.hidden_dim}"),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.hidden_dim}", out_features="${net.out_channels}"),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(AdaLNZeroResidualBlock)(
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
                            min_std=0.02,
                            max_std=1.5,
                            init_std_low=0.05,
                            init_std_high=1.2,
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
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
            condition_norm_cfg="${net.norm_cfg}",
            hidden_dim="${net.hidden_dim}",
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
        condition_in_proj_cfg=LazyConfig(torch.nn.Linear)(
            in_features="${net.hidden_dim}", out_features="${net.hidden_dim}"
        ),
    )

    config.lightning_wrapper_class = LazyConfig(DiffusionWrapper)()

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    # Compose diffusion config with explicit schedule, sampling, and EMA parameters.
    config.diffusion = DiffusionConfig(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        beta_schedule=BETA_SCHEDULE,
        cosine_schedule_logsnr_min=COSINE_SCHEDULE_LOGSNR_MIN,
        cosine_schedule_logsnr_max=COSINE_SCHEDULE_LOGSNR_MAX,
        cosine_schedule_image_resolution=FINAL_IMAGE_SIZE,
        cosine_schedule_noise_res_high=FINAL_IMAGE_SIZE,
        cosine_schedule_noise_res_low=max(32, FINAL_IMAGE_SIZE // 2),
        prediction_type=PREDICTION_TYPE,
        time_embed_dim=TIME_EMBED_DIM,
        max_period=MAX_PERIOD,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_samples=NUM_SAMPLES,
        log_samples=LOG_SAMPLES,
        ddim_eta=DDIM_ETA,
        use_sigmoid_loss_weighting=USE_SIGMOID_LOSS_WEIGHTING,
        sigmoid_loss_bias=SIGMOID_LOSS_BIAS,
        use_classifier_free_guidance=CLASSIFIER_FREE_GUIDANCE,
        guidance_scale=GUIDANCE_SCALE,
        condition_dropout_prob=CONDITION_DROPOUT_PROB,
        num_classes=NUM_CLASSES,
        fid_enabled=FID_ENABLED,
        fid_num_batches=FID_NUM_BATCHES,
        fid_num_inference_steps=FID_NUM_INFERENCE_STEPS,
    )

    config.ema = EMAConfig(
        enabled=EMA_ENABLED,
        decay=EMA_DECAY,
        warmup_steps=EMA_WARMUP_STEPS,
    )

    config.wandb = WandbConfig(
        job_group="imagenet-diffusion",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    return config
