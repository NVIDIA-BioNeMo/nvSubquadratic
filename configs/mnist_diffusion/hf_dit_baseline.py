# TODO: Add license header here

"""Baseline diffusion configuration leveraging a Hugging Face DiT transformer."""

import torch

from experiments.datamodules.mnist import MNISTDataModule
from experiments.default_cfg import (
    DiffusionConfig,
    DiffusionExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.diffusion_wrapper import DiffusionWrapper
from nvsubq_paper.lazy_config import LazyConfig
from nvsubq_paper.networks.huggingface_diffusers import DiffusersDiTWrapper, HuggingFaceDiTConfig


WANDB_ENTITY = "dafidofff"
PLACEHOLDER = None

# Dataset
BATCH_SIZE = 32
NUM_WORKERS = 16
IMAGE_SIZE = 28
NUM_CLASSES = 10

# Optimisation
TRAINING_ITERATIONS = 100_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Diffusion
NUM_TRAIN_TIMESTEPS = 1_000
BETA_START = 1e-4
BETA_END = 0.02
BETA_SCHEDULE = "squaredcos_cap_v2"
COSINE_SCHEDULE_LOGSNR_MIN = -10.0
COSINE_SCHEDULE_LOGSNR_MAX = 10.0
COSINE_SCHEDULE_IMAGE_RESOLUTION = IMAGE_SIZE
COSINE_SCHEDULE_NOISE_RES_HIGH = IMAGE_SIZE
COSINE_SCHEDULE_NOISE_RES_LOW = max(16, IMAGE_SIZE // 2)
PREDICTION_TYPE = "v_prediction"
TIME_EMBED_DIM = None
MAX_PERIOD = 10_000.0
NUM_INFERENCE_STEPS = 50
NUM_SAMPLES = 16
LOG_SAMPLES = True
DDIM_ETA = 0.0
EMA_ENABLED = False
EMA_DECAY = 0.999
EMA_WARMUP_STEPS = 0
EMA_UPDATE_EVERY = 1
CFG_ENABLED = True
GUIDANCE_SCALE = 3.0
CONDITION_DROPOUT_PROB = 0.1
USE_SIGMOID_LOSS_WEIGHTING = True
SIGMOID_LOSS_BIAS = 0.0
FID_ENABLED = False
FID_NUM_BATCHES = 0
FID_NUM_INFERENCE_STEPS = None


def get_config() -> DiffusionExperimentConfig:
    """Build the experiment configuration."""
    config = DiffusionExperimentConfig()

    config.dataset = LazyConfig(MNISTDataModule)(
        data_dir=".data/mnist",
        data_type="image",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        use_deterministic_worker_init=config.deterministic,
        seed=config.seed,
        task="generation",
    )

    hf_cfg = HuggingFaceDiTConfig(
        sample_size=28,
        patch_size=2,
        in_channels=1,
        out_channels=1,
        num_layers=6,
        num_attention_heads=4,
        attention_head_dim=64,
        dropout=0.0,
        num_embeds_ada_norm=NUM_TRAIN_TIMESTEPS,
        activation_fn="gelu-approximate",
        norm_type="ada_norm_zero",
        norm_num_groups=32,
    )

    config.net = LazyConfig(DiffusersDiTWrapper)(hf_config=hf_cfg)
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

    config.diffusion = DiffusionConfig(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        beta_schedule=BETA_SCHEDULE,
        cosine_schedule_logsnr_min=COSINE_SCHEDULE_LOGSNR_MIN,
        cosine_schedule_logsnr_max=COSINE_SCHEDULE_LOGSNR_MAX,
        cosine_schedule_image_resolution=COSINE_SCHEDULE_IMAGE_RESOLUTION,
        cosine_schedule_noise_res_high=COSINE_SCHEDULE_NOISE_RES_HIGH,
        cosine_schedule_noise_res_low=COSINE_SCHEDULE_NOISE_RES_LOW,
        prediction_type=PREDICTION_TYPE,
        time_embed_dim=TIME_EMBED_DIM,
        max_period=MAX_PERIOD,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_samples=NUM_SAMPLES,
        log_samples=LOG_SAMPLES,
        ddim_eta=DDIM_ETA,
        ema_enabled=EMA_ENABLED,
        ema_decay=EMA_DECAY,
        ema_update_every=EMA_UPDATE_EVERY,
        ema_warmup_steps=EMA_WARMUP_STEPS,
        use_sigmoid_loss_weighting=USE_SIGMOID_LOSS_WEIGHTING,
        sigmoid_loss_bias=SIGMOID_LOSS_BIAS,
        num_classes=NUM_CLASSES,
        use_classifier_free_guidance=CFG_ENABLED,
        guidance_scale=GUIDANCE_SCALE,
        condition_dropout_prob=CONDITION_DROPOUT_PROB,
        fid_enabled=FID_ENABLED,
        fid_num_batches=FID_NUM_BATCHES,
        fid_num_inference_steps=FID_NUM_INFERENCE_STEPS,
    )

    config.wandb = WandbConfig(
        job_group="mnist_diffusion_hf_baseline",
        entity=WANDB_ENTITY,
    )

    return config
