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
from experiments.lightning_wrappers import DiffusionWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.huggingface_diffusers import DiffusersDiTWrapper, HuggingFaceDiTConfig


WANDB_ENTITY = "dafidofff"
PLACEHOLDER = None

# Dataset 
BATCH_SIZE = 32
NUM_WORKERS = 16

# Optimisation 
TRAINING_ITERATIONS = 100_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Diffusion 
NUM_TRAIN_TIMESTEPS = 1_000
NUM_INFERENCE_STEPS = 50
NUM_SAMPLES = 16
LOG_SAMPLES = True


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
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_samples=NUM_SAMPLES,
        log_samples=LOG_SAMPLES,
    )

    config.wandb = WandbConfig(
        job_group="mnist_diffusion_hf_baseline",
        entity=WANDB_ENTITY,
    )

    return config
