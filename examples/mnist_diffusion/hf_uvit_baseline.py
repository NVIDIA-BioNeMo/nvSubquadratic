# TODO: Add license header here

"""Baseline diffusion configuration leveraging Hugging Face UVit transformer."""

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
from nvsubquadratic.networks.huggingface_diffusers import DiffusersUVitWrapper, HuggingFaceUVitConfig


PLACEHOLDER = None
WANDB_ENTITY = "dafidofff"

# Dataset 
BATCH_SIZE = 16
NUM_WORKERS = 16

# UVit architecture ------------------------------------------------------------
UVIT_SAMPLE_SIZE = 28
UVIT_IN_CHANNELS = 1
UVIT_OUT_CHANNELS = 1
UVIT_HIDDEN_SIZE = 256
UVIT_COND_EMBED_DIM = 128
UVIT_ENCODER_HIDDEN_SIZE = 128
UVIT_BLOCK_OUT_CHANNELS = 256
UVIT_NUM_HIDDEN_LAYERS = 8
UVIT_NUM_ATTENTION_HEADS = 8
UVIT_INTERMEDIATE_SIZE = 512
UVIT_LAYER_NORM_EPS = 1e-5
UVIT_MICRO_COND_ENCODE_DIM = None
UVIT_MICRO_COND_EMBED_DIM = None
UVIT_CODEBOOK_SIZE = None
UVIT_VOCAB_SIZE = None

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

    hf_cfg = HuggingFaceUVitConfig(
        sample_size=UVIT_SAMPLE_SIZE,
        in_channels=UVIT_IN_CHANNELS,
        out_channels=UVIT_OUT_CHANNELS,
        hidden_size=UVIT_HIDDEN_SIZE,
        cond_embed_dim=UVIT_COND_EMBED_DIM,
        encoder_hidden_size=UVIT_ENCODER_HIDDEN_SIZE,
        block_out_channels=UVIT_BLOCK_OUT_CHANNELS,
        num_hidden_layers=UVIT_NUM_HIDDEN_LAYERS,
        num_attention_heads=UVIT_NUM_ATTENTION_HEADS,
        intermediate_size=UVIT_INTERMEDIATE_SIZE,
        layer_norm_eps=UVIT_LAYER_NORM_EPS,
        micro_cond_encode_dim=UVIT_MICRO_COND_ENCODE_DIM,
        micro_cond_embed_dim=UVIT_MICRO_COND_EMBED_DIM,
        codebook_size=UVIT_CODEBOOK_SIZE,
        vocab_size=UVIT_VOCAB_SIZE,
    )

    config.net = LazyConfig(DiffusersUVitWrapper)(hf_config=hf_cfg)
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
        job_group="mnist_diffusion_hf_uvit_baseline",
        entity=WANDB_ENTITY,
    )

    return config
