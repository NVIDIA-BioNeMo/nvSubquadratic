# TODO: Add license header here


"""ImageNet diffusion experiment using nvSubquadratic CKConv backbones."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from examples.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from nvsubquadratic.lazy_config import LazyConfig


@dataclass
class DiffusionHyperParams:
    """Container gathering the essential hyper-parameters for the experiment."""

    image_size: int = 256
    batch_size: int = 64
    hidden_dim: int = 256
    num_blocks: int = 12
    dropout_in: float = 0.0
    dropout_block: float = 0.1
    mlp_ratio: float = 2.0
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    training_iterations: int = 800_000
    warmup_fraction: float = 0.02
    diffusion_steps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: str = "linear"
    inference_steps: int = 50
    sample_grid: int = 4


PLACEHOLDER = None


def get_config() -> ExperimentConfig:
    """Return the LazyConfig-backed experiment specification."""

    hyper = DiffusionHyperParams()
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.comment = "ImageNet diffusion w/ CKConv residual network"

    hf_token = os.environ.get("HF_TOKEN")
    num_workers = max(4, (os.cpu_count() or 8) // 2)

    config.dataset = LazyConfig(
        "examples.imagenet_diffusion.imagenet_datamodule.ImageNetDataModule"
    )(
        data_dir="/media/davidknigge/hard-disk2/huggingface/imagenet",
        batch_size=hyper.batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        seed=config.seed,
        image_size=hyper.image_size,
        drop_labels=True,
        use_deterministic_worker_init=True,
        hf_dataset_name="imagenet-1k",
        hf_dataset_config=None,
        hf_auth_token=hf_token,
    )

    config.lightning_wrapper_class = LazyConfig("examples.lightning_wrappers.DiffusionWrapper")(
        diffusion_cfg={
            "num_train_timesteps": hyper.diffusion_steps,
            "beta_start": hyper.beta_start,
            "beta_end": hyper.beta_end,
            "beta_schedule": hyper.beta_schedule,
            "time_embed_dim": hyper.hidden_dim,
        },
        sample_cfg={
            "num_inference_steps": hyper.inference_steps,
            "num_samples": hyper.sample_grid,
            "log_samples": True,
        },
        ema_cfg={
            "enabled": True,
            "decay": 0.999,
            "warmup_steps": 1000,
            "update_every": 1,
        },
    )

    config.net = LazyConfig("nvsubquadratic.networks.diffusion_resnet.DiffusionResNet")(
        in_channels=PLACEHOLDER,
        out_channels=PLACEHOLDER,
        hidden_dim=hyper.hidden_dim,
        num_blocks=hyper.num_blocks,
        kernel_cfg=LazyConfig("nvsubquadratic.modules.kernels_nd.RandomFourierKernelND")(
            data_dim=2,
            out_dim=hyper.hidden_dim,
            mlp_hidden_dim=64,
            num_layers=3,
            embedding_dim=64,
            omega_0=100.0,
            L_cache=32,
            use_bias=True,
            nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
        ),
        mask_cfg=LazyConfig("nvsubquadratic.modules.masks_nd.GaussianModulationND")(
            data_dim=2,
            num_channels=hyper.hidden_dim,
            min_std=0.02,
            max_std=1.5,
            init_std_low=0.05,
            init_std_high=1.2,
            parametrization="direct",
        ),
        grid_type="double",
        dropout_in=hyper.dropout_in,
        block_dropout=hyper.dropout_block,
        mlp_ratio=hyper.mlp_ratio,
        positional_encoding_cfg=LazyConfig("nvsubquadratic.modules.position_encoding.PositionEmbeddingND")(
            embedding_dim=hyper.hidden_dim,
            data_dim=2,
            max_dim_lengths=(hyper.image_size, hyper.image_size),
        ),
        condition_proj_cfg=LazyConfig(torch.nn.Linear)(
            in_features=hyper.hidden_dim,
            out_features=hyper.hidden_dim,
        ),
    )

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        lr=hyper.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=hyper.weight_decay,
    )

    total_iterations = hyper.training_iterations
    warmup_iterations = int(total_iterations * hyper.warmup_fraction)

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations=warmup_iterations,
        total_iterations="${train.iterations}",
    )

    config.train = TrainConfig(
        batch_size=hyper.batch_size,
        iterations=total_iterations,
        grad_clip=hyper.grad_clip,
    )

    config.wandb = WandbConfig(job_group="imagenet-diffusion")

    config.hooks_enabled = False

    return config
