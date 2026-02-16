"""Base configuration for causal language modeling on WikiText-103.

Parameterized by model size (number of blocks, hidden dim, etc.) and
training hyperparameters. Individual tier configs just call this with
appropriate arguments.

Usage:
    from examples.language_modeling.base_config import lm_experiment_config
    from examples.language_modeling.mixer_defaults import get_causal_hyena_mixer_cfg

    def get_config():
        config = lm_experiment_config(num_blocks=8, hidden_dim=384, ...)
        config.net.block_cfg.sequence_mixer_cfg = get_causal_hyena_mixer_cfg(L_cache=512)
        return config
"""

import os

import torch

from experiments.datamodules.wikitext103 import WikiText103DataModule
from experiments.default_cfg import (
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.autoregressive_wrapper import AutoregressiveWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork

VOCAB_SIZE = 50257  # GPT-2


def lm_experiment_config(
    # Model architecture
    num_blocks: int = 4,
    hidden_dim: int = 128,
    dropout_rate: float = 0.1,
    # Data
    seq_len: int = 256,
    batch_size: int = 64,
    data_dir: str = "/ivi/zfs/s0/original_homes/dwessel/data",
    # Training
    training_iterations: int = 5_000,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_pct: float = 0.05,
    grad_clip: float = 1.0,
    precision: str = "bf16-mixed",
    accumulate_grad_steps: int = 1,
    # Trainer
    val_check_interval: float | int = 1000,
    limit_val_batches: int = 50,
    checkpoint_every_n_steps: int | None = None,
    # Wandb
    wandb_entity: str = "dafidofff",
    wandb_project: str = "nvsubq-lm",
    wandb_job_group: str = "lm_wikitext103",
) -> ExperimentConfig:
    """Get WikiText-103 LM experiment configuration.

    After calling this, you must set:
    - config.net.block_cfg.sequence_mixer_cfg via mixer_defaults

    Returns:
        ExperimentConfig with everything except the mixer.
    """
    config = ExperimentConfig()
    config.debug = False

    # =========================================================================
    # Dataset
    # =========================================================================
    config.dataset = LazyConfig(WikiText103DataModule)(
        seq_len=seq_len,
        batch_size=batch_size,
        data_dir=data_dir,
        num_workers=min(4, os.cpu_count() or 4),
    )

    # =========================================================================
    # Network — Embedding → ResidualBlocks → LM head
    # =========================================================================
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=VOCAB_SIZE,
        out_channels=VOCAB_SIZE,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        data_dim=1,
        in_proj_cfg=LazyConfig(torch.nn.Embedding)(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=hidden_dim,
        ),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(
            in_features=hidden_dim,
            out_features=VOCAB_SIZE,
            bias=False,  # No bias when tying with Embedding weights
        ),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=PLACEHOLDER,
            sequence_mixer_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=4.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=dropout_rate),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            mlp_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=dropout_rate),
        ),
        # NOTE: dropout_in is applied BEFORE in_proj in ResidualNetwork.
        # With nn.Embedding, the input is Long (token IDs), so dropout_in must be 0.0.
        # Dropout is still applied inside each ResidualBlock.
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        # Weight tying: share Embedding weights with LM head (Press & Wolf, 2017)
        tie_weights=True,
    )

    # =========================================================================
    # Training
    # =========================================================================
    config.lightning_wrapper_class = LazyConfig(AutoregressiveWrapper)(
        mode="discrete",
        vocab_size=VOCAB_SIZE,
    )

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=warmup_pct,
        total_iterations="${train.iterations}",
        mode="min",
    )

    config.train = TrainConfig(
        batch_size=batch_size,
        iterations=training_iterations,
        grad_clip=grad_clip,
        precision=precision,
        accumulate_grad_steps=accumulate_grad_steps,
    )

    config.trainer = TrainerConfig(
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
    )

    config.wandb = WandbConfig(
        job_group=wandb_job_group,
        entity=wandb_entity,
        project=wandb_project,
    )

    return config
