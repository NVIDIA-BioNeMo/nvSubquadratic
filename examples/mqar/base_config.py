"""Base configuration for MQAR (Multi-Query Associative Recall) experiments.

Usage:
    from examples.mqar.base_config import mqar_experiment_config
    from examples.language_modeling.mixer_defaults import get_causal_hyena_mixer_cfg

    def get_config():
        config = mqar_experiment_config()
        config.net.block_cfg.sequence_mixer_cfg = get_causal_hyena_mixer_cfg(L_cache=256)
        return config
"""

import os

import torch

from experiments.datamodules.mqar import MQARDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, TrainerConfig, WandbConfig
from experiments.lightning_wrappers.autoregressive_wrapper import AutoregressiveWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


def mqar_experiment_config(
    # MQAR task params
    vocab_size: int = 8192,
    seq_len: int = 256,
    num_kv_pairs: int = 8,
    num_train_examples: int = 100_000,
    num_val_examples: int = 5_000,
    batch_size: int = 64,
    # Model architecture
    num_blocks: int = 4,
    hidden_dim: int = 128,
    dropout_rate: float = 0.0,
    # Training
    training_iterations: int = 100_000,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    warmup_pct: float = 0.05,
    grad_clip: float = 1.0,
    # Wandb
    wandb_entity: str = "dafidofff",
    wandb_project: str = "nvsubq-mqar",
) -> ExperimentConfig:
    """Get MQAR experiment configuration.

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
    config.dataset = LazyConfig(MQARDataModule)(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_kv_pairs=num_kv_pairs,
        num_train_examples=num_train_examples,
        num_val_examples=num_val_examples,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 4),
    )

    # =========================================================================
    # Network — Embedding → ResidualBlocks → Linear head
    # =========================================================================
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=vocab_size,   # Used as num_embeddings
        out_channels=vocab_size,  # Predict over vocab
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        data_dim=1,
        # Embedding layer (instead of Linear for discrete tokens)
        in_proj_cfg=LazyConfig(torch.nn.Embedding)(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
        ),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(
            in_features=hidden_dim,
            out_features=vocab_size,
        ),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=PLACEHOLDER,  # Set by experiment config
            sequence_mixer_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=2.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=dropout_rate),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            mlp_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=dropout_rate),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
    )

    # =========================================================================
    # Training
    # =========================================================================
    config.lightning_wrapper_class = LazyConfig(AutoregressiveWrapper)(
        mode="discrete",
        vocab_size=vocab_size,
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
        precision="bf16-mixed",
    )

    config.trainer = TrainerConfig(
        val_check_interval=1000,
        limit_val_batches=50,
    )

    config.wandb = WandbConfig(
        job_group="mqar",
        entity=wandb_entity,
        project=wandb_project,
    )

    return config
