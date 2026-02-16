"""Tier 4 — Scale: Hyena causal LM on WikiText-103.

~350M params, 32x A100, ~1-2 days. Placeholder for future scaling.
"""

from examples.language_modeling.base_config import lm_experiment_config
from examples.language_modeling.mixer_defaults import get_causal_hyena_mixer_cfg


def get_config():
    config = lm_experiment_config(
        num_blocks=24,
        hidden_dim=1024,
        dropout_rate=0.1,
        seq_len=2048,
        batch_size=8,
        training_iterations=200_000,
        learning_rate=3e-4,
        weight_decay=0.1,
        accumulate_grad_steps=4,
        val_check_interval=5000,
        checkpoint_every_n_steps=5000,
    )
    config.net.block_cfg.sequence_mixer_cfg = get_causal_hyena_mixer_cfg(
        L_cache=2048,
        kernel_mlp_hidden_dim=64,
    )
    config.num_nodes = 4  # 32 GPUs = 4 nodes x 8 GPUs
    return config
