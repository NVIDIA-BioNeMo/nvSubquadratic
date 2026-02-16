"""Tier 3 — Medium: Hyena causal LM on WikiText-103.

~125M params, 4x A6000, ~8-12 hours.
"""

from examples.language_modeling.base_config import lm_experiment_config
from examples.language_modeling.mixer_defaults import get_causal_hyena_mixer_cfg


def get_config():
    config = lm_experiment_config(
        num_blocks=12,
        hidden_dim=768,
        dropout_rate=0.1,
        seq_len=1024,
        batch_size=16,
        training_iterations=100_000,
        learning_rate=3e-4,
        weight_decay=0.1,
        val_check_interval=5000,
        checkpoint_every_n_steps=5000,
    )
    config.net.block_cfg.sequence_mixer_cfg = get_causal_hyena_mixer_cfg(
        L_cache=1024,
        kernel_mlp_hidden_dim=64,
    )
    return config
