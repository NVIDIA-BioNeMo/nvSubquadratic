"""Tier 1 — Debug: Hyena causal LM on WikiText-103.

~2M params, 1 GPU, ~5 minutes. For quick sanity checks.
"""

from examples.language_modeling.base_config import lm_experiment_config
from examples.language_modeling.mixer_defaults import get_causal_hyena_mixer_cfg


def get_config():
    config = lm_experiment_config(
        # Tiny model
        num_blocks=4,
        hidden_dim=128,
        # Short sequences
        seq_len=256,
        batch_size=64,
        # Quick run
        training_iterations=5_000,
        learning_rate=1e-3,
        weight_decay=0.0,
        val_check_interval=500,
    )
    config.net.block_cfg.sequence_mixer_cfg = get_causal_hyena_mixer_cfg(
        L_cache=256,
    )
    return config
