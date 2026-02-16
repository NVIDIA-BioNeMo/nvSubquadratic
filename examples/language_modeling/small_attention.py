"""Tier 2 — Small: Attention causal LM on WikiText-103.

~25M params, 4x RTX 3090, ~2-4 hours.
"""

from examples.language_modeling.base_config import lm_experiment_config
from examples.language_modeling.mixer_defaults import get_causal_attention_mixer_cfg


def get_config():
    config = lm_experiment_config(
        num_blocks=8,
        hidden_dim=384,
        dropout_rate=0.1,
        seq_len=512,
        batch_size=32,
        training_iterations=50_000,
        learning_rate=3e-4,
        weight_decay=0.1,
        val_check_interval=2000,
        checkpoint_every_n_steps=5000,
    )
    config.net.block_cfg.sequence_mixer_cfg = get_causal_attention_mixer_cfg(
        num_heads=6,  # 384 / 6 = 64 per head
    )
    return config
