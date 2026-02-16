"""MQAR experiment with causal Attention mixer."""

from examples.language_modeling.mixer_defaults import get_causal_attention_mixer_cfg
from examples.mqar.base_config import mqar_experiment_config


def get_config():
    config = mqar_experiment_config()
    config.net.block_cfg.sequence_mixer_cfg = get_causal_attention_mixer_cfg(
        num_heads=4,  # hidden_dim=128, so 128/4=32 per head
    )
    return config
