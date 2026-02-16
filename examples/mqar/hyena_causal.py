"""MQAR experiment with causal Hyena mixer."""

from examples.language_modeling.mixer_defaults import get_causal_hyena_mixer_cfg
from examples.mqar.base_config import mqar_experiment_config


def get_config():
    config = mqar_experiment_config()
    config.net.block_cfg.sequence_mixer_cfg = get_causal_hyena_mixer_cfg(
        L_cache=256,  # Match seq_len
    )
    return config
