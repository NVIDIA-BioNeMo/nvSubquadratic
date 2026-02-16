"""MQAR experiment: Hyena with MLP expansion_factor=32.0 (Ablation 4b)."""

from examples.language_modeling.mixer_defaults import get_causal_hyena_mixer_cfg
from examples.mqar.base_config import mqar_experiment_config


def get_config():
    config = mqar_experiment_config()
    config.net.block_cfg.sequence_mixer_cfg = get_causal_hyena_mixer_cfg(
        L_cache=256,
    )
    config.net.block_cfg.mlp_cfg.expansion_factor = 32.0
    return config
