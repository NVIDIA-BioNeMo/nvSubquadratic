"""MQAR experiment: Hyena with slow exponential filter decay (Ablation 4c)."""

from examples.language_modeling.mixer_defaults import get_causal_hyena_mixer_cfg
from examples.mqar.base_config import mqar_experiment_config
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import ExponentialModulationND


def get_config():
    config = mqar_experiment_config()
    config.net.block_cfg.sequence_mixer_cfg = get_causal_hyena_mixer_cfg(
        L_cache=256,
        mask_cfg=LazyConfig(ExponentialModulationND)(
            data_dim="${net.data_dim}",
            num_channels="${net.hidden_dim}",
            slow_decay_pct=0.5,
            fast_decay_pct=2.0,
        ),
    )
    return config
