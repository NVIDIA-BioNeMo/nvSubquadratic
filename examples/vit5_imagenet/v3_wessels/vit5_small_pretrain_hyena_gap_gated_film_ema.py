"""ViT-5-Small + Hyena ImageNet-1k — GAP (global average pooling), FiLM-conditioned SIREN.

Like the CLS-row variant but without a CLS token. Classification reads out via
global average pooling over patch tokens. 14 register tokens fill the first row
of the 15×14 spatial grid (same shape as CLS-row), so the Hyena mixer is
unaffected. Registers are still used for FiLM kernel conditioning via RegisterPooling.
"""

from examples.vit5_imagenet.v3_wessels._base_config import (
    NUM_REGISTERS_NO_CLS,
    build_film_cfg,
    build_hyena_mixer,
    build_cls_row_network,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Hyena GAP + FiLM config."""
    config = get_base_config()

    film_cfg = build_film_cfg()
    mixer_cfg = build_hyena_mixer(film_cfg=film_cfg, use_rope=False)
    config.net, trainer_overrides = build_cls_row_network(
        mixer_cfg,
        use_cls_token=False,
        num_registers=NUM_REGISTERS_NO_CLS,
        register_start_idx=0,
    )
    for k, v in trainer_overrides.items():
        setattr(config.trainer, k, v)

    return config
