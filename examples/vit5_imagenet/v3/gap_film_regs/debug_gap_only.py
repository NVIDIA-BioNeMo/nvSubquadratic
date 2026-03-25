"""Debug validation — load tcji9tfx checkpoint and verify ~81.5% accuracy.

No training, no WandB logging. Just downloads the checkpoint, remaps old
SIREN keys, loads weights, and runs a single validation pass.
"""

from examples.vit5_imagenet.v3.gap_film_regs._base import get_config as _base
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Return a validation-only config for the GAP Hyena model."""
    config = _base(train_do=False)
    config.debug = True
    return config
