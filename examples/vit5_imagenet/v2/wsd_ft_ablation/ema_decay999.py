"""WSD finetuning ablation — EMA decay=0.999 (half-life ~1K steps)."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base
from experiments.callbacks.model_ema import LabeledEMAWeightAveraging
from nvsubquadratic.lazy_config import LazyConfig


def get_config():
    """Return config with EMA decay=0.999."""
    config = _base(lr=3e-5, wd=0.05)
    config.callbacks = [LazyConfig(LabeledEMAWeightAveraging)(decay=0.999)]
    return config
