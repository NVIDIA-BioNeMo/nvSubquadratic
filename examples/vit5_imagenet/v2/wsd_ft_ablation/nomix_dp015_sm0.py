"""WSD finetuning ablation — no Mixup, drop path 0.15, no smoothing.

Drop path 0.15 at 82.03%. No smoothing at 81.98%. Testing combination.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.15, no smoothing."""
    return _base(lr=3e-5, wd=0.05, mixup=0.0, cutmix=0.0, drop_path_rate=0.15, smoothing=0.0)
