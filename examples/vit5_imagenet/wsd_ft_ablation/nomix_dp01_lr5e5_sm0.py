"""WSD finetuning ablation — no Mixup, drop path 0.1, LR=5e-5, no smoothing.

dp01_lr5e5 at 82.04% and dp01_smoothing0 at 82.00%. Testing combination.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, dp 0.1, LR=5e-5, no smoothing."""
    return _base(lr=5e-5, wd=0.05, mixup=0.0, cutmix=0.0, drop_path_rate=0.1, smoothing=0.0)
