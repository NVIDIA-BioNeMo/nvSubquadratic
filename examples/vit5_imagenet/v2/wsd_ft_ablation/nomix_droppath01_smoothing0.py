"""WSD finetuning ablation — no Mixup, drop path 0.1, no label smoothing.

Combines the two best single modifications: drop path 0.1 (82.00%)
and no label smoothing (81.98%).
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.1, no label smoothing."""
    return _base(lr=3e-5, wd=0.05, mixup=0.0, cutmix=0.0, drop_path_rate=0.1, smoothing=0.0)
