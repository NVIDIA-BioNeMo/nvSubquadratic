"""WSD finetuning ablation — no Mixup, drop path 0.2 (4x default).

Testing if even higher drop path further improves on the 0.1 result (82.05%).
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path = 0.2."""
    return _base(lr=3e-5, wd=0.05, mixup=0.0, cutmix=0.0, drop_path_rate=0.2)
