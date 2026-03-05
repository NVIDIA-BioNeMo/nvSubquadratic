"""WSD finetuning ablation — no Mixup, drop path 0.25 (5x default).

Testing if even higher drop path pushes beyond the 0.15 result (82.07%).
The 0.2 run is at 82.02% (ep7). Exploring the upper range.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path = 0.25."""
    return _base(lr=3e-5, wd=0.05, mixup=0.0, cutmix=0.0, drop_path_rate=0.25)
