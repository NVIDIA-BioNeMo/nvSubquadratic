"""WSD finetuning ablation — no Mixup, no drop path (0.0).

Default is 0.05. Removing drop path may help in a regime where Mixup/CutMix
are already disabled, since the model is less regularized overall.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path = 0.0."""
    return _base(lr=3e-5, wd=0.05, mixup=0.0, cutmix=0.0, drop_path_rate=0.0)
