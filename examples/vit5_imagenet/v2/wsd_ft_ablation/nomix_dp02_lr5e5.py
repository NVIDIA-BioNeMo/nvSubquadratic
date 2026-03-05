"""WSD finetuning ablation — no Mixup, drop path 0.2, LR=5e-5.

dp02 peaked at 82.02% with LR=3e-5. Testing with LR=5e-5 since
dp01_lr5e5 reached 82.04%.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.2, LR=5e-5."""
    return _base(lr=5e-5, wd=0.05, mixup=0.0, cutmix=0.0, drop_path_rate=0.2)
