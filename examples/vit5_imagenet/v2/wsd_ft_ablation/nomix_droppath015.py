"""WSD finetuning ablation — no Mixup, drop path 0.15 (3x default).

nomix_droppath01 (0.1) holding at 81.95% while nomix_droppath0 (0.0)
is declining. Testing if even more structural regularization helps
in the no-mixup regime.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path = 0.15."""
    return _base(lr=3e-5, wd=0.05, mixup=0.0, cutmix=0.0, drop_path_rate=0.15)
