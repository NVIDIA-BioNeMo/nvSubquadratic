"""WSD finetuning ablation — no Mixup, drop path 0.1, LR=1e-4, WSD 10/0/90.

Combines the three best findings: no mixup, higher drop path (0.1),
higher LR (1e-4) with aggressive decay. Aiming to beat 81.99%.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.1, LR=1e-4, 90% decay."""
    return _base(
        lr=1e-4,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        drop_path_rate=0.1,
        warmup_pct=0.10,
        stable_pct=0.0,
    )
