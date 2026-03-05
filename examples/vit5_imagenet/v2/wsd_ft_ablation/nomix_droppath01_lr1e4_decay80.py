"""WSD finetuning ablation — no Mixup, drop path 0.1, LR=1e-4, WSD 10/10/80.

The 80% decay schedule was the most stable (81.99% with minimal degradation).
Combines with higher drop path and higher LR.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.1, LR=1e-4, 80% decay."""
    return _base(
        lr=1e-4,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        drop_path_rate=0.1,
        warmup_pct=0.10,
        stable_pct=0.10,
    )
