"""WSD finetuning ablation — no Mixup, LR=1e-4, WSD 10/10/80.

The 10/10/80 schedule is the most stable so far (81.99% at ep13).
Testing with higher LR to see if we can push the peak higher.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, LR=1e-4, 80% decay."""
    return _base(
        lr=1e-4,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        warmup_pct=0.10,
        stable_pct=0.10,
    )
