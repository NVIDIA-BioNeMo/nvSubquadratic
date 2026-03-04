"""WSD finetuning ablation — no Mixup, LR=3e-4, WSD 10/0/90 (heavy decay).

nomix_lr3e4 peaked at 82.01% (ep4) then degraded to 81.52% (ep8).
Aggressive decay should capture the fast learning while preventing degradation.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, LR=3e-4, 90% decay."""
    return _base(
        lr=3e-4,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        warmup_pct=0.10,
        stable_pct=0.0,
    )
