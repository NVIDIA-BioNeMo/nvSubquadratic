"""WSD finetuning ablation — cosine, no Mixup, no augmentation policy.

Minimal finetuning with cosine schedule: just random-resized-crop + flip.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with cosine schedule and minimal augmentation."""
    return _base(
        lr=1e-5,
        wd=0.1,
        mixup=0.0,
        cutmix=0.0,
        rand_augment=None,
        use_three_augment=False,
        scheduler_name="cosine",
        warmup_pct=0.25,
        stable_pct=0.0,
    )
