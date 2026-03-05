"""WSD finetuning ablation — no-mixup minimal augmentation (crop/flip only)."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config for no-mixup with minimal augmentation."""
    return _base(
        lr=3e-5,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        rand_augment=None,
        use_three_augment=False,
    )
