"""WSD finetuning ablation — minimal augmentation, LR=1e-5, WD=0.1.

Closest to pure finetuning: no Mixup, no CutMix, no RandAugment, no
ThreeAugment. Only basic random-resized-crop and horizontal flip.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with minimal augmentation at LR=1e-5."""
    return _base(
        lr=1e-5,
        wd=0.1,
        mixup=0.0,
        cutmix=0.0,
        rand_augment=None,
        use_three_augment=False,
    )
