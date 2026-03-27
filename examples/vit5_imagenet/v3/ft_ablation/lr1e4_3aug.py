"""FiLM finetuning — lr=1e-4, dp=0.2, wd=0.1, three-augment.

Switches from RandAugment to three-augment (grayscale, solarize, gaussian
blur) which was used during pretraining. Tests whether matching the pretrain
augmentation strategy helps generalization.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.1, drop_path_rate=0.2, use_three_augment=True)
