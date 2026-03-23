"""FiLM finetuning — lr=1e-4, dp=0.2, wd=0.1, mixup=0.8 + cutmix=1.0.

Re-enables pretrain-level mixup/cutmix augmentation alongside higher LR.
The model was trained with these augmentations for 800 epochs; removing
them during finetuning may have contributed to overfitting in wave 1.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.1, drop_path_rate=0.2, mixup=0.8, cutmix=1.0)
