"""FiLM finetuning — lr=1e-4, dp=0.2, wd=0.1, full pretrain augmentation.

Full pretrain augmentation pipeline: mixup=0.8, cutmix=1.0, three-augment.
This matches the data distribution the model saw during 800 epochs of
pretraining, which may be key for continued generalization.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.1,
        drop_path_rate=0.2,
        mixup=0.8,
        cutmix=1.0,
        use_three_augment=True,
    )
