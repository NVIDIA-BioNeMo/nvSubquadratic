"""FiLM finetuning — lr=1e-4, dp=0.2, wd=0.3, free FiLM + three-augment.

Direct variant of the winning lr1e4_wd03 recipe with three-augment added.
Tests whether the pretrain-style augmentation can further delay overfitting
on top of the already-effective wd=0.3 + free FiLM foundation.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.3,
        drop_path_rate=0.2,
        film_wd=True,
        use_three_augment=True,
    )
