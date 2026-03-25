"""FiLM finetuning — LLRD 0.75 + stronger RandAugment (m14), 10 epochs.

Standard was m9; bumping to m14 increases augmentation intensity. Combined
with LLRD, this aggressively regularizes while protecting features.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.3,
        drop_path_rate=0.2,
        film_wd=True,
        epochs=10,
        layer_decay=0.75,
        rand_augment="rand-m14-mstd0.5-inc1",
    )
