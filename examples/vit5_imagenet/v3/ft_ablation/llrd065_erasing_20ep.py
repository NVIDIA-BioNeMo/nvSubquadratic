"""FiLM finetuning — aggressive LLRD 0.65 + random erasing + 20 epochs.

Combines the most protective LLRD (0.65) with the best-loss augmentation
(RE=0.25) for extended training.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.3,
        drop_path_rate=0.2,
        film_wd=True,
        epochs=20,
        layer_decay=0.65,
        random_erasing_prob=0.25,
    )
