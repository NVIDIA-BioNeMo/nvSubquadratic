"""FiLM finetuning — LLRD 0.75 + head re-init + three-augment, 10 epochs.

Maximum structural + data regularization combo: protect backbone via LLRD,
re-learn the head from scratch, and use three-augment for data diversity.
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
        reinit_head=True,
        use_three_augment=True,
    )
