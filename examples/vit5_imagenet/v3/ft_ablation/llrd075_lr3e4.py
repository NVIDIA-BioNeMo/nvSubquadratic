"""FiLM finetuning — LLRD 0.75 + higher LR 3e-4, 10 epochs.

LR=3e-4 diverged in Wave 2 without LLRD. With LLRD, lower layers are
protected so the higher LR only fully applies to the head/upper blocks.
Tests whether LLRD can rescue an otherwise-too-aggressive learning rate.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=3e-4, wd=0.3, drop_path_rate=0.2, film_wd=True, epochs=10, layer_decay=0.75)
