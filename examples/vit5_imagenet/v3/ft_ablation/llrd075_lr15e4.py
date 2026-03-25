"""FiLM finetuning — LLRD 0.75 + LR 1.5e-4, 10 epochs.

Higher LR than 1e-4 but lower than 3e-4 (which diverged). LLRD protects
lower layers from the stronger update signal.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1.5e-4, wd=0.3, drop_path_rate=0.2, film_wd=True, epochs=10, layer_decay=0.75)
