"""FiLM finetuning — aggressive LLRD 0.65 + 20 epochs.

The most protective LLRD with extended training. If the ceiling is in the
upper layers, this maximally protects the backbone while giving extra
time for the head and top blocks to converge.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.3, drop_path_rate=0.2, film_wd=True, epochs=20, layer_decay=0.65)
