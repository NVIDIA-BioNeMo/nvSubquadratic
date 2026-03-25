"""FiLM finetuning — lr=3e-4, dp=0.3, wd=0.1.

Aggressive 10x learning rate with heavy regularization. Tests whether
the model can reach a better basin far from the pretrained weights.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=3e-4, wd=0.1, drop_path_rate=0.3)
