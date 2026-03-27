"""FiLM finetuning — higher learning rate (5e-5).

Was #2 in the v2 attention sweep (82.06% test with dp=0.15).
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=5e-5)
