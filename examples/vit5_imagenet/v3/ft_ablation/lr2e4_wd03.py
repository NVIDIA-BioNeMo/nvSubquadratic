"""FiLM finetuning — lr=2e-4, dp=0.2, wd=0.3, free FiLM.

Intermediate LR between the winning lr=1e-4 (wd=0.3) and the diverged
lr=3e-4. Tests whether a slightly higher LR can push past the 0.817
plateau while wd=0.3 prevents overfitting.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=2e-4, wd=0.3, drop_path_rate=0.2, film_wd=True)
