"""FiLM finetuning — lr=1.5e-4, dp=0.2, wd=0.3, free FiLM.

Intermediate LR bracket between lr=1e-4 (winning, slow convergence) and
lr=2e-4 (also launched). With wd=0.3 + free FiLM providing regularization,
tests if 50% more LR yields faster convergence without diverging.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1.5e-4, wd=0.3, drop_path_rate=0.2, film_wd=True)
