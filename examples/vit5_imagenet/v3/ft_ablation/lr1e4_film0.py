"""FiLM finetuning — lr=1e-4, dp=0.2, wd=0.1, film_wd=0 (no WD on FiLM).

Removes weight decay on the FiLM generator networks to allow maximum
input-conditioning freedom during finetuning at higher learning rate.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.1, drop_path_rate=0.2, film_wd=True)
