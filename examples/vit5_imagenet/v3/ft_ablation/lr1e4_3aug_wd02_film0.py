"""FiLM finetuning — lr=1e-4, dp=0.2, wd=0.2, free FiLM + three-augment.

Combines the winning free FiLM recipe with moderate backbone WD (0.2,
between the stable 0.1 and the leader 0.3) and three-augment to delay
overfitting while keeping convergence fast enough.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.2,
        drop_path_rate=0.2,
        film_wd=True,
        use_three_augment=True,
    )
