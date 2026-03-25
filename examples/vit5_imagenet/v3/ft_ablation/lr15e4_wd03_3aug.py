"""FiLM finetuning — lr=1.5e-4, dp=0.2, wd=0.3, free FiLM + three-augment.

Combines the faster convergence of lr=1.5e-4 with three-augment (the most
effective data regularizer found) and the winning wd=0.3 + free FiLM
foundation. Both lr=1.5e-4 and three-augment individually showed good
results; tests whether they combine well.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1.5e-4,
        wd=0.3,
        drop_path_rate=0.2,
        film_wd=True,
        use_three_augment=True,
    )
