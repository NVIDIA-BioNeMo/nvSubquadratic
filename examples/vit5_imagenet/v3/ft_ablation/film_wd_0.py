"""FiLM finetuning — no weight decay on FiLM generator.

Fully removes WD on FiLM params (no_weight_decay=True), giving maximum
freedom for input-dependent conditioning.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(film_wd=True)
