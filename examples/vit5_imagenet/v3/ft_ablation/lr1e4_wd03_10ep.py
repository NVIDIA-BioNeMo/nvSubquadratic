"""FiLM finetuning — lr=1e-4, dp=0.2, wd=0.3, free FiLM, 10 epochs only.

Directly tests the wave 2 recommendation: the model peaks at epoch 5-7
and overfits beyond that. Running only 10 epochs should capture the peak
while avoiding the overfit tail. Uses the winning wd=0.3 + free FiLM
recipe from the 25-epoch runs.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.3, drop_path_rate=0.2, film_wd=True, epochs=10)
