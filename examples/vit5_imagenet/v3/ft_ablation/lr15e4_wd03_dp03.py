"""FiLM finetuning — lr=1.5e-4, dp=0.3, wd=0.3, free FiLM.

Combines the two best regularization strategies found so far:
lr=1.5e-4 (faster convergence than 1e-4) + dp=0.3 (stronger structural
regularization than 0.2) + wd=0.3 + free FiLM. Both dp=0.3 and lr=1.5e-4
independently gave val_loss=0.713 at epoch 7; testing if combining them
extends the optimal window further.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1.5e-4, wd=0.3, drop_path_rate=0.3, film_wd=True)
