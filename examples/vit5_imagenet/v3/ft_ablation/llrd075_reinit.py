"""FiLM finetuning — LLRD 0.75 + re-initialized classification head, 10 epochs.

The pretrained head has already converged to a specific feature mapping.
Re-initializing it forces the model to re-learn the final mapping from
scratch, which combined with LLRD gives the head maximum freedom while
protecting the backbone.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.3,
        drop_path_rate=0.2,
        film_wd=True,
        epochs=10,
        layer_decay=0.75,
        reinit_head=True,
    )
