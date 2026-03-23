"""FiLM finetuning — LLRD 0.75 + random erasing + 20 epochs.

Random erasing had the best val_loss in Wave 4 (0.7135). Extended to
20 epochs to see if the regularization benefit extends with longer training.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.3,
        drop_path_rate=0.2,
        film_wd=True,
        epochs=20,
        layer_decay=0.75,
        random_erasing_prob=0.25,
    )
