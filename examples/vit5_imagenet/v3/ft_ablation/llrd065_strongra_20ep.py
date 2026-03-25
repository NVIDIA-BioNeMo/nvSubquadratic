"""FiLM finetuning — aggressive LLRD 0.65 + strong RA + 20 epochs.

The llrd065_20ep run sustains 0.814 at epoch 14 (best long-run result).
Adding strong RA (m14) may further slow overfitting in the extended
training window.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.3,
        drop_path_rate=0.2,
        film_wd=True,
        epochs=20,
        layer_decay=0.65,
        rand_augment="rand-m14-mstd0.5-inc1",
    )
