"""Debug: verify zero-init registers + identity FiLM = exact pretrained accuracy.

With L_cache=14, zero registers, zero padding, and identity FiLM, the patch
token outputs should be identical to the pretrained model. Expected: 81.50%.
"""

from examples.vit5_imagenet.v3.gap_film_regs._base import get_config as _base


def get_config():  # noqa: D103
    config = _base(
        num_registers=14,
        num_film_layers=3,
        film_after_pos_embed=True,
        lr=3e-5,
        wd=0.05,
        drop_path_rate=0.15,
        reg_init="zeros",
        train_do=False,
    )
    config.debug = True
    return config
