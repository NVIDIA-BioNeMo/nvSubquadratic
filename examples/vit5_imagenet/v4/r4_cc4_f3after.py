"""4 regs, compress-concat (cr=4), film3_after, register_concat readout (cr=4), RA x3."""

from examples.vit5_imagenet.v4._base import get_config as _base


def get_config():  # noqa: D103
    return _base(
        num_registers=4,
        num_film_layers=3,
        film_after_pos_embed=True,
        register_pooling_mode="compress_concat",
        film_compression_ratio=4,
        num_repeats=3,
        readout="register_concat",
        neck_compression_ratio=4,
    )
