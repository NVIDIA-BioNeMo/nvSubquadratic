"""4 regs, compress-concat (cr=4), film3_after, register_concat readout (cr=4), RA x3.

First attempt — uses small_init/wang_init, no second gate, rand-m9 augment.
Superseded by r4_cc4_f3after_v2.py which matches the v3 baseline recipe.
"""

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
        # Pin to original (non-v3-matching) settings
        init_style="v2_small_wang",
        use_gated_hyena=False,
        rand_augment="rand-m9-mstd0.5-inc1",
    )
