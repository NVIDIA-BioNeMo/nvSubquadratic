# TODO: Add license header here

"""3D Color Conditioning -- Attention XS (v2).

Hidden dim: 240, 4 blocks, num_heads=8, head_dim=30.
3D volume [D=8, H=64, W=64].
3D RoPE enabled (head_dim=30, 30 % 6 == 0).
Compile: max-autotune.
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.color_conditioning_3d._base import base_experiment_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER


HIDDEN_DIM = 240
NUM_HEADS = 8  # head_dim = 30 (divisible by 6 for 3D RoPE)


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.compile_mode = "max-autotune"

    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_attention_mixer_cfg(
        num_heads=NUM_HEADS,
        apply_qk_norm=True,
        use_rope=True,
        rope_spatial_dims=("${dataset.canvas_depth}", "${dataset.canvas_size}", "${dataset.canvas_size}"),
    )

    return config
