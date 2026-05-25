"""v6_hierarchical — pure Hyena 4-stage (Swin-T layout) with BD + learnable ω₀.

No registers, no FiLM.  Stages [96, 192, 384, 768] x depths [2, 2, 6, 2] on
initial patch size 4 (grids 56 -> 28 -> 14 -> 7).  PatchMerging between stages.
SIREN kernel: ``BlockDiagonalLearnableOmegaSIRENKernelND``.

Effective batch = 2048 (8 GPUs x 16 per-GPU x 16 accum).
"""

from examples.vit5_imagenet.v6_hierarchical._base_config import (
    build_hyena_hier_net,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Return the pure (no-FiLM) hierarchical Hyena config."""
    config = get_base_config()
    config.net = build_hyena_hier_net(layout="pure")
    return config
