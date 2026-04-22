"""Hybrid Hyena/Attention – 3:1 Hyena-to-Attention ratio (12 blocks), block-diagonal kernel.

Layout: H H H A H H H A H H H A
         └─────── 3 groups ─────┘

Same architecture as ``hybrid_hhha`` but every Hyena block uses
``BlockDiagonalMultiOmegaSIRENKernelND`` paired with
``BlockAlignedGaussianModulationND`` instead of the scalar-ω₀ SIREN + standard
Gaussian mask.

Override ``net.patch_size`` to change resolution (default 16).
"""

from examples.vit5_imagenet.v5._base import NUM_BLOCKS, PATCH_SIZE
from examples.vit5_imagenet.vit5_hybrid._base_config import (
    build_hybrid_net,
    get_base_config,
)
from examples.vit5_imagenet.vit5_hybrid._blockdiag import apply_block_diag_overrides
from experiments.default_cfg import ExperimentConfig


LAYER_PATTERN = "HHHA" * (NUM_BLOCKS // 4)


def get_config() -> ExperimentConfig:
    """Build the HHHA hybrid config with block-diagonal Hyena kernels."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.net = build_hybrid_net(layer_pattern=LAYER_PATTERN, patch_size=PATCH_SIZE)
    apply_block_diag_overrides(config)
    config.wandb.job_group = "vit5_hybrid_blockdiag"
    return config
