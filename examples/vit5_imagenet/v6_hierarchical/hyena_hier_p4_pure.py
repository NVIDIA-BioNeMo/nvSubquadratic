"""v6_hierarchical — pure Hyena, 4-stage Swin-T layout, BD + learnable ω₀.

Modelled on ``vit5_hybrid/full_hyena_learnable_omega_blockdiag.py``.

Layout : 4 stages × [2, 2, 6, 2] blocks, dims [96, 192, 384, 768].
Kernel : BlockDiagonalLearnableOmegaSIRENKernelND + BlockAlignedGaussianModulationND.
Readout: GAP (no CLS, no registers).
Batch  : 2048 (8 GPUs × 16 per-GPU × 16 accum steps).
"""

from examples.vit5_imagenet.v6_hierarchical._base_config import (
    build_hyena_hier_net,
    get_base_config,
)
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.callbacks.omega_scale_monitor import OmegaScaleMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


def get_config() -> ExperimentConfig:
    """Return the pure (no-FiLM) hierarchical Hyena ImageNet config."""
    config = get_base_config()
    config.net = build_hyena_hier_net(layout="pure")
    config.callbacks.append(LazyConfig(MaskMonitorCallback)(log_every_n_steps=50))
    config.callbacks.append(LazyConfig(OmegaScaleMonitorCallback)(log_every_n_steps=50))
    config.wandb.job_group = "v6_hier_pure"
    return config
