"""v6_hierarchical — Hyena 4-stage with register-row FiLM, BD + learnable ω₀.

Same architecture as ``hyena_hier_p4_pure`` but with 4 register tokens
prepended as the first row of the 2D grid at every stage.  Each block pools
its registers (RegisterPooling) and uses the result to FiLM-modulate the
SIREN kernel.  No CLS token; GAP readout skips the register row.

Stages [96, 192, 384, 768] x depths [2, 2, 6, 2]; effective batch = 2048.
"""

from examples.vit5_imagenet.v6_hierarchical._base_config import (
    build_hyena_hier_net,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Return the FiLM-conditioned hierarchical Hyena config."""
    config = get_base_config()
    config.net = build_hyena_hier_net(layout="register_row")
    return config
