# David W. Romero, 2025-09-09

"""Default configuration for experiments with nvSubQuadratic."""

from dataclasses import dataclass, field
from typing import Literal

from nvsubquadratic.lazy_config import LazyConfig


PLACEHOLDER = None


@dataclass
class TrainConfig:
    """Train configuration."""

    do: bool = True
    precision: str = "32-true"
    iterations: int = -1
    batch_size: int = -1
    grad_clip: float = 0.0
    track_grad_norm: int = -1  # -1 for no tracking
    accumulate_grad_steps: int = 1  # Accumulate gradient over different batches
    distributed: bool = False
    num_nodes: int = -1
    avail_gpus: int = -1


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    name: str = PLACEHOLDER
    warmup_iterations: int = 0
    total_iterations: int = PLACEHOLDER
    mode: str = "max"


@dataclass
class WandbConfig:
    """Wandb configuration."""

    project: str = "nvsubquadratic"
    entity: str = "dromeroguzma"
    job_group: str = ""


@dataclass
class ExperimentConfig:
    """Default configuration for experiments with nvSubQuadratic."""

    device: str = "cuda"
    debug: bool = True
    deterministic: bool = False  # Need to be set to True for deterministic behavior
    seed: int = 0

    # Dataset configuration that MUST be set in experiment config
    # This should be instantiated with a LazyConfig object, e.g.:
    #   config.dataset = LazyConfig("datamodules.mnist.MNISTDataModule", {
    #       "data_dir": "/data",
    #       "batch_size": 32,
    #       "permuted": False
    #   })
    dataset: LazyConfig = PLACEHOLDER  # Must be resolved in the experiment config.

    # Network configuration that MUST be set in experiment config
    # This should be instantiated with a LazyConfig object, e.g.:
    #   config.net = LazyConfig(ResNet)(
    #       in_channels=1,
    #       out_channels=10,
    #       num_blocks=4,
    #       ...
    #   })
    net: LazyConfig = PLACEHOLDER

    lightning_wrapper_class: Literal[
        "examples.lightning_wrappers.ClassificationWrapper", "examples.lightning_wrappers.RegressionWrapper"
    ] = PLACEHOLDER

    # Base optimizer MUST be set in experiment config
    # This should be instantiated with a LazyConfig object, e.g.:
    #   config.optimizer = LazyConfig(torch.optim.Adam)(
    #       lr=0.01,
    #       weight_decay=1e-6,
    #   )
    optimizer: LazyConfig = PLACEHOLDER

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Optional: additional Trainer callbacks defined per-experiment and appended during construction
    callbacks: list[LazyConfig] = field(default_factory=list)
