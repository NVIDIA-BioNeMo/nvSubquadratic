# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

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


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    name: str = PLACEHOLDER
    warmup_iterations_percentage: float = 0.0
    total_iterations: int = PLACEHOLDER
    mode: str = "max"


@dataclass
class WandbConfig:
    """Wandb configuration."""

    project: str = "nvsubquadratic"
    entity: str = "dromeroguzma"
    job_group: str = ""


@dataclass
class AutoResumeConfig:
    """Auto-resume configuration via Weights & Biases run name.

    If enabled, the launcher will:
    - compute a stable run name (no timestamp; optionally includes username),
    - look up an existing W&B run with that exact name under the configured entity/project,
    - assert there is at most one such run,
    - download the checkpoint artifact for `alias` and resume Trainer from it.
    """

    enabled: bool = False
    # Which artifact alias to resume from when found
    alias: Literal["best", "latest"] = "latest"
    # Run name
    run_name: str | None = None


@dataclass
class ResumeFromCheckpointConfig:
    """Configuration to specify wether to start training from a previously saved checkpoint."""

    load: bool = False  # Whether to load the checkpoint
    alias: Literal["best", "latest"] = "latest"  # Either best or latest
    strict: bool = True  # Whether to raise an error if the checkpoint does not exactly match the model architecture
    partial_load: bool = False  # When strict is False, copy overlapping tensor slices from checkpoint into model
    run_path: str = (
        ""  # entity/project/run_id | When set, download checkpoint from this W&B run path (entity/project/run_id)
    )
    output_dir: str = ".artifacts/{run_id}/{alias}"  # Optional output directory to store downloaded artifacts; defaults to .artifacts/{run_id}/{alias}


@dataclass
class ExperimentConfig:
    """Default configuration for experiments with nvSubQuadratic."""

    device: str = "cuda"
    debug: bool = True
    deterministic: bool = False  # Need to be set to True for deterministic behavior
    seed: int = 0
    comment: str = ""
    enable_checkpointing: bool = True # False to disable during development
    enable_progress_bar: bool = True  # False to disable during development

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

    lightning_wrapper_class: Literal[type("ClassificationWrapper"), type("RegressionWrapper")] = PLACEHOLDER

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

    resume_from_checkpoint: ResumeFromCheckpointConfig = field(default_factory=ResumeFromCheckpointConfig)

    # Auto-resume behavior based on W&B run name
    autoresume: AutoResumeConfig = field(default_factory=AutoResumeConfig)

    # Optional: additional Trainer callbacks defined per-experiment and appended during construction
    callbacks: list[LazyConfig] = field(default_factory=list)
