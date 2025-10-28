# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Default configuration for experiments with nvSubQuadratic."""

from dataclasses import dataclass, field
from typing import Literal, Optional

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
class TrainerConfig:
    """Lightning Trainer configuration overrides."""

    val_check_interval: Optional[float] = None


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
    alias: Literal["best", "latest"] = "latest"
    run_name: str | None = None


@dataclass
class ResumeFromCheckpointConfig:
    """Configuration to specify whether to start training from a previously saved checkpoint."""

    load: bool = False
    alias: Literal["best", "latest"] = "latest"
    strict: bool = True
    partial_load: bool = False
    run_path: str = ""
    output_dir: str = ".artifacts/{run_id}/{alias}"


@dataclass
class DiffusionScheduleConfig:
    """Noise schedule configuration."""

    num_train_timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    time_embed_dim: Optional[int] = None
    max_period: float = 10_000.0


@dataclass
class DiffusionSamplingConfig:
    """Sampling loop configuration."""

    num_inference_steps: int = 50
    num_samples: int = 4
    log_samples: bool = True


@dataclass
class DiffusionEMAConfig:
    """Exponential moving average configuration."""

    enabled: bool = False
    decay: float = 0.999
    update_every: int = 1
    warmup_steps: int = 0


@dataclass
class DiffusionConfig:
    """Grouped configuration for diffusion wrappers."""

    schedule: DiffusionScheduleConfig = field(default_factory=DiffusionScheduleConfig)
    sampling: DiffusionSamplingConfig = field(default_factory=DiffusionSamplingConfig)
    ema: DiffusionEMAConfig = field(default_factory=DiffusionEMAConfig)


@dataclass
class ExperimentConfig:
    """Default configuration for experiments with nvSubQuadratic."""

    device: str = "cuda"
    debug: bool = True
    deterministic: bool = False
    seed: int = 0
    comment: str = ""

    dataset: LazyConfig = PLACEHOLDER
    net: LazyConfig = PLACEHOLDER
    lightning_wrapper_class: LazyConfig = PLACEHOLDER
    optimizer: LazyConfig = PLACEHOLDER

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    resume_from_checkpoint: ResumeFromCheckpointConfig = field(default_factory=ResumeFromCheckpointConfig)
    autoresume: AutoResumeConfig = field(default_factory=AutoResumeConfig)
    callbacks: list[LazyConfig] = field(default_factory=list)


@dataclass
class DiffusionExperimentConfig(ExperimentConfig):
    """Experiment configuration for diffusion runs."""

    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
