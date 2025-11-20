# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Default configuration for experiments with nvSubQuadratic."""

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

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

    # Check once every epoch by default.
    val_check_interval: float = 1.0

    # Run through all validation batches every epoch by default.
    limit_val_batches: Union[int, float] = 1.0


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    name: str = PLACEHOLDER
    warmup_iterations_percentage: float = 0.0
    total_iterations: int = PLACEHOLDER
    mode: str = "max"
    monitor: Optional[str] = None  # in case we'd like to track e.g. val/iou


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
class ExperimentConfig:
    """Default configuration for experiments with nvSubQuadratic."""

    device: str = "cuda"
    debug: bool = True
    deterministic: bool = False
    seed: int = 0
    comment: str = ""
    compile: bool = False  # Whether to compile the model with torch.compile

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
class DiffusionConfig:
    """Diffusion configuration for schedule, sampling, and EMA."""

    num_train_timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine_interpolated"  # one of "linear", "scaled_linear", "cosine", "cosine_interpolated"
    cosine_schedule_logsnr_min: float = -10.0
    cosine_schedule_logsnr_max: float = 10.0
    cosine_schedule_image_resolution: int = 64
    cosine_schedule_noise_res_low: int = 32
    cosine_schedule_noise_res_high: int = 64
    prediction_type: str = "v_prediction"  # one of "epsilon", "v_prediction", "sample"
    time_embed_dim: Optional[int] = None
    max_period: float = 10_000.0

    num_inference_steps: int = 150
    num_samples: int = 25
    log_samples: bool = True
    ddim_eta: float = 0.0

    use_sigmoid_loss_weighting: bool = True
    sigmoid_loss_bias: float = -1.0

    ema_enabled: bool = True
    ema_decay: float = 0.9995
    ema_update_every: int = 1
    ema_warmup_steps: int = 5_000

    # Classifier-free guidance settings, enabled by default.
    use_classifier_free_guidance: bool = True
    guidance_scale: float = 3.5
    condition_dropout_prob: float = 0.1
    num_classes: Optional[int] = 1000

    # Online evaluation knobs.
    fid_enabled: bool = False
    fid_num_batches: int = 0
    fid_num_inference_steps: Optional[int] = None


@dataclass
class DiffusionExperimentConfig(ExperimentConfig):
    """Experiment configuration for diffusion runs."""

    # Override debug mode.
    debug: bool = False

    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
