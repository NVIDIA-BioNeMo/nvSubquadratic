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
    run_start_time: Optional[float] = None  # This is to keep track of the start time of the job
    run_time_limit_hours: Optional[float] = (
        None  # If both run_start_time and run_time_limit_hours are set, the WalltimeCheckpointer will stop training when the time limit is reached. If either is None, no walltime limit is enforced.
    )


@dataclass
class TrainerConfig:
    """Lightning Trainer configuration overrides."""

    # Validate every N training iterations (maps to Lightning's val_check_interval).
    # None = rely on check_val_every_n_epoch only.
    check_val_every_n_iterations: Optional[int] = None

    # Validate every N epochs (Lightning's check_val_every_n_epoch). Default: 1.
    check_val_every_n_epoch: int = 1

    # Run through all validation batches every epoch by default.
    limit_val_batches: Union[int, float] = 1.0

    # Checkpoint saving frequency (in training steps). If None, only save after validation.
    # Recommended: 2000-5000 for long runs to avoid losing progress on crashes.
    checkpoint_every_n_steps: Optional[int] = None

    # Override the metric monitored by ModelCheckpoint. If None, auto-derived
    # from scheduler.mode ("val/acc" for max, "val/loss" for min).
    checkpoint_monitor: Optional[str] = None

    # Enable DDP find_unused_parameters (required when some model parameters
    # are not part of every forward pass, e.g. multi-head CKConv variants).
    find_unused_parameters: bool = False
    # Whether to upload checkpoints to W&B and run cache cleanup.
    # Set to False to disable WandbSelectiveCheckpointUploader and WandbCacheCleanupCallback.
    # Local ModelCheckpoint saving is unaffected by this flag.
    wandb_checkpoint_upload: bool = True


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    name: str = PLACEHOLDER
    warmup_iterations_percentage: float = 0.0
    stable_iterations_percentage: float = (
        0.0  # WSD only: fraction of total iters at constant LR between warmup and decay
    )
    total_iterations: int = PLACEHOLDER
    eta_min: float = 0.0
    mode: str = "max"
    monitor: Optional[str] = None
    # WSD-specific parameters
    decay_iterations_percentage: float = 0.1  # Fraction of training for decay phase
    min_lr_ratio: float = 0.01  # Minimum LR as fraction of peak LR  # in case we'd like to track e.g. val/iou


@dataclass
class WandbConfig:
    """Wandb configuration."""

    project: str = "nvsubquadratic"
    entity: str = "dromeroguzma"

    job_group: str = ""
    run_id: Optional[str] = None  # Explicit W&B run ID for resuming or linking runs


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
class StartFromCheckpointConfig:
    """Configuration to start training from weights of a previously saved checkpoint (weights only, no optimizer/scheduler state)."""

    load: bool = False
    alias: Literal["best", "latest"] = "latest"
    strict: bool = True
    partial_load: bool = False
    run_path: str = ""
    callbacks: list = field(default_factory=list)  # List of LazyConfig callbacks to process state_dict before loading


@dataclass
class ExperimentConfig:
    """Default configuration for experiments with nvSubQuadratic."""

    device: str = "cuda"
    debug: bool = True
    deterministic: bool = False
    seed: int = 0
    comment: str = ""
    compile: bool = False  # Whether to compile the model with torch.compile
    compile_mode: Optional[str] = None  # torch.compile mode: None (default), "reduce-overhead", "max-autotune"
    compile_compatible_fftconv: bool = (
        False  # Use real-valued complex multiply in FFT conv (needed for torch.compile + FFT models)
    )
    experiment_dir: Optional[str] = None
    num_nodes: int = 1

    dataset: LazyConfig = PLACEHOLDER
    net: LazyConfig = PLACEHOLDER
    lightning_wrapper_class: LazyConfig = PLACEHOLDER
    optimizer: LazyConfig = PLACEHOLDER

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    start_from_checkpoint: StartFromCheckpointConfig = field(default_factory=StartFromCheckpointConfig)
    autoresume: AutoResumeConfig = field(default_factory=AutoResumeConfig)
    callbacks: list[LazyConfig] = field(default_factory=list)

    # Layer-wise learning rate decay (LLRD): deeper layers get lower LR.
    # None = disabled; float in (0, 1) = decay factor per layer.
    layer_decay: Optional[float] = None
    num_blocks: Optional[int] = None  # required when layer_decay is set


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
