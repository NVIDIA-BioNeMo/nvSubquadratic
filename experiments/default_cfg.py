# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Typed configuration dataclasses for nvSubquadratic experiments.

Every training run is fully described by an :class:`ExperimentConfig`.
These are plain Python :mod:`dataclasses` so they can be instantiated directly
in a Python config file, serialised via OmegaConf, and overridden at the CLI.

**Sub-configs**

- :class:`TrainConfig` — batch size, iterations, gradient clip, wall-time limit.
- :class:`TrainerConfig` — Lightning Trainer overrides (validation frequency,
  checkpoint interval, DDP settings).
- :class:`SchedulerConfig` — schedule name (``"cosine"``, ``"wsd"``,
  ``"constant"``), warmup fraction, total iterations.
- :class:`WandbConfig` — project, entity, run resumption.
- :class:`OptimizerConfig` — optimizer class and hyperparameters.
- :class:`AutoResumeConfig` — automatic checkpoint resumption from local or
  W&B artifact.

Network and Lightning wrapper are specified as
:class:`~nvsubquadratic.lazy_config.LazyConfig` objects (``net_cfg``,
``lightning_wrapper_cfg``) so the full experiment is config-driven.

:data:`PLACEHOLDER` is re-exported from :mod:`nvsubquadratic.lazy_config` for
convenience in config files.

Adapted from https://github.com/implicit-long-convs/ccnn_v2.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig


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

    # Number of training samples per epoch.  Used with
    # OmegaConf resolvers to derive step-based validation intervals via
    # interpolation so that changing batch_size auto-adjusts everything.
    samples_per_epoch: Optional[int] = None

    # Validate every N training iterations (maps to Lightning's val_check_interval).
    # None = rely on check_val_every_n_epoch only.
    check_val_every_n_iterations: Optional[int] = None

    # Validate every N epochs (Lightning's check_val_every_n_epoch). Default: 1.
    check_val_every_n_epoch: int = 1

    # Run through all validation batches every epoch by default.
    limit_val_batches: Union[int, float] = 1.0

    # Run through all test batches by default.
    limit_test_batches: Union[int, float] = 1.0

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
    tags: list = field(default_factory=list)
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
    """Top-level configuration for a single nvSubquadratic training run.

    All fields have sensible defaults; task-specific overrides are specified
    in experiment config files under ``experiments/``.  The config is loaded
    by ``experiments/run.py``, printed as a tree via Rich, and passed to
    :func:`~experiments.trainer.construct_trainer` and the Lightning wrapper.

    **Key optional fields**

    - ``compile = True``: wrap the network with ``torch.compile``.  Mutually
      exclusive with the QuACK kernel path (use ``use_quack=False`` in norm
      layers when compiling).
    - ``experiment_dir``: override the default ``runs/<run_name>/`` checkpoint
      directory with an absolute path.
    - ``num_nodes``: number of multi-node training hosts (passed to the
      Lightning :class:`~pytorch_lightning.Trainer`).
    """

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

    # Multiprocessing sharing strategy for DataLoader workers.
    # Set to "file_system" to avoid /dev/shm exhaustion when running many
    # workers or multiple jobs on the same node.  None keeps the PyTorch
    # default ("file_descriptor" on Linux, which uses /dev/shm).
    mp_sharing_strategy: Optional[str] = None

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
