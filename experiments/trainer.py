# TODO: Add licence header

# Adapted from https://github.com/implicit-long-convs/ccnn_v2
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks as pl_callbacks

from experiments.callbacks.walltime_checkpointer import WalltimeCheckpointer
from experiments.callbacks.wandb_cache_cleanup import WandbCacheCleanupCallback
from experiments.default_cfg import ExperimentConfig
from experiments.utils.checkpointing import WandbSelectiveCheckpointUploader
from nvsubquadratic.lazy_config import instantiate


def _scheduler_phase_boundaries(cfg: ExperimentConfig) -> dict[str, tuple[int, int]]:
    """Derive per-phase step boundaries from the scheduler config.

    Returns a mapping ``{phase_name: (start_step, end_step)}`` suitable for
    :class:`WandbSelectiveCheckpointUploader`.  Warmup is excluded because it
    is typically too short to warrant dedicated checkpoints.
    """
    sched = cfg.scheduler
    total = sched.total_iterations
    if total is None or total <= 0:
        return {}
    warmup_end = int(sched.warmup_iterations_percentage * total)

    name = getattr(sched, "name", None)
    if name == "wsd":
        stable_pct = getattr(sched, "stable_iterations_percentage", 0.0)
        stable_end = warmup_end + int(stable_pct * total)
        return {"stable": (warmup_end, stable_end), "decay": (stable_end, total)}
    if name == "cosine":
        return {"cosine": (warmup_end, total)}
    if name == "constant":
        return {"constant": (warmup_end, total)}
    return {}


def construct_trainer(
    cfg: ExperimentConfig,
    wandb_logger: pl.loggers.WandbLogger,
    run_name: str,
    experiment_dir: Optional[Path] = None,
    num_nodes: int = 1,
    #
) -> tuple[pl.Trainer, pl.Callback]:
    """Construct a trainer and the checkpoint callback from a configuration.

    Args:
        cfg (ExperimentConfig): The configuration.
        wandb_logger (pl.loggers.WandbLogger): The wandb logger.
        run_name (str): The run name, used only if experiment_dir is not provided.
        experiment_dir (Optional[Path]): The experiment directory. If not provided, the run name is used to create the checkpoint directory.
        num_nodes (int): The number of nodes to use for training.

    Returns:
        tuple[pl.Trainer, pl.Callback]: The constructed trainer and the checkpoint callback.
    """
    # Set up determinism
    if cfg.deterministic:
        deterministic = True
        benchmark = False
    else:
        deterministic = False
        benchmark = True

    # Metric to monitor
    if cfg.trainer.checkpoint_monitor is not None:
        monitor = cfg.trainer.checkpoint_monitor
    elif cfg.scheduler.mode == "max":
        monitor = "val/acc"
    elif cfg.scheduler.mode == "min":
        monitor = "val/loss"

    # Derive checkpoint directory based on run name.
    if experiment_dir is not None:
        checkpoint_dir = experiment_dir / "checkpoints"
    else:
        checkpoint_dir = Path("runs") / run_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] Saving checkpoints to: {checkpoint_dir.resolve()}")

    # Callback for model checkpointing:
    checkpoint_kwargs = {
        "dirpath": str(checkpoint_dir),
        "monitor": monitor,
        "mode": cfg.scheduler.mode,  # Save on best validation accuracy
        "save_top_k": 1,
        "save_last": True,  # Keep track of the model at the last epoch
        "verbose": True,
    }
    # Add step-based checkpointing if configured (useful for long runs to avoid losing progress)
    if cfg.trainer.checkpoint_every_n_steps is not None:
        checkpoint_kwargs["every_n_train_steps"] = cfg.trainer.checkpoint_every_n_steps
        print(f"[checkpoint] Saving every {cfg.trainer.checkpoint_every_n_steps} steps")
    checkpoint_callback = pl_callbacks.ModelCheckpoint(**checkpoint_kwargs)

    # Distributed training params
    assert cfg.device == "cuda", "Only CUDA training is supported."

    device_count = torch.cuda.device_count()
    if device_count > 1:  # Multi-GPU training
        if cfg.trainer.find_unused_parameters:
            strategy = "ddp_find_unused_parameters_true"
        else:
            strategy = "ddp"
        sync_batchnorm = True
    else:
        strategy = "auto"
        sync_batchnorm = False
    # num_nodes = 1  # Multi-node training not verified.

    # Merge default callbacks with any experiment-defined callbacks
    user_callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks] if cfg.callbacks else []

    callbacks_list = [
        # Checkpoint callback (local saving — always enabled)
        checkpoint_callback,
        # Model summary callback
        pl_callbacks.ModelSummary(max_depth=-1),
        # Learning rate monitor callback
        pl_callbacks.LearningRateMonitor(log_weight_decay=True),
        # Timer callback
        pl_callbacks.Timer(),
        # Progress bar for SLURM/non-TTY environments - prints training progress with it/s
        pl_callbacks.TQDMProgressBar(refresh_rate=10),
        # Append user-defined callbacks
        *user_callbacks,
    ]

    # Optionally add W&B checkpoint upload and cache cleanup callbacks
    if cfg.trainer.wandb_checkpoint_upload:
        callbacks_list.extend(
            [
                # Wandb selective checkpoint uploader
                WandbSelectiveCheckpointUploader(
                    upload_best=True,
                    upload_last=True,
                    remove_local_after_upload=False,
                    keep_last_k_versions=2,
                    phase_boundaries=_scheduler_phase_boundaries(cfg),
                    mode=cfg.scheduler.mode,
                ),
                # Wandb cache cleanup callback to prevent W&B cache from growing too large (Disk Space OOM errors)
                WandbCacheCleanupCallback(
                    max_cache_size="5GB",
                    every_n_epochs=2,
                    executable="wandb",
                    run_on_fit_start=True,
                    background=True,
                    timeout=60,
                ),
            ]
        )

    if cfg.train.run_start_time is not None and cfg.train.run_time_limit_hours is not None:
        callbacks_list.append(
            WalltimeCheckpointer(
                start_time=cfg.train.run_start_time,
                time_limit_hours=cfg.train.run_time_limit_hours,
                buffer_minutes=5.0,
                checkpoint_dir=checkpoint_dir,
            )
        )

    # create trainer
    trainer = pl.Trainer(
        max_steps=cfg.train.iterations,
        logger=wandb_logger,
        gradient_clip_val=cfg.train.grad_clip,
        accumulate_grad_batches=cfg.train.accumulate_grad_steps,
        # Callbacks
        callbacks=callbacks_list,
        # Multi-GPU
        num_nodes=num_nodes,
        devices=list(range(device_count)),  # [0, ..., device_count-1]
        strategy=strategy,
        sync_batchnorm=sync_batchnorm,
        # Precision
        precision=cfg.train.precision,
        # Determinism
        deterministic=deterministic,
        benchmark=benchmark,
        val_check_interval=cfg.trainer.check_val_every_n_iterations,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        # Logging frequency
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    return trainer, checkpoint_callback
