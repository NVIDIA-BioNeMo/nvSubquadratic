# TODO: Add licence header

# Adapted from https://github.com/implicit-long-convs/ccnn_v2
from typing import Optional
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks as pl_callbacks

from experiments.callbacks.wandb_cache_cleanup import WandbCacheCleanupCallback
from experiments.default_cfg import ExperimentConfig
from experiments.utils.checkpointing import WandbSelectiveCheckpointUploader
from nvsubquadratic.lazy_config import instantiate


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
    if cfg.scheduler.mode == "max":
        monitor = "val/acc"
    elif cfg.scheduler.mode == "min":
        monitor = "val/loss"

    # Derive checkpoint directory based on run name.
    if not experiment_dir is None:
        checkpoint_dir = experiment_dir / "checkpoints"
    else:
        checkpoint_dir = Path("runs") / run_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] Saving checkpoints to: {checkpoint_dir.resolve()}")

    # Callback for model checkpointing:
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor=monitor,
        mode=cfg.scheduler.mode,  # Save on best validation accuracy
        save_top_k=1,
        save_last=True,  # Keep track of the model at the last epoch
        verbose=True,
    )

    # Distributed training params
    assert cfg.device == "cuda", "Only CUDA training is supported."

    device_count = torch.cuda.device_count()
    if device_count > 1:  # Multi-GPU training
        strategy = "ddp"
        sync_batchnorm = True
    else:
        strategy = "auto"
        sync_batchnorm = False
    # num_nodes = 1  # Multi-node training not verified.

    # Merge default callbacks with any experiment-defined callbacks
    user_callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks] if cfg.callbacks else []

    callbacks_list = [
        # Checkpoint callback
        checkpoint_callback,
        # Model summary callback
        pl_callbacks.ModelSummary(max_depth=-1),
        # Learning rate monitor callback
        pl_callbacks.LearningRateMonitor(log_weight_decay=True),
        # Timer callback
        pl_callbacks.Timer(),
        # Progress bar for SLURM/non-TTY environments - prints training progress with it/s
        pl_callbacks.TQDMProgressBar(refresh_rate=10),
        # Wandb selective checkpoint uploader
        WandbSelectiveCheckpointUploader(
            upload_best=True,
            upload_last=True,
            remove_local_after_upload=False,
            keep_last_k_versions=2,
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
        # Append user-defined callbacks
        *user_callbacks,
    ]

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
        val_check_interval=cfg.trainer.val_check_interval,
        limit_val_batches=cfg.trainer.limit_val_batches,
        # Logging frequency
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    return trainer, checkpoint_callback
