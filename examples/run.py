# TODO: Add license header here


"""Entry point to run experiments.

Usage:
    # MNIST classification
    PYTHONPATH=. python nvsubquadratic/examples/run.py --config examples/mnist_classification/experiments/mnist_classification_ccnn_4_160_hyena_rope_qknorm.py
"""

import argparse
import dataclasses
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.loggers import WandbLogger
from rich import print as rprint
from rich.tree import Tree

import wandb
from examples.default_cfg import ExperimentConfig
from examples.utils import (
    add_to_tree,
    apply_config_overrides,
    config_to_dict_for_rich,
    get_deterministic_run_name,
    load_config_from_file,
    verify_no_interpolator_overwrites,
)
from nvsubquadratic.lazy_config import instantiate


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MNIST Classification Training")

    # Config file path
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file, e.g., config/experiments/mnist/mnist_classification_cfg.py",
    )

    # Checkpoint path for resuming training
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g., lightning_logs/version_0/checkpoints/last.ckpt)",
    )

    # Gradient logging for testing (optional)
    parser.add_argument(
        "--log_gradients",
        type=str,
        default=None,
        help="Directory to save gradient statistics for testing (default: disabled)",
    )
    parser.add_argument(
        "--gradient_log_steps",
        type=int,
        default=1,
        help="Log gradients every N steps (default: 1)",
    )

    # Experiment directory for organizing outputs
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Directory for checkpoints and logs (default: ./experiments/{run_name})",
    )

    # Add a catch-all for arbitrary config overrides
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides, e.g., dataset.batch_size=64",
    )

    return parser.parse_args()


def main():
    """Main function to run the MNIST classification experiment."""
    # Parse command line arguments
    args = parse_args()

    # Load configuration from file
    config = load_config_from_file(args.config)

    # Validate that overrides do not target interpolated fields, then apply
    verify_no_interpolator_overwrites(config, args.overrides)
    config = apply_config_overrides(config, args.overrides)

    # Set seed
    pl.seed_everything(config.seed, workers=True)

    # Set deterministic mode
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cudnn.benchmark = not config.deterministic

    # Set float32 matmul precision
    torch.set_float32_matmul_precision("high")

    # Construct data_module, prepare and setup
    datamodule = instantiate(config.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    # Construct model
    network = instantiate(config.net, in_channels=datamodule.input_channels, out_channels=datamodule.output_channels)
    # Wrap network in a pl.LightningModule
    model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)

    # Initialize wandb logger
    if config.debug:
        log_model = False
        offline = True
    else:
        log_model = True
        offline = False
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=get_deterministic_run_name(args.config, args.overrides),  # Use our deterministic run name with overrides
        config=dataclasses.asdict(config),  # Convert dataclass config to dict
        log_model=log_model,  # used to save models to wandb during training
        offline=offline,
        save_code=True,
        group=config.wandb.job_group,
    )

    # Recreate the command that instantiated this run for reproducibility.
    if isinstance(wandb_logger.experiment.settings, wandb.Settings):
        command = f"python examples/run.py --config {args.config}"
        if args.overrides:
            command += " " + " ".join(args.overrides)
        # Log the command.
        wandb_logger.experiment.config.update({"command": command})

    # Print the config files prior to training
    config_dict = config_to_dict_for_rich(config)
    tree = Tree("Configuration")
    add_to_tree(tree, config_dict)
    rprint(tree)

    # Create trainer
    trainer, checkpoint_callback = construct_trainer(config, wandb_logger, experiment_dir=args.experiment_dir)

    # Add gradient logging callback if requested (for testing)
    if args.log_gradients:
        from nvsubquadratic.testing.callbacks import GradientLoggingCallback

        gradient_callback = GradientLoggingCallback(
            save_dir=Path(args.log_gradients),
            log_every_n_steps=args.gradient_log_steps,
            max_steps=args.gradient_log_steps,
        )
        trainer.callbacks.append(gradient_callback)
        print(f"Gradient logging enabled: saving to {args.log_gradients} every {args.gradient_log_steps} steps")

    # Train
    if config.train.do:
        # Resume from checkpoint if provided
        if args.ckpt_path:
            print(f"Resuming training from checkpoint: {args.ckpt_path}")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=args.ckpt_path)
        else:
            trainer.fit(model=model, datamodule=datamodule)

        # Load state dict from best performing model
        if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
            model.load_state_dict(
                torch.load(checkpoint_callback.best_model_path)["state_dict"],
            )

    # Validate and test before finishing
    if config.validate:
        trainer.validate(
            model,
            datamodule=datamodule,
        )
    if config.test:
        trainer.test(
            model,
            datamodule=datamodule,
        )


def construct_trainer(
    cfg: ExperimentConfig,
    wandb_logger: WandbLogger,
    experiment_dir: str | None = None,
) -> tuple[pl.Trainer, pl.Callback]:
    """Construct a trainer and the checkpoint callback from a configuration.

    Args:
        cfg (ExperimentConfig): The configuration.
        wandb_logger (pl.loggers.WandbLogger): The wandb logger.
        experiment_dir (str): Optional experiment directory path.

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

    # Set experiment directory for checkpoints and logs
    # This keeps all experiment artifacts (checkpoints, logs, wandb) organized
    if experiment_dir is None:
        # Auto-create: ./experiments/{wandb_run_name}/
        experiment_dir = f"./experiments/{wandb_logger.experiment.name}"

    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")

    # Callback to print model summary
    modelsummary_callback = pl_callbacks.ModelSummary(
        max_depth=-1,
    )

    # Metric to monitor
    if cfg.scheduler.mode == "max":
        monitor = "val/acc"
    elif cfg.scheduler.mode == "min":
        monitor = "val/loss"

    # Callback for model checkpointing
    # Note: Currently using standard Lightning checkpoints for reliability
    # Megatron distributed checkpoints can be enabled later via use_distributed_checkpoint config
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=experiment_dir,
        monitor=monitor,
        mode=cfg.scheduler.mode,  # Save on best validation accuracy
        save_top_k=1,
        save_last=True,  # Keep track of the model at the last epoch
        verbose=True,
    )
    print(f"Using ModelCheckpoint (dirpath={experiment_dir}, monitor={monitor}, mode={cfg.scheduler.mode})")

    # Callback for learning rate monitoring
    lrmonitor_callback = pl_callbacks.LearningRateMonitor(log_weight_decay=True)

    # Callback for timing information
    timer_callback = pl_callbacks.Timer()

    # Distributed training params
    assert cfg.device == "cuda", "Only CUDA training is supported."

    device_count = torch.cuda.device_count()
    num_nodes = 1  # Multi-node training not verified.

    # Configure distributed strategy
    if cfg.distributed.enabled and cfg.distributed.context_parallel_size > 1:
        # Use Context Parallel strategy
        from nvsubquadratic.strategies import ContextParallelStrategy

        strategy = ContextParallelStrategy(
            backend_type=cfg.distributed.backend,
            context_parallel_size=cfg.distributed.context_parallel_size,
            tensor_parallel_size=cfg.distributed.tensor_parallel_size,
            pipeline_parallel_size=cfg.distributed.pipeline_parallel_size,
            use_distributed_checkpoint=cfg.distributed.use_distributed_checkpoint,
            checkpoint_dir=cfg.distributed.checkpoint_dir,
        )
        sync_batchnorm = True
        print(
            f"Using Context Parallel strategy: "
            f"backend={cfg.distributed.backend}, "
            f"CP size={cfg.distributed.context_parallel_size}"
        )
    elif device_count > 1:  # Standard multi-GPU training
        strategy = "ddp"
        sync_batchnorm = True
    else:
        strategy = "auto"
        sync_batchnorm = False

    # Merge default callbacks with any experiment-defined callbacks
    user_callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks] if cfg.callbacks else []

    callbacks_list = [
        modelsummary_callback,
        lrmonitor_callback,
        checkpoint_callback,
        timer_callback,
        # Append any experiment-defined callbacks
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
        # Default root directory (prevents checkpoint dirs in source code)
        default_root_dir=experiment_dir,
    )
    return trainer, checkpoint_callback


if __name__ == "__main__":
    main()
