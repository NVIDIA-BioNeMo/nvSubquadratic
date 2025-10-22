# TODO: Add license header here


"""Entry point to run experiments.

Usage:
    # MNIST classification
    PYTHONPATH=. python nvsubquadratic/examples/run.py --config examples/mnist_classification/experiments/mnist_classification_ccnn_4_160_hyena_rope_qknorm.py
"""

import argparse
import dataclasses

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
    trainer, checkpoint_callback = construct_trainer(config, wandb_logger)

    # Train
    if config.train.do:
        # TODO(@dwromero): Add support for training resume.
        trainer.fit(model=model, datamodule=datamodule)
        # Load state dict from best performing model
        model.load_state_dict(
            torch.load(checkpoint_callback.best_model_path)["state_dict"],
        )

    # Validate and test before finishing
    trainer.validate(
        model,
        datamodule=datamodule,
    )
    trainer.test(
        model,
        datamodule=datamodule,
    )


def construct_trainer(
    cfg: ExperimentConfig,
    wandb_logger: WandbLogger,
) -> tuple[pl.Trainer, pl.Callback]:
    """Construct a trainer and the checkpoint callback from a configuration.

    Args:
        cfg (ExperimentConfig): The configuration.
        wandb_logger (pl.loggers.WandbLogger): The wandb logger.

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

    # Callback to print model summary
    modelsummary_callback = pl_callbacks.ModelSummary(
        max_depth=-1,
    )

    # Metric to monitor
    if cfg.scheduler.mode == "max":
        monitor = "val/acc"
    elif cfg.scheduler.mode == "min":
        monitor = "val/loss"

    # Callback for model checkpointing:
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=cfg.scheduler.mode,  # Save on best validation accuracy
        save_top_k=1,
        save_last=True,  # Keep track of the model at the last epoch
        verbose=True,
    )

    # Callback for learning rate monitoring
    lrmonitor_callback = pl_callbacks.LearningRateMonitor(log_weight_decay=True)

    # Callback for timing information
    timer_callback = pl_callbacks.Timer()

    # Distributed training params
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but no GPU is available.")
        device_count = torch.cuda.device_count()
        if device_count < 1:
            raise RuntimeError("CUDA device requested but torch.cuda.device_count() returned 0.")
        if device_count > 1:  # Multi-GPU training
            strategy = "ddp"
            sync_batchnorm = True
            devices_arg: int | list[int] = list(range(device_count))
        else:
            strategy = "auto"
            sync_batchnorm = False
            devices_arg = 1
        accelerator = "gpu"
    elif cfg.device == "cpu":
        strategy = "auto"
        sync_batchnorm = False
        devices_arg = 1
        accelerator = "cpu"
        device_count = 1
    else:
        raise ValueError(f"Unsupported device type: {cfg.device}")

    num_nodes = 1  # Multi-node training not verified.

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
        accelerator=accelerator,
        num_nodes=num_nodes,
        devices=devices_arg,
        strategy=strategy,
        sync_batchnorm=sync_batchnorm,
        # Precision
        precision=cfg.train.precision,
        # Determinism
        deterministic=deterministic,
        benchmark=benchmark,
    )
    return trainer, checkpoint_callback


if __name__ == "__main__":
    main()
