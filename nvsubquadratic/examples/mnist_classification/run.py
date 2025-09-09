# David W. Romero, 2025-09-09

"""
Entry point to run the MNIST classification experiment.

Usage:
    python run.py --config config/experiments/mnist/mnist_classification_cfg.py
"""

import argparse
from pathlib import Path
import importlib.util
import torch
import numpy as np
import random
import os
import re
from typing import Any, List

from nvsubquadratic.src.utils.lazy_config import instantiate

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from rich import print as rprint
from rich.tree import Tree
from omegaconf import to_dict
from omegaconf import OmegaConf
from nvsubquadratic.examples.mnist_classification.utils import load_config_from_file, verify_no_interpolator_overwrites, apply_config_overrides, set_global_seed
from nvsubquadratic.examples.mnist_classification.lightning_wrappers import ClassificationWrapper


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



def construct_trainer(
    cfg: OmegaConf,
) -> tuple[pl.Trainer, pl.Callback]:
    """Construct a trainer and the checkpoint callback from a configuration.

    Args:
        cfg (OmegaConf): The configuration.
        wandb_logger (pl.loggers.WandbLogger): The wandb logger.

    Returns:
        tuple[pl.Trainer, pl.Callback]: The constructed trainer and the checkpoint callback.
    """
    # Set up determinism
    deterministic = False
    benchmark = True

    # Callback to print model summary
    modelsummary_callback = pl.callbacks.ModelSummary(
        max_depth=-1,
    )

    # Metric to monitor
    monitor = "val/acc"
    monitor_mode = "max"

    # Callback for model checkpointing:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=monitor_mode,  # Save on best validation accuracy
        save_top_k=1,
        save_last=True,  # Keep track of the model at the last epoch
        verbose=True,
    )

    # Callback for learning rate monitoring
    lrmonitor_callback = pl.callbacks.LearningRateMonitor(log_weight_decay=True)

    # Callback for timing information
    timer_callback = pl.callbacks.Timer()

    # Distributed training params
    assert cfg.device == "cuda", "Only CUDA training is supported."

    sync_batchnorm = cfg.train.distributed and torch.cuda.device_count() != 1
    strategy = "ddp_find_unused_parameters_false" if cfg.train.distributed else "auto"
    gpus = torch.cuda.device_count() if cfg.train.distributed else 1
    num_nodes = 1


    callbacks_list = [
        modelsummary_callback,
        lrmonitor_callback,
        checkpoint_callback,
        timer_callback,
    ]

    # create trainer
    trainer = pl.Trainer(
        max_steps=cfg.train.iterations,
        # Callbacks
        callbacks=callbacks_list,
        # Multi-GPU
        num_nodes=num_nodes,
        devices=gpus,
        strategy=strategy,
        sync_batchnorm=sync_batchnorm,
        # Precision
        precision=cfg.train.precision,
        # Determinism
        deterministic=deterministic,
        benchmark=benchmark,
    )
    return trainer, checkpoint_callback



def main():
    # Parse command line arguments
    args = parse_args()

    # Load configuration from file
    config = load_config_from_file(args.config)

    # Validate that overrides do not target interpolated fields, then apply
    verify_no_interpolator_overwrites(config, args.overrides)
    config = apply_config_overrides(config, args.overrides)

    # Set seed
    set_global_seed(config.seed)

    torch.set_float32_matmul_precision("high")

    # Construct data_module, prepare and setup
    datamodule = instantiate(config.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    # Construct model
    network = instantiate(config.net, in_channels=datamodule.input_channels, out_channels=datamodule.output_channels)
    # Wrap network in a pl.LightningModule
    model = ClassificationWrapper(network, config)

    # Print the config files prior to training
    print(f"Config:\n {config}")

    # Create trainer
    trainer, checkpoint_callback = construct_trainer(config)

    # Test before training
    if config.test.before_train:
        trainer.validate(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

    # Train
    if config.train.do:
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


if __name__ == "__main__":
    main()
