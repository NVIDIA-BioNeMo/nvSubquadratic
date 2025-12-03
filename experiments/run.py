# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

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
from pytorch_lightning.loggers import WandbLogger
from rich import print as rprint
from rich.tree import Tree

import wandb
from experiments.trainer import construct_trainer
from experiments.utils.checkpointing import (
    download_checkpoint,
    load_checkpoint_state_dict,
    load_state_dict_partially,
    preview_state_dict_compatibility,
)

from experiments.utils.cli import (
    add_to_tree,
    apply_config_overrides,
    config_to_dict_for_rich,
    get_deterministic_run_name,
    load_config_from_file,
    verify_no_interpolator_overwrites,
)
from nvsubquadratic.lazy_config import instantiate


torch._dynamo.config.cache_size_limit = 32


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the experiment.

    Sets up and parses arguments for the configuration file path and any command-line overrides.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments. Includes 'config' for the
                            configuration file path and 'overrides' for any specified configuration overrides.
    """
    parser = argparse.ArgumentParser(description="MNIST Classification Training")

    # Config file path
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file, e.g., config/experiments/mnist/mnist_classification_cfg.py",
    )
    
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=False,
        default=None,
        help="Path to the experiment directory, e.g., workspace/results. If not provided, the run name is used to create the checkpoint directory.",
    )
    
    parser.add_argument(
        "--num_nodes",
        type=int,
        required=False,
        default=1,
        help="Number of nodes to use for training, default is 1",
    )

    # Add a catch-all for arbitrary config overrides
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides, e.g., dataset.batch_size=64",
    )

    return parser.parse_args()


def main() -> None:
    """Main function to run the experiment.

    This function orchestrates the entire experiment lifecycle, including:
    1.  Parsing command-line arguments.
    2.  Loading and overriding configuration from files and command line.
    3.  Setting up the environment, including seeding for reproducibility and configuring torch settings.
    4.  Instantiating the data module, network model, and the Lightning wrapper.
    5.  Setting up the Weights & Biases logger, with support for auto-resuming runs.
    6.  Handling checkpoint loading for resuming training or fine-tuning.
    7.  Constructing the PyTorch Lightning trainer with appropriate callbacks.
    8.  Executing the training, validation, and testing phases of the experiment.
    """
    # Parse command line arguments
    args = parse_args()
    
    num_nodes = args.num_nodes

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

    # Compile the model if specified
    if config.compile:
        print("Compiling model with torch.compile...")
        network = torch.compile(network)

    # Wrap network in a pl.LightningModule
    model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)

    # Initialize wandb logger
    if config.debug:
        log_model = False
        offline = True
    else:
        # Avoid auto logging all checkpoints; selective uploader handles best/last
        log_model = False
        offline = False

    if config.autoresume.enabled:
        # If run name is not provided, use the deterministic run name without timestamp
        run_name = (
            get_deterministic_run_name(args.config, args.overrides, use_timestamp=False)
            if config.autoresume.run_name is None
            else config.autoresume.run_name
        )
    else:
        # Use the deterministic run name with timestamp
        run_name = get_deterministic_run_name(args.config, args.overrides, use_timestamp=True)
    
    
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir is not None else Path("runs") / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    autoresume_ckpt_path = None
    run_id_file = experiment_dir / "run.id"
    if run_id_file.exists():
        attach_run_id = run_id_file.read_text().strip()
    else:
        raise RuntimeError(f"[autoresume] No run ID file found in experiment directory '{experiment_dir}'.")
    
    if config.autoresume.enabled:
        wandb_logger = WandbLogger(
            project=config.wandb.project,
            entity=config.wandb.entity,
            save_dir=experiment_dir,
            id=attach_run_id,
            resume="allow",
            name=run_name,
            log_model=log_model,
            offline=offline,
            save_code=True,
            group=config.wandb.job_group,
        )
    else:
        wandb_logger = WandbLogger(
            project=config.wandb.project,
            entity=config.wandb.entity,
            save_dir=experiment_dir,
            id=attach_run_id,
            resume="allow",
            name=run_name,
            config=dataclasses.asdict(config),  # Convert dataclass config to dict
            log_model=log_model,
            offline=offline,
            save_code=True,
            group=config.wandb.job_group,
        )

    ckpt_dir = experiment_dir / "checkpoints"
    
    if ckpt_dir.exists():
        last_path = ckpt_dir / "last.ckpt"
        if last_path.exists():
            autoresume_ckpt_path = last_path
            print(f"Resuming from {autoresume_ckpt_path}")
        else:
            print(f"No last.ckpt found in {ckpt_dir}, starting from scratch.")   

    # Recreate the command that instantiated this run.
    if isinstance(wandb_logger.experiment.settings, wandb.Settings):
        command = f"python run.py --config {args.config}"
        if args.overrides:
            command += " " + " ".join(args.overrides)
        # Log the command.
        wandb_logger.experiment.config.update({"command": command}, allow_val_change=True)

    # Print the config files prior to training
    config_dict = config_to_dict_for_rich(config)
    tree = Tree("Configuration")
    add_to_tree(tree, config_dict)
    rprint(tree)
    
    # Create trainer
    trainer, checkpoint_callback = construct_trainer(config, wandb_logger, run_name, experiment_dir, num_nodes)

    # Validate that the checkpoint has been correctly loaded before training (for no autoresume)
    if autoresume_ckpt_path is None and config.resume_from_checkpoint.load:
        print("[resume] Running validation to verify loaded checkpoint...")
        trainer.validate(model, datamodule=datamodule)
        print("[resume] Validation after resume completed.")

    # Train
    if config.train.do:
        # Fit with full-state resume if autoresume provided a checkpoint, otherwise it will act as if no autoresume_ckpt_path passed in (it's None).
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=autoresume_ckpt_path)
        # Load state dict from best performing model when available
        best_ckpt_path = getattr(checkpoint_callback, "best_model_path", None)
        if best_ckpt_path:
            best_ckpt_path = str(best_ckpt_path)
        if best_ckpt_path and os.path.isfile(best_ckpt_path):
            model.load_state_dict(torch.load(best_ckpt_path)["state_dict"])
        else:
            print(f"[checkpoint] Skipping weight reload; best checkpoint not found (path={best_ckpt_path!r}).")

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
