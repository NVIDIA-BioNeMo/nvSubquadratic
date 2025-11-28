# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Profiling script for experiments.

Usage:
    torchrun --standalone --nproc_per_node=1 -m experiments.profile --config examples/pde_regression/swe/cfg.py
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from rich import print as rprint
from rich.tree import Tree

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

# Profiler settings
WAIT, WARMUP, ACTIVE, REPEAT = 10, 11, 10, 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the experiment.

    Sets up and parses arguments for the configuration file path and any command-line overrides.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments. Includes 'config' for the
                            configuration file path and 'overrides' for any specified configuration overrides.
    """
    parser = argparse.ArgumentParser(description="Profile Training")

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


def main() -> None:
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

    # Compile the model if specified
    if config.compile:
        print("Compiling model with torch.compile...")
        network = torch.compile(network)

    # Wrap network in a pl.LightningModule
    model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)

    # Print the config files prior to training
    config_dict = config_to_dict_for_rich(config)
    tree = Tree("Configuration")
    add_to_tree(tree, config_dict)
    rprint(tree)

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Configure optimizer and scheduler manually (since we're not using trainer.fit)
    optimizer_dict = model.configure_optimizers()
    optimizer = optimizer_dict['optimizer']
    scheduler = optimizer_dict.get('lr_scheduler', {}).get('scheduler', None)

    # Get training dataloader
    train_dataloader = datamodule.train_dataloader()
    train_iterator = iter(train_dataloader)

    # Setup profiler output directory (in repo root, not tracked by git)
    rank = int(os.environ.get('RANK', 0))
    repo_root = Path(__file__).parent.parent  # Go up from experiments/ to repo root
    logdir = repo_root / "profile_results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"Starting profiler. Results will be saved to: {logdir}")
    print(f"Profiler schedule: wait={WAIT}, warmup={WARMUP}, active={ACTIVE}, repeat={REPEAT}")
    print(f"Total steps to run: {(WAIT + WARMUP + ACTIVE) * REPEAT}")

    # Run profiling with manual training loop
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(
            wait=WAIT,
            warmup=WARMUP,
            active=ACTIVE,
            repeat=REPEAT
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(logdir), worker_name=f'rank{rank}'),
        record_shapes=True,
        profile_memory=False,
        with_stack=True
    ) as p:
        model.train()

        for step in range((WAIT + WARMUP + ACTIVE) * REPEAT):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)

            # Move batch to device with non_blocking transfer (data is in pinned memory from dataloader)
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device, non_blocking=True)
            elif isinstance(batch, (list, tuple)):
                batch = [b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Training step (returns dict with 'loss' key)
            output = model.training_step(batch, step)
            loss = output['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Step scheduler if available
            if scheduler is not None:
                scheduler.step()

            # Step profiler
            p.step()

            if step % 10 == 0:
                print(f"Step {step}/{(WAIT + WARMUP + ACTIVE) * REPEAT}, loss: {loss.item():.4f}")

    print(f"Profiling complete. Results saved to: {logdir}")
    print(f"To view results, run: tensorboard --logdir={logdir}")


if __name__ == "__main__":
    main()
