# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Entry point to run experiments.

Usage:
    # MNIST classification
    PYTHONPATH=. python nvsubquadratic/examples/run.py --config examples/mnist_classification/experiments/mnist_classification_ccnn_4_160_hyena_rope_qknorm.py
"""

import argparse
import os
from pathlib import Path

# Force-initialize PIL plugins in the main process before DataLoader workers
# are forked.  Prevents crashes from lazy initialization in child processes.
import PIL.Image


PIL.Image.init()

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import torch.multiprocessing  # noqa: E402
import wandb  # noqa: E402
from pytorch_lightning.loggers import WandbLogger  # noqa: E402
from rich import print as rprint  # noqa: E402
from rich.tree import Tree  # noqa: E402

from experiments.trainer import construct_trainer  # noqa: E402
from experiments.utils.checkpointing import (  # noqa: E402
    download_checkpoint,
    load_checkpoint_state_dict,
    load_state_dict_partially,
    preview_state_dict_compatibility,
)
from experiments.utils.cli import (  # noqa: E402
    add_to_tree,
    apply_config_overrides,
    config_to_dict,
    get_deterministic_run_name,
    load_config_from_file,
    verify_no_interpolator_overwrites,
)
from nvsubquadratic.lazy_config import instantiate  # noqa: E402


torch._dynamo.config.cache_size_limit = 32

try:
    import warp as wp
    wp.init()
except Exception:
    pass


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

    # Add a catch-all for arbitrary config overrides
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides, e.g., dataset.batch_size=64",
    )

    args, unknown = parser.parse_known_args()

    # Support --key=value format (common in wandb sweeps)
    for arg in unknown:
        if arg.startswith("--"):
            # Strip the leading --
            # This turns --dataset.batch_size=32 into dataset.batch_size=32
            args.overrides.append(arg[2:])
        else:
            # If it doesn't start with -- but wasn't caught by positional, keep it.
            # (Though effectively 'overrides' nargs='*' should catch non-dashed args)
            args.overrides.append(arg)

    return args


def main() -> None:
    """Main function to run the experiment.

    This function orchestrates the entire experiment lifecycle, including:
    1.  Parsing command-line arguments.
    2.  Loading and overriding configuration from files and command line.
    3.  Setting up the environment, including seeding for reproducibility and configuring torch settings.
    4.  Instantiating the data module, network model, and the Lightning wrapper.
    5.  Setting up the Weights & Biases logger, with support for auto-resuming runs.
    6.  Handling checkpoint loading for starting from pretrained weights or fine-tuning.
    7.  Constructing the PyTorch Lightning trainer with appropriate callbacks.
    8.  Executing the training, validation, and testing phases of the experiment.
    """
    # Initialize PIL plugins in the main process before any dataloader workers are created.
    PIL.Image.init()

    # Parse command line arguments
    args = parse_args()

    # Load configuration from file
    config = load_config_from_file(args.config)

    # Validate that overrides do not target interpolated fields, then apply
    verify_no_interpolator_overwrites(config, args.overrides)
    config = apply_config_overrides(config, args.overrides)

    num_nodes = config.num_nodes
    experiment_dir = config.experiment_dir

    # Set seed
    pl.seed_everything(config.seed, workers=True)

    # Set deterministic mode
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cudnn.benchmark = not config.deterministic

    # Set float32 matmul precision
    torch.set_float32_matmul_precision("high")

    # Isolate Triton cache per DDP rank to prevent file-lock races during
    # concurrent compilation (all ranks compile the same kernels in parallel).
    local_rank = os.environ.get("LOCAL_RANK", "0")
    base_triton_dir = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    os.environ["TRITON_CACHE_DIR"] = os.path.join(base_triton_dir, f"rank_{local_rank}")

    # Override multiprocessing sharing strategy if requested (e.g. "file_system"
    # to avoid /dev/shm exhaustion with many workers on a shared node).
    if getattr(config, "mp_sharing_strategy", None):
        torch.multiprocessing.set_sharing_strategy(config.mp_sharing_strategy)
        print(f"[run] multiprocessing sharing strategy → {config.mp_sharing_strategy}", flush=True)

    # Construct data_module, prepare and setup
    datamodule = instantiate(config.dataset)
    print("[run] prepare_data …", flush=True)
    datamodule.prepare_data()
    print("[run] setup …", flush=True)
    datamodule.setup()
    print("[run] datamodule ready.", flush=True)

    # Construct model
    network = instantiate(config.net)

    # Enable compile-compatible FFT path if requested (needed for models with FFT conv, e.g. Hyena + FiLM)
    if getattr(config, "compile_compatible_fftconv", False):
        import nvsubquadratic.ops.fftconv as _fftconv

        _fftconv.COMPILE_COMPATIBLE = True
        print("[compile] Using compile-compatible FFT convolution (real-valued complex multiply)")

    # Compile the model if specified
    if config.compile:
        mode = getattr(config, "compile_mode", None)
        mode_str = f" (mode={mode})" if mode else ""
        print(f"Compiling model with torch.compile{mode_str}...")
        compile_kwargs = {"mode": mode} if mode else {}
        network = torch.compile(network, **compile_kwargs)

    # Wrap network in a pl.LightningModule
    wrapper_kwargs: dict = {"network": network, "cfg": config}
    if hasattr(datamodule, "metadata"):
        wrapper_kwargs["metadata"] = datamodule.metadata
    if hasattr(datamodule, "normalization"):
        wrapper_kwargs["normalization"] = datamodule.normalization
    model = instantiate(config.lightning_wrapper_class, **wrapper_kwargs)

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

    experiment_dir = Path(experiment_dir) if experiment_dir is not None else Path("runs") / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Determine if we should attach to an existing run by name (autoresume)
    autoresume_ckpt_path = None
    attach_run_id = None

    # If autoresume is enabled, search W&B for existing run and download checkpoint
    if config.autoresume.enabled and not offline:
        api = wandb.Api()
        runs = api.runs(
            path=f"{config.wandb.entity}/{config.wandb.project}",
            filters={"display_name": run_name},
        )
        if len(runs) > 1:
            raise RuntimeError(
                f"[autoresume] Multiple runs found with name '{run_name}'. Refusing to resume. Count={len(runs)}"
            )
        if len(runs) == 1:
            # Exactly one run found - download checkpoint and attach to it
            target_run = runs[0]
            attach_run_id = target_run.id
            target_run_path = f"{target_run.entity}/{target_run.project}/{target_run.id}"
            ckpt_alias = config.autoresume.alias
            print(f"[autoresume] Found existing run '{run_name}' ({target_run_path}), downloading checkpoint...")
            try:
                autoresume_ckpt_path = download_checkpoint(run_path=target_run_path, alias=ckpt_alias)
                print(f"[autoresume] Checkpoint downloaded to: {autoresume_ckpt_path}")
            except Exception as e:
                print(f"[autoresume] Failed to download checkpoint: {e}")
                print("[autoresume] Will check for local checkpoint instead.")
        else:
            print(f"[autoresume] No existing run found with name '{run_name}', starting fresh.")

    # If autoresume enabled but no W&B checkpoint found, check for local checkpoint
    if config.autoresume.enabled and autoresume_ckpt_path is None:
        ckpt_dir = experiment_dir / "checkpoints"
        if ckpt_dir.exists():
            last_path = ckpt_dir / "last.ckpt"
            if last_path.exists():
                autoresume_ckpt_path = last_path
                print(f"[autoresume] Found local checkpoint: {autoresume_ckpt_path}")
            else:
                print(f"[autoresume] No last.ckpt found in {ckpt_dir}, starting from scratch.")

    # Generate or reuse run ID
    run_id_file = experiment_dir / "run.id"
    if config.wandb.run_id is not None:
        # Explicit run_id provided via config override
        attach_run_id = config.wandb.run_id
        run_id_file.write_text(attach_run_id)
    elif attach_run_id is not None:
        # Use the run ID from W&B and save it locally
        run_id_file.write_text(attach_run_id)
    elif run_id_file.exists():
        # Resume existing local run
        attach_run_id = run_id_file.read_text().strip()
    else:
        # Fresh run - generate new run ID
        attach_run_id = wandb.util.generate_id()
        run_id_file.write_text(attach_run_id)

    # Serialize config once for both WandB and tree printing
    config_dict = config_to_dict(config)

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
            config=config_dict,
            log_model=log_model,
            offline=offline,
            save_code=True,
            group=config.wandb.job_group,
        )

    # Recreate the command that instantiated this run.
    if isinstance(wandb_logger.experiment.settings, wandb.Settings):
        command = f"python run.py --config {args.config}"
        if args.overrides:
            command += " " + " ".join(args.overrides)
        # Log the command.
        wandb_logger.experiment.config.update({"command": command}, allow_val_change=True)

    # Print the config tree
    tree = Tree("Configuration")
    add_to_tree(tree, config_dict)
    rprint(tree)

    # If we are not autoresuming and we want to start training from pretrained weights,
    # download and load the checkpoint (weights only, no optimizer/scheduler state).
    # Note: start_from_checkpoint is skipped when autoresume is enabled (they are mutually exclusive).
    if autoresume_ckpt_path is None and not config.autoresume.enabled:
        if config.start_from_checkpoint.load:
            # Validate run_path is set
            if not config.start_from_checkpoint.run_path:
                raise ValueError(
                    "[start] start_from_checkpoint.run_path must be set when start_from_checkpoint.load=True. "
                    "Example: start_from_checkpoint.run_path=entity/project/run_id"
                )
            print(
                f"[start] Loading pretrained weights: run_path={config.start_from_checkpoint.run_path}, "
                f"alias={config.start_from_checkpoint.alias}, strict={config.start_from_checkpoint.strict}, "
                f"partial_load={config.start_from_checkpoint.partial_load}"
            )
            # Download checkpoint from W&B run and load state dict.
            start_ckpt_path = download_checkpoint(
                run_path=config.start_from_checkpoint.run_path,
                alias=config.start_from_checkpoint.alias,
            )
            print(f"[start] Checkpoint downloaded to: {start_ckpt_path}")
            state_dict = load_checkpoint_state_dict(start_ckpt_path)

            # Process callbacks if configured
            if hasattr(config.start_from_checkpoint, "callbacks") and config.start_from_checkpoint.callbacks:
                for cb_cfg in config.start_from_checkpoint.callbacks:
                    callback = instantiate(cb_cfg)
                    result = callback(
                        state_dict=state_dict,
                        model=model,
                        config=config,
                        datamodule=datamodule,
                        checkpoint_path=start_ckpt_path,
                    )
                    if result is not None:
                        state_dict = result

            # Preview detailed compatibility before loading
            try:
                preview_state_dict_compatibility(model, state_dict)
            except Exception as e:
                print(f"[start] Compatibility preview failed: {e}")

            print("[start] Loading weights into model...")
            if config.start_from_checkpoint.strict:
                res = model.load_state_dict(state_dict, strict=True)
                if hasattr(res, "missing_keys") and hasattr(res, "unexpected_keys"):
                    print(f"[start/strict] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
                    for k in res.missing_keys:
                        print(f"  [missing] {k}")
                    for k in res.unexpected_keys:
                        print(f"  [unexpected] {k}")
            else:
                if config.start_from_checkpoint.partial_load:
                    # Perform tolerant partial parameter loading (overlapping slices)
                    load_state_dict_partially(model, state_dict)
                else:
                    # Non-strict load: only exact-shape matches are loaded; others are ignored
                    res = model.load_state_dict(state_dict, strict=False)
                    if hasattr(res, "missing_keys") and hasattr(res, "unexpected_keys"):
                        print(
                            f"[start/non-strict] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
                        )
                        for k in res.missing_keys:
                            print(f"  [missing] {k}")
                        for k in res.unexpected_keys:
                            print(f"  [unexpected] {k}")
            print("[start] Weight loading completed.")

    # Create trainer
    trainer, checkpoint_callback = construct_trainer(config, wandb_logger, run_name, experiment_dir, num_nodes)

    # Validate that the checkpoint has been correctly loaded before training (for start_from_checkpoint)
    if autoresume_ckpt_path is None and not config.autoresume.enabled and config.start_from_checkpoint.load:
        print("[start] Running validation to verify loaded weights...")
        trainer.validate(model, datamodule=datamodule)
        print("[start] Validation completed.")

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

    # Validate and test after training before finishing
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
