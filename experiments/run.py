# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Entry point to run experiments.

Usage:
    # MNIST classification
    PYTHONPATH=. python nvsubquadratic/examples/run.py --config examples/mnist_classification/experiments/mnist_classification_ccnn_4_160_hyena_rope_qknorm.py
"""

import argparse
import os
import dataclasses

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.loggers import WandbLogger
from rich import print as rprint
from rich.tree import Tree

import wandb
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
from experiments.trainer import construct_trainer


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
    
    # Compile the model
    network = torch.compile(network)
    
    # Wrap network in a pl.LightningModule
    model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)
    if config.do_torch_compile:
        model = torch.compile(model, mode=config.torch_compile_mode)

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

    # Determine if we should attach to an existing run by name (autoresume)
    autoresume_ckpt_path = None
    attach_run_id = None
    if config.autoresume.enabled and not offline:
        api = wandb.Api()
        runs = api.runs(path=f"{config.wandb.entity}/{config.wandb.project}", filters={"display_name": run_name})
        if len(runs) > 1:
            raise RuntimeError(
                f"[autoresume] Multiple runs found with name '{run_name}'. Refusing to resume. Count={len(runs)}"
            )
        if len(runs) == 0:
            raise RuntimeError(
                f"[autoresume] No run found with name '{run_name}' in {config.wandb.entity}/{config.wandb.project}."
            )
        # Exactly one run found
        target_run = runs[0]
        attach_run_id = target_run.id
        target_run_path = f"{target_run.entity}/{target_run.project}/{target_run.id}"
        ckpt_alias = config.autoresume.alias
        autoresume_ckpt_path = download_checkpoint(run_path=target_run_path, alias=ckpt_alias)
        print(f"[autoresume] Found existing run '{target_run_path}', downloaded ckpt: {autoresume_ckpt_path}")

    # Create logger, attaching to existing run if applicable
    if attach_run_id is not None and not offline:
        wandb_logger = WandbLogger(
            project=config.wandb.project,
            entity=config.wandb.entity,
            id=attach_run_id,
            resume="allow",
            name=None,
            # config=dataclasses.asdict(config),
            log_model=log_model,
            offline=offline,
            save_code=True,
            group=config.wandb.job_group,
        )
    else:
        # Start a fresh run otherwise
        wandb_logger = WandbLogger(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=run_name,
            config=dataclasses.asdict(config),  # Convert dataclass config to dict
            log_model=log_model,  # used to save models to wandb during training
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
        wandb_logger.experiment.config.update({"command": command})

    # Print the config files prior to training
    config_dict = config_to_dict_for_rich(config)
    tree = Tree("Configuration")
    add_to_tree(tree, config_dict)
    rprint(tree)

    # If we are not autoresuming and we are starting training from a given checkpoint, check whether we are starting training
    # from a predefined checkpoint.
    if autoresume_ckpt_path is None:
        if config.resume_from_checkpoint.load:
            print(
                f"[resume] Starting checkpoint resume: run_path={config.resume_from_checkpoint.run_path}, "
                f"alias={config.resume_from_checkpoint.alias}, strict={config.resume_from_checkpoint.strict}, "
                f"partial_load={config.resume_from_checkpoint.partial_load}"
            )
            # Download checkpoint from W&B run and load state dict.
            resume_ckpt_path = download_checkpoint(
                run_path=config.resume_from_checkpoint.run_path,
                # output_dir=config.resume_from_checkpoint.output_dir,
                alias=config.resume_from_checkpoint.alias,
            )
            print(f"[resume] Checkpoint downloaded to: {resume_ckpt_path}")
            state_dict = load_checkpoint_state_dict(resume_ckpt_path)
            # Preview detailed compatibility before loading
            try:
                preview_state_dict_compatibility(model, state_dict)
            except Exception as e:
                print(f"[resume] Compatibility preview failed: {e}")

            print("[resume] Loading weights into model...")
            if config.resume_from_checkpoint.strict:
                res = model.load_state_dict(state_dict, strict=True)
                if hasattr(res, "missing_keys") and hasattr(res, "unexpected_keys"):
                    print(f"[resume/strict] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
                    for k in res.missing_keys:
                        print(f"  [missing] {k}")
                    for k in res.unexpected_keys:
                        print(f"  [unexpected] {k}")
            else:
                if config.resume_from_checkpoint.partial_load:
                    # Perform tolerant partial parameter loading (overlapping slices)
                    load_state_dict_partially(model, state_dict)
                else:
                    # Non-strict load: only exact-shape matches are loaded; others are ignored
                    res = model.load_state_dict(state_dict, strict=False)
                    if hasattr(res, "missing_keys") and hasattr(res, "unexpected_keys"):
                        print(
                            f"[resume/non-strict] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
                        )
                        for k in res.missing_keys:
                            print(f"  [missing] {k}")
                        for k in res.unexpected_keys:
                            print(f"  [unexpected] {k}")
            print("[resume] Weight loading completed.")

    # Create trainer
    trainer, checkpoint_callback = construct_trainer(config, wandb_logger, run_name)

    # Validate that the checkpoint has been correctly loaded before training (for no autoresume)
    if autoresume_ckpt_path is None and config.resume_from_checkpoint.load:
        print("[resume] Running validation to verify loaded checkpoint...")
        trainer.validate(model, datamodule=datamodule)
        print("[resume] Validation after resume completed.")

    # # register hooks
    # if config.hooks_enabled:
    #     model.configure_callbacks = partial(register_hooks, config, model)

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
