"""Compare validation metrics between our dataloader changes and upstream defaults.

Loads a single checkpoint and runs validation twice:
  1. "Ours" — shuffle=False, drop_last=False, persistent_workers, prefetch_factor
  2. "Upstream" — shuffle=True (val), drop_last=True (matches upstream BaseWellDataModule)

Usage (via srun):
    srun ... python scripts/validate_comparison.py \
        --config examples/well/gray_scott_reaction_diffusion/cfg_hyena_gaussian_mask.py \
        --ckpt runs/.../checkpoints/last.ckpt \
        [overrides...]
"""

import argparse

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from experiments.utils.cli import (
    apply_config_overrides,
    load_config_from_file,
    verify_no_interpolator_overwrites,
)
from nvsubquadratic.lazy_config import instantiate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("overrides", nargs="*")
    args, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith("--"):
            args.overrides.append(arg[2:])
        else:
            args.overrides.append(arg)
    return args


def make_loader(dataset, batch_size, num_workers, *, shuffle, drop_last, persistent_workers, prefetch_factor):
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers and use_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
    )


def run_validation(
    trainer, model, datamodule, *, variant_name, val_shuffle, val_drop_last, persistent_workers, prefetch_factor
):
    """Monkey-patch dataloaders and run validation."""
    bs = datamodule.batch_size
    nw = datamodule.num_workers
    dm = datamodule._well_datamodule

    common = {"persistent_workers": persistent_workers, "prefetch_factor": prefetch_factor}

    datamodule.val_dataloader = lambda: make_loader(
        dm.val_dataset, bs, nw, shuffle=val_shuffle, drop_last=val_drop_last, **common
    )

    datamodule.test_dataloader = lambda: make_loader(
        dm.test_dataset, bs, nw, shuffle=False, drop_last=val_drop_last, **common
    )

    print(f"\n{'=' * 60}")
    print(f"  Variant: {variant_name}")
    print(f"  val  shuffle={val_shuffle}  drop_last={val_drop_last}")
    print(f"  test shuffle=False  drop_last={val_drop_last}")
    print(f"  persistent_workers={persistent_workers}  prefetch_factor={prefetch_factor}")
    print(f"{'=' * 60}\n")

    val_results = trainer.validate(model, datamodule=datamodule)
    test_results = trainer.test(model, datamodule=datamodule)
    return val_results, test_results


def main():
    args = parse_args()
    config = load_config_from_file(args.config)
    verify_no_interpolator_overwrites(config, args.overrides)
    config = apply_config_overrides(config, args.overrides)

    pl.seed_everything(config.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    if getattr(config, "mp_sharing_strategy", None):
        torch.multiprocessing.set_sharing_strategy(config.mp_sharing_strategy)

    datamodule = instantiate(config.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    network = instantiate(config.net)

    wrapper_kwargs = {"network": network, "cfg": config}
    if hasattr(datamodule, "metadata"):
        wrapper_kwargs["metadata"] = datamodule.metadata
    if hasattr(datamodule, "normalization"):
        wrapper_kwargs["normalization"] = datamodule.normalization
    model = instantiate(config.lightning_wrapper_class, **wrapper_kwargs)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"]
    # Strip _orig_mod. prefix added by torch.compile checkpoints
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("._orig_mod", "")
        cleaned[new_key] = v
    model.load_state_dict(cleaned)
    print(f"[validate] Loaded checkpoint: {args.ckpt}")

    from pytorch_lightning.loggers import WandbLogger

    logger = WandbLogger(
        project="nvsubquadratic-validate",
        save_dir="runs/validate_comparison",
        offline=True,
        name="val-compare",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=config.train.precision,
        logger=logger,
        enable_checkpointing=False,
    )

    # --- Variant 1: OURS (shuffle=False, drop_last=False) ---
    ours_val, ours_test = run_validation(
        trainer,
        model,
        datamodule,
        variant_name="OURS",
        val_shuffle=False,
        val_drop_last=False,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # --- Variant 2: UPSTREAM (shuffle=True for val, drop_last=True) ---
    # Re-seed for fairness — upstream shuffles, so seed controls order
    pl.seed_everything(config.seed, workers=True)

    upstream_val, upstream_test = run_validation(
        trainer,
        model,
        datamodule,
        variant_name="UPSTREAM",
        val_shuffle=True,
        val_drop_last=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)

    for split_name, ours_res, upstream_res in [
        ("VAL", ours_val, upstream_val),
        ("TEST", ours_test, upstream_test),
    ]:
        print(f"\n  [{split_name}]")
        if ours_res and upstream_res:
            ours_d = ours_res[0] if isinstance(ours_res, list) else ours_res
            upstream_d = upstream_res[0] if isinstance(upstream_res, list) else upstream_res
            all_keys = sorted(set(ours_d.keys()) | set(upstream_d.keys()))
            for k in all_keys:
                ov = ours_d.get(k, "N/A")
                uv = upstream_d.get(k, "N/A")
                match = "==" if ov == uv else "!="
                print(f"    {k:40s}  ours={ov}  upstream={uv}  {match}")

    print("\n" + "=" * 70)
    print("  Done.")


if __name__ == "__main__":
    main()
