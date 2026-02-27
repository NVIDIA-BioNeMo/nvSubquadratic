"""Validate a ViT-5 checkpoint and print metrics for comparison with W&B."""

import sys


sys.path.insert(0, ".")

import pytorch_lightning as pl
import torch

from examples.vit5_imagenet.vit5_small_pretrain import get_config
from nvsubquadratic.lazy_config import instantiate


CKPT_PATH = "runs/DW_examples_vit5_imagenet_vit5_small_pretrain_2026-02-21-05-14-44/checkpoints/last.ckpt"

EXPECTED_VAL_ACC = 0.2385
EXPECTED_VAL_LOSS = 3.958


def main():
    config = get_config()

    print("=" * 60)
    print("Checkpoint validation (single GPU)")
    print("=" * 60)
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Expected val/acc:  {EXPECTED_VAL_ACC:.4f}")
    print(f"Expected val/loss: {EXPECTED_VAL_LOSS:.4f}")
    print()

    datamodule = instantiate(config.dataset)
    datamodule.setup()

    network = instantiate(config.net)
    model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)

    # Load checkpoint manually to inspect key mapping
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    ckpt_keys = set(ckpt["state_dict"].keys())
    model_keys = set(model.state_dict().keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    print(f"Checkpoint keys: {len(ckpt_keys)}")
    print(f"Model keys:      {len(model_keys)}")
    if missing:
        print(f"  Missing from checkpoint ({len(missing)}): {sorted(missing)[:10]}...")
    if unexpected:
        print(f"  Unexpected in checkpoint ({len(unexpected)}): {sorted(unexpected)[:10]}...")
    if not missing and not unexpected:
        print("  All keys match exactly.")
    print()

    # Use single GPU to avoid DistributedSampler issues
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=config.train.precision,
        logger=False,
    )

    print("Running validation on single GPU...")
    results = trainer.validate(model, datamodule=datamodule, ckpt_path=CKPT_PATH)

    if results:
        val_acc = results[0].get("val/acc", None)
        val_loss = results[0].get("val/loss", None)
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"  val/acc:  {val_acc:.4f}  (expected {EXPECTED_VAL_ACC:.4f})")
        print(f"  val/loss: {val_loss:.4f}  (expected {EXPECTED_VAL_LOSS:.4f})")

        acc_match = abs(val_acc - EXPECTED_VAL_ACC) < 0.001
        loss_match = abs(val_loss - EXPECTED_VAL_LOSS) < 0.01
        print()
        if acc_match and loss_match:
            print("PASS: Metrics match W&B — checkpoint is valid, safe to resume.")
        else:
            print("MISMATCH: Metrics differ from W&B. Investigate before resuming.")
            if not acc_match:
                print(f"  acc diff: {abs(val_acc - EXPECTED_VAL_ACC):.6f}")
            if not loss_match:
                print(f"  loss diff: {abs(val_loss - EXPECTED_VAL_LOSS):.6f}")


if __name__ == "__main__":
    main()
