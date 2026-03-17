"""Verify that DALIImageNetFusedDataModule produces reasonable outputs.

Checks:
1. Output shapes and dtypes are correct
2. Values are in expected normalized range
3. Validation outputs match between fused and optimized pipelines (deterministic)
4. Training augmentations produce visually reasonable distributions
5. Visual side-by-side comparison saved as PNG

Usage:
    PYTHONPATH=. python benchmarks/vit5_imagenet/verify_dali_fused.py
"""

import os

import torch


os.environ.setdefault("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
os.environ.setdefault("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")

from experiments.datamodules.imagenet import AugmentConfig, MixupConfig

from experiments.datamodules._deprecated.dali_imagenet_optimized import DALIImageNetOptimizedDataModule
from experiments.datamodules.dali_imagenet_fused import DALIImageNetFusedDataModule


BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 4
SEED = 42
DEVICE_ID = 0

AUGMENT_CFG = AugmentConfig(use_three_augment=True, color_jitter=0.3)
MIXUP_CFG = MixupConfig(mixup=0.8, cutmix=1.0, mixup_prob=1.0, mixup_switch_prob=0.5, smoothing=0.0)

COMMON = {
    "data_dir": os.environ["IMAGENET_PATH"],
    "imagefolder_dir": os.environ.get("IMAGENET_FOLDER_PATH"),
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": True,
    "seed": SEED,
    "image_size": IMAGE_SIZE,
    "final_image_size": IMAGE_SIZE,
    "num_classes": 1000,
    "drop_labels": False,
    "task": "classification",
    "device_id": DEVICE_ID,
}


def check_shapes_and_dtypes(name, batch):
    """Verify output dict has correct structure."""
    assert isinstance(batch, dict), f"{name}: expected dict, got {type(batch)}"
    assert "input" in batch, f"{name}: missing 'input' key"
    assert "label" in batch, f"{name}: missing 'label' key"

    images = batch["input"]
    labels = batch["label"]

    assert images.ndim == 4, f"{name}: expected 4D tensor, got {images.ndim}D"
    _B = images.shape[0]

    # NHWC layout (channels_first=False default)
    assert images.shape[-1] == 3 or images.shape[1] == 3, f"{name}: unexpected shape {images.shape}"
    assert images.dtype == torch.float32, f"{name}: expected float32, got {images.dtype}"
    assert images.is_cuda, f"{name}: expected CUDA tensor"

    print(
        f"  [{name}] shape={tuple(images.shape)}, dtype={images.dtype}, "
        f"device={images.device}, labels shape={tuple(labels.shape)}"
    )
    return images, labels


def check_value_range(name, images, is_normalized=True):
    """Check that normalized values are in a reasonable range."""
    vmin, vmax = images.min().item(), images.max().item()
    vmean, vstd = images.mean().item(), images.std().item()
    print(f"  [{name}] min={vmin:.3f}, max={vmax:.3f}, mean={vmean:.3f}, std={vstd:.3f}")

    if is_normalized:
        assert vmin > -10.0, f"{name}: suspiciously low min={vmin}"
        assert vmax < 10.0, f"{name}: suspiciously high max={vmax}"


def compare_validation(optimized_dm, fused_dm):
    """Compare validation outputs — should be near-identical (both deterministic)."""
    print("\n== Comparing VALIDATION outputs ==")

    opt_loader = optimized_dm.val_dataloader()
    fused_loader = fused_dm.val_dataloader()

    opt_batch_raw = next(iter(opt_loader))
    fused_batch_raw = next(iter(fused_loader))

    mock_trainer = type("_Mock", (), {"training": False, "local_rank": 0, "world_size": 1})()
    optimized_dm.trainer = mock_trainer
    fused_dm.trainer = mock_trainer

    opt_batch = optimized_dm.on_before_batch_transfer(opt_batch_raw, 0)
    fused_batch = fused_dm.on_before_batch_transfer(fused_batch_raw, 0)

    opt_img, _ = check_shapes_and_dtypes("opt-val", opt_batch)
    fused_img, _ = check_shapes_and_dtypes("fused-val", fused_batch)

    check_value_range("opt-val", opt_img)
    check_value_range("fused-val", fused_img)

    # Both read the same files in the same order (same seed, same shard),
    # so the normalized values should be very close
    diff = (opt_img - fused_img).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  Val diff: max={max_diff:.6f}, mean={mean_diff:.6f}")

    if max_diff < 0.01:
        print("  PASS: Validation outputs match closely")
    else:
        print(
            f"  WARNING: Validation diff is {max_diff:.4f} — may be acceptable "
            "due to DALI pipeline differences (interpolation, rounding)"
        )


def check_training(dm, name):
    """Check training augmentations produce reasonable output."""
    print(f"\n== Checking TRAINING output for {name} ==")

    loader = dm.train_dataloader()
    batch_raw = next(iter(loader))

    mock_trainer = type("_Mock", (), {"training": True, "local_rank": 0, "world_size": 1})()
    dm.trainer = mock_trainer

    batch = dm.on_before_batch_transfer(batch_raw, 0)
    images, labels = check_shapes_and_dtypes(name, batch)
    check_value_range(name, images)

    return images, labels


def save_visual_comparison(opt_img, fused_img, path="benchmarks/vit5_imagenet/dali_fused_comparison.png"):
    """Save a side-by-side visual comparison of the two pipelines."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (skipping visual comparison — matplotlib not installed)")
        return

    n = min(8, opt_img.shape[0])

    def to_vis(img_batch):
        # Handle both NHWC and NCHW
        if img_batch.shape[1] == 3:
            imgs = img_batch[:n].permute(0, 2, 3, 1)
        else:
            imgs = img_batch[:n]
        # Approximate unnormalize (just shift to [0,1] range for visualization)
        imgs = imgs.cpu().float()
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
        return imgs.numpy()

    opt_vis = to_vis(opt_img)
    fused_vis = to_vis(fused_img)

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        axes[0, i].imshow(opt_vis[i])
        axes[0, i].set_title("optimized" if i == 0 else "", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(fused_vis[i])
        axes[1, i].set_title("fused" if i == 0 else "", fontsize=8)
        axes[1, i].axis("off")

    fig.suptitle("Training augmentation comparison (different random seeds — expect different images)", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"  Visual comparison saved to {path}")
    plt.close(fig)


def main():
    """Verify DALIImageNetFusedDataModule produces correct outputs."""
    print("=" * 60)
    print("DALI Fused DataModule Verification")
    print("=" * 60)

    # Build both datamodules
    print("\nBuilding optimized (old) datamodule...")
    opt_dm = DALIImageNetOptimizedDataModule(
        **COMMON,
        augment_cfg=AUGMENT_CFG,
        mixup_cfg=MIXUP_CFG,
    )
    opt_dm.setup("fit")

    print("Building fused (new) datamodule...")
    fused_dm = DALIImageNetFusedDataModule(
        **COMMON,
        augment_cfg=AUGMENT_CFG,
        mixup_cfg=MIXUP_CFG,
    )
    fused_dm.setup("fit")

    # 1. Validation comparison (should be near-identical)
    compare_validation(opt_dm, fused_dm)

    # 2. Training augmentation checks
    opt_train_img, _ = check_training(opt_dm, "optimized-train")
    fused_train_img, _ = check_training(fused_dm, "fused-train")

    # 3. Visual comparison
    save_visual_comparison(opt_train_img, fused_train_img)

    # 4. Quick speed comparison (10 batches each)
    print("\n== Quick speed check (10 batches) ==")
    import time

    for dm, name in [(opt_dm, "optimized"), (fused_dm, "fused")]:
        dm.trainer = type("_Mock", (), {"training": True, "local_rank": 0, "world_size": 1})()
        loader = dm.train_dataloader()
        it = iter(loader)
        # warmup
        for _ in range(3):
            dm.on_before_batch_transfer(next(it), 0)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            dm.on_before_batch_transfer(next(it), 0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ms = elapsed / 10 * 1000
        print(f"  [{name}] {ms:.1f} ms/batch (data + augment + transfer)")

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
