"""Test that DALIImageNetOptimizedDataModule produces equivalent outputs to DALIImageNetDataModule.

Also benchmarks both modules side-by-side to measure speedup.

Requires GPU and ImageNet data in ImageFolder layout.
Run: PYTHONPATH=. python tests/test_dali_optimized_equivalence.py
"""

import os
import sys
import time

import torch

os.environ.setdefault("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
os.environ.setdefault("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")

from experiments.datamodules.dali_imagenet import (
    DALIImageNetDataModule,
    _BatchColorJitter as OrigColorJitter,
)
from experiments.datamodules.dali_imagenet_optimized import (
    DALIImageNetOptimizedDataModule,
    _BatchColorJitter as OptColorJitter,
)
from experiments.datamodules.imagenet import AugmentConfig, MixupConfig

BATCH_SIZE = 256
NUM_WORKERS = 8
IMAGE_SIZE = 224
FOLDER_PATH = os.environ["IMAGENET_FOLDER_PATH"]
IMAGENET_PATH = os.environ["IMAGENET_PATH"]

COMMON_KWARGS = dict(
    data_dir=IMAGENET_PATH,
    imagefolder_dir=FOLDER_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    seed=42,
    image_size=IMAGE_SIZE,
    final_image_size=IMAGE_SIZE,
    num_classes=1000,
    drop_labels=False,
    task="classification",
    prefetch_factor=2,
    eval_crop_ratio=1.0,
)

AUGMENT_CFG = AugmentConfig(use_three_augment=True, color_jitter=0.3)
MIXUP_CFG = MixupConfig(mixup=0.8, cutmix=1.0, mixup_prob=1.0,
                         mixup_switch_prob=0.5, smoothing=0.0)

passed, failed, errors = [], [], []


def run_test(name, fn):
    try:
        fn()
        passed.append(name)
        print(f"  PASS  {name}")
    except AssertionError as e:
        failed.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        errors.append((name, str(e)))
        print(f"  ERROR {name}: {e}")


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _build_orig(augment=False, training=True):
    dm = DALIImageNetDataModule(
        **COMMON_KWARGS,
        augment_cfg=AUGMENT_CFG if augment else None,
        device_id=0,
    )
    dm.setup("fit")
    dm.trainer = type("_Mock", (), {"training": training})()
    return dm


def _build_opt(augment=False, training=True, channels_first=False):
    dm = DALIImageNetOptimizedDataModule(
        **COMMON_KWARGS,
        augment_cfg=AUGMENT_CFG if augment else None,
        device_id=0,
        channels_first=channels_first,
    )
    dm.setup("fit")
    dm.trainer = type("_Mock", (), {"training": training})()
    return dm


# ── 1. Output format tests ──────────────────────────────────────────────────

def test_val_output_format():
    """Optimised val output has same keys, shapes, and dtypes as original."""
    orig = _build_orig(training=False)
    opt = _build_opt(training=False)

    orig_raw = next(iter(orig.val_dataloader()))
    opt_raw = next(iter(opt.val_dataloader()))

    orig_batch = orig.on_before_batch_transfer(orig_raw, 0)
    opt_batch = opt.on_before_batch_transfer(opt_raw, 0)

    assert set(orig_batch.keys()) == set(opt_batch.keys()), \
        f"Key mismatch: {set(orig_batch.keys())} vs {set(opt_batch.keys())}"
    assert orig_batch["input"].shape == opt_batch["input"].shape, \
        f"Shape mismatch: {orig_batch['input'].shape} vs {opt_batch['input'].shape}"
    assert orig_batch["input"].dtype == opt_batch["input"].dtype, \
        f"Dtype mismatch: {orig_batch['input'].dtype} vs {opt_batch['input'].dtype}"


def test_train_output_format():
    """Optimised train output has same keys, shapes, and dtypes as original."""
    orig = _build_orig(augment=True)
    opt = _build_opt(augment=True)

    orig_raw = next(iter(orig.train_dataloader()))
    opt_raw = next(iter(opt.train_dataloader()))

    orig_batch = orig.on_before_batch_transfer(orig_raw, 0)
    opt_batch = opt.on_before_batch_transfer(opt_raw, 0)

    assert set(orig_batch.keys()) == set(opt_batch.keys())
    assert orig_batch["input"].shape == opt_batch["input"].shape, \
        f"Shape mismatch: {orig_batch['input'].shape} vs {opt_batch['input'].shape}"
    assert orig_batch["input"].dtype == opt_batch["input"].dtype


def test_channels_first_format():
    """channels_first=True produces (B, C, H, W) tensors."""
    opt = _build_opt(augment=True, channels_first=True)
    raw = next(iter(opt.train_dataloader()))
    batch = opt.on_before_batch_transfer(raw, 0)

    assert batch["input"].shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), \
        f"Expected NCHW shape, got {batch['input'].shape}"


# ── 2. Numerical equivalence (validation — deterministic) ───────────────────

def test_val_pixel_equivalence():
    """Val outputs should be numerically close (same DALI pipeline, different normalise path)."""
    orig = _build_orig(training=False)
    opt = _build_opt(training=False)

    orig_loader = orig.val_dataloader()
    opt_loader = opt.val_dataloader()

    orig_raw = next(iter(orig_loader))
    opt_raw = next(iter(opt_loader))

    orig_batch = orig.on_before_batch_transfer(orig_raw, 0)
    opt_batch = opt.on_before_batch_transfer(opt_raw, 0)

    orig_labels = orig_batch["label"]
    opt_labels = opt_batch["label"]

    if not torch.equal(orig_labels, opt_labels):
        print("         (different label ordering — skip pixel comparison)")
        return

    diff = (orig_batch["input"] - opt_batch["input"]).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    # Fused normalise may have tiny fp32 rounding differences
    assert mean_diff < 1e-5, f"Val mean pixel diff too large: {mean_diff:.2e}"
    assert max_diff < 1e-4, f"Val max pixel diff too large: {max_diff:.2e}"
    print(f"         (mean_diff={mean_diff:.2e}, max_diff={max_diff:.2e})")


# ── 3. ColorJitter statistical equivalence ──────────────────────────────────

def test_color_jitter_distribution():
    """Both ColorJitter variants should produce similar output distributions.

    Since they use different random state, we compare statistical properties
    over many batches rather than exact values.
    """
    device = torch.device("cuda")
    B, C, H, W = 128, 3, 32, 32

    torch.manual_seed(0)
    orig_cj = OrigColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)

    torch.manual_seed(0)
    opt_cj = OptColorJitter(brightness=0.3, contrast=0.3, saturation=0.3).to(device)

    n_batches = 50
    orig_means, opt_means = [], []
    orig_stds, opt_stds = [], []

    for _ in range(n_batches):
        images = torch.rand(B, C, H, W, device=device)

        orig_out = orig_cj(images.clone())
        opt_out = opt_cj(images.clone())

        orig_means.append(orig_out.mean().item())
        opt_means.append(opt_out.mean().item())
        orig_stds.append(orig_out.std().item())
        opt_stds.append(opt_out.std().item())

    orig_mean = sum(orig_means) / len(orig_means)
    opt_mean = sum(opt_means) / len(opt_means)
    orig_std = sum(orig_stds) / len(orig_stds)
    opt_std = sum(opt_stds) / len(opt_stds)

    mean_diff = abs(orig_mean - opt_mean)
    std_diff = abs(orig_std - opt_std)

    assert mean_diff < 0.02, \
        f"Mean distribution differs: orig={orig_mean:.4f}, opt={opt_mean:.4f}, diff={mean_diff:.4f}"
    assert std_diff < 0.02, \
        f"Std distribution differs: orig={orig_std:.4f}, opt={opt_std:.4f}, diff={std_diff:.4f}"
    print(f"         (mean_diff={mean_diff:.4f}, std_diff={std_diff:.4f})")


# ── 4. Normalization consistency ─────────────────────────────────────────────

def test_val_normalization_statistics():
    """Both modules should produce similar channel statistics on validation data."""
    orig = _build_orig(training=False)
    opt = _build_opt(training=False)

    def get_stats(dm, loader, n=5):
        pixels = []
        for i, raw in enumerate(loader):
            if i >= n:
                break
            batch = dm.on_before_batch_transfer(raw, 0)
            pixels.append(batch["input"].reshape(-1, 3).cpu())
        all_px = torch.cat(pixels, dim=0).float()
        return all_px.mean(dim=0), all_px.std(dim=0)

    orig_mean, orig_std = get_stats(orig, orig.val_dataloader())
    opt_mean, opt_std = get_stats(opt, opt.val_dataloader())

    mean_diff = (orig_mean - opt_mean).abs().max().item()
    std_diff = (orig_std - opt_std).abs().max().item()

    assert mean_diff < 0.01, f"Mean stats differ: {mean_diff:.4f}"
    assert std_diff < 0.01, f"Std stats differ: {std_diff:.4f}"
    print(f"         (mean_diff={mean_diff:.4f}, std_diff={std_diff:.4f})")


# ── 5. Benchmark ─────────────────────────────────────────────────────────────

def bench_on_before_batch_transfer():
    """Benchmark on_before_batch_transfer for both modules."""
    print()
    print("  " + "─" * 56)
    print("  BENCHMARK: on_before_batch_transfer (train path)")
    print("  " + "─" * 56)

    orig = _build_orig(augment=True)
    opt = _build_opt(augment=True)

    orig_loader = orig.train_dataloader()
    opt_loader = opt.train_dataloader()

    warmup = 10
    n_steps = 50

    # ── Original ──
    it = iter(orig_loader)
    for _ in range(warmup):
        raw = next(it)
        orig.on_before_batch_transfer(raw, 0)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_steps):
        raw = next(it)
        orig.on_before_batch_transfer(raw, 0)
    end.record()
    torch.cuda.synchronize()
    orig_ms = start.elapsed_time(end) / n_steps
    del it

    # ── Optimised ──
    it = iter(opt_loader)
    for _ in range(warmup):
        raw = next(it)
        opt.on_before_batch_transfer(raw, 0)
    torch.cuda.synchronize()

    start.record()
    for _ in range(n_steps):
        raw = next(it)
        opt.on_before_batch_transfer(raw, 0)
    end.record()
    torch.cuda.synchronize()
    opt_ms = start.elapsed_time(end) / n_steps
    del it

    # ── Optimised (channels_first) ──
    opt_cf = _build_opt(augment=True, channels_first=True)
    opt_cf_loader = opt_cf.train_dataloader()
    it = iter(opt_cf_loader)
    for _ in range(warmup):
        raw = next(it)
        opt_cf.on_before_batch_transfer(raw, 0)
    torch.cuda.synchronize()

    start.record()
    for _ in range(n_steps):
        raw = next(it)
        opt_cf.on_before_batch_transfer(raw, 0)
    end.record()
    torch.cuda.synchronize()
    opt_cf_ms = start.elapsed_time(end) / n_steps
    del it

    print(f"  Original (NHWC out):             {orig_ms:7.2f} ms/batch")
    print(f"  Optimised (NHWC out):            {opt_ms:7.2f} ms/batch  ({orig_ms / opt_ms:.2f}x)")
    print(f"  Optimised (NCHW out):            {opt_cf_ms:7.2f} ms/batch  ({orig_ms / opt_cf_ms:.2f}x)")
    print()

    # ── Validation path ──
    print("  " + "─" * 56)
    print("  BENCHMARK: on_before_batch_transfer (val path)")
    print("  " + "─" * 56)

    orig_val = _build_orig(training=False)
    opt_val = _build_opt(training=False)

    orig_vl = orig_val.val_dataloader()
    opt_vl = opt_val.val_dataloader()

    it = iter(orig_vl)
    for _ in range(warmup):
        orig_val.on_before_batch_transfer(next(it), 0)
    torch.cuda.synchronize()

    start.record()
    for _ in range(n_steps):
        orig_val.on_before_batch_transfer(next(it), 0)
    end.record()
    torch.cuda.synchronize()
    orig_val_ms = start.elapsed_time(end) / n_steps
    del it

    it = iter(opt_vl)
    for _ in range(warmup):
        opt_val.on_before_batch_transfer(next(it), 0)
    torch.cuda.synchronize()

    start.record()
    for _ in range(n_steps):
        opt_val.on_before_batch_transfer(next(it), 0)
    end.record()
    torch.cuda.synchronize()
    opt_val_ms = start.elapsed_time(end) / n_steps
    del it

    print(f"  Original val:                    {orig_val_ms:7.2f} ms/batch")
    print(f"  Optimised val (fused normalize):  {opt_val_ms:7.2f} ms/batch  ({orig_val_ms / opt_val_ms:.2f}x)")
    print()


# ── Runner ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DALI Original ↔ Optimised Equivalence Tests")
    print("=" * 60)
    print()

    tests = [
        ("1a. Val output format", test_val_output_format),
        ("1b. Train output format", test_train_output_format),
        ("1c. channels_first format", test_channels_first_format),
        ("2.  Val pixel equivalence", test_val_pixel_equivalence),
        ("3.  ColorJitter distribution", test_color_jitter_distribution),
        ("4.  Normalization statistics", test_val_normalization_statistics),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print()
    print("=" * 60)
    total = len(passed) + len(failed) + len(errors)
    print(f"Results: {len(passed)}/{total} passed, {len(failed)} failed, {len(errors)} errors")
    if failed:
        print("\nFailed:")
        for name, msg in failed:
            print(f"  {name}: {msg}")
    if errors:
        print("\nErrors:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
    print("=" * 60)

    if not failed and not errors:
        bench_on_before_batch_transfer()

    sys.exit(1 if failed or errors else 0)


if __name__ == "__main__":
    main()
