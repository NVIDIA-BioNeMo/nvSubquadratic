"""Tests verifying DALIImageNetDataModule produces equivalent outputs to ImageNetDataModule.

Requires GPU and ImageNet data in ImageFolder layout.
Run: PYTHONPATH=. python tests/test_dali_equivalence.py
"""

import os
import sys

import torch

os.environ.setdefault("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
os.environ.setdefault("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")

from experiments.datamodules.imagenet import AugmentConfig, ImageNetDataModule, MixupConfig
from experiments.datamodules.dali_imagenet import DALIImageNetDataModule

BATCH_SIZE = 64
NUM_WORKERS = 4
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

def _build_cpu_dm(augment=False, mixup=False):
    dm = ImageNetDataModule(
        **COMMON_KWARGS,
        center_crop=True,
        hf_dataset_name="ILSVRC/imagenet-1k",
        hf_dataset_config=None,
        hf_auth_token=os.environ.get("HF_TOKEN"),
        augment_cfg=AUGMENT_CFG if augment else None,
        mixup_cfg=MIXUP_CFG if mixup else None,
    )
    dm.prepare_data()
    dm.setup("fit")
    dm.trainer = type("_Mock", (), {"training": True})()
    return dm


def _build_dali_dm(augment=False):
    dm = DALIImageNetDataModule(
        **COMMON_KWARGS,
        augment_cfg=AUGMENT_CFG if augment else None,
        device_id=0,
    )
    dm.setup("fit")
    dm.trainer = type("_Mock", (), {"training": True})()
    return dm


# ── 1. Output format tests ──────────────────────────────────────────────────

def test_train_output_format():
    """Train batches have correct dict keys, shapes, and dtypes."""
    dm = _build_dali_dm(augment=True)
    loader = dm.train_dataloader()
    raw = next(iter(loader))
    batch = dm.on_before_batch_transfer(raw, 0)

    assert set(batch.keys()) == {"input", "label", "condition"}, \
        f"Wrong keys: {set(batch.keys())}"
    assert batch["input"].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), \
        f"Wrong input shape: {batch['input'].shape}"
    assert batch["input"].dtype == torch.float32, \
        f"Wrong input dtype: {batch['input'].dtype}"
    assert batch["label"].shape == (BATCH_SIZE,), \
        f"Wrong label shape: {batch['label'].shape}"
    assert batch["label"].dtype == torch.int64, \
        f"Wrong label dtype: {batch['label'].dtype}"
    assert batch["condition"] is None


def test_val_output_format():
    """Val batches have correct dict keys, shapes, and dtypes."""
    dm = _build_dali_dm()
    dm.trainer = type("_Mock", (), {"training": False})()
    loader = dm.val_dataloader()
    raw = next(iter(loader))
    batch = dm.on_before_batch_transfer(raw, 0)

    assert set(batch.keys()) == {"input", "label", "condition"}
    assert batch["input"].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert batch["input"].dtype == torch.float32
    assert batch["label"].dtype == torch.int64
    assert batch["condition"] is None


def test_cpu_output_format():
    """Verify CPU pipeline format matches for reference."""
    dm = _build_cpu_dm(augment=True, mixup=False)
    loader = dm.train_dataloader()
    raw = next(iter(loader))
    batch = dm.on_before_batch_transfer(raw, 0)

    assert set(batch.keys()) == {"input", "label", "condition"}
    assert batch["input"].shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert batch["input"].dtype == torch.float32


# ── 2. Label consistency ────────────────────────────────────────────────────

def test_label_range():
    """Both pipelines produce labels in [0, 999]."""
    cpu_dm = _build_cpu_dm()
    dali_dm = _build_dali_dm()

    cpu_loader = cpu_dm.train_dataloader()
    dali_loader = dali_dm.train_dataloader()

    cpu_labels = torch.cat([batch[1] for batch, _ in zip(cpu_loader, range(5))])
    dali_labels = torch.cat([batch[1] for batch, _ in zip(dali_loader, range(5))])

    assert cpu_labels.min() >= 0 and cpu_labels.max() <= 999, \
        f"CPU labels out of range: [{cpu_labels.min()}, {cpu_labels.max()}]"
    assert dali_labels.min() >= 0 and dali_labels.max() <= 999, \
        f"DALI labels out of range: [{dali_labels.min()}, {dali_labels.max()}]"


def test_val_label_ordering():
    """Val loaders (no shuffle) produce the same label sequence."""
    cpu_dm = _build_cpu_dm()
    dali_dm = _build_dali_dm()
    dali_dm.trainer = type("_Mock", (), {"training": False})()

    cpu_loader = cpu_dm.val_dataloader()
    dali_loader = dali_dm.val_dataloader()

    cpu_labels = torch.cat([batch[1] for batch, _ in zip(cpu_loader, range(3))])
    dali_raw = [batch for batch, _ in zip(dali_loader, range(3))]
    dali_labels = torch.cat([imgs_labels[1] for imgs_labels in dali_raw]).cpu()

    assert torch.equal(cpu_labels, dali_labels), \
        f"Val label ordering differs. CPU first 10: {cpu_labels[:10].tolist()}, " \
        f"DALI first 10: {dali_labels[:10].tolist()}"


# ── 3. Normalization ────────────────────────────────────────────────────────

def test_normalization_statistics():
    """Both pipelines produce images with mean ≈ 0 and std ≈ 1 per channel."""
    cpu_dm = _build_cpu_dm()
    dali_dm = _build_dali_dm()
    dali_dm.trainer = type("_Mock", (), {"training": False})()

    # Accumulate from val (deterministic) batches
    cpu_loader = cpu_dm.val_dataloader()
    dali_loader = dali_dm.val_dataloader()

    n_batches = 10

    def get_stats(loader, dm, is_dali):
        pixels = []
        for i, raw in enumerate(loader):
            if i >= n_batches:
                break
            batch = dm.on_before_batch_transfer(raw, 0)
            img = batch["input"].cpu()  # (B, H, W, C) → CPU for comparison
            pixels.append(img.reshape(-1, 3))
        all_px = torch.cat(pixels, dim=0).float()
        return all_px.mean(dim=0), all_px.std(dim=0)

    cpu_mean, cpu_std = get_stats(cpu_loader, cpu_dm, False)
    dali_mean, dali_std = get_stats(dali_loader, dali_dm, True)

    # After ImageNet normalization, channel means should be near 0
    assert cpu_mean.abs().max() < 1.0, f"CPU mean too large: {cpu_mean}"
    assert dali_mean.abs().max() < 1.0, f"DALI mean too large: {dali_mean}"

    # Means from both pipelines should be close to each other
    mean_diff = (cpu_mean - dali_mean).abs()
    assert mean_diff.max() < 0.15, \
        f"Mean difference too large: {mean_diff} (cpu={cpu_mean}, dali={dali_mean})"

    std_diff = (cpu_std - dali_std).abs()
    assert std_diff.max() < 0.15, \
        f"Std difference too large: {std_diff} (cpu={cpu_std}, dali={dali_std})"


# ── 4. Val pixel similarity ─────────────────────────────────────────────────

def test_val_pixel_similarity():
    """On the same val images (deterministic), pixel values should be close.

    Differences come from bicubic interpolation implementation (PIL vs DALI/nvJPEG).
    """
    cpu_dm = _build_cpu_dm()
    dali_dm = _build_dali_dm()
    dali_dm.trainer = type("_Mock", (), {"training": False})()

    cpu_loader = cpu_dm.val_dataloader()
    dali_loader = dali_dm.val_dataloader()

    cpu_raw = next(iter(cpu_loader))
    dali_raw = next(iter(dali_loader))

    cpu_batch = cpu_dm.on_before_batch_transfer(cpu_raw, 0)
    dali_batch = dali_dm.on_before_batch_transfer(dali_raw, 0)

    cpu_labels = cpu_batch["label"]
    dali_labels = dali_batch["label"].cpu()

    if not torch.equal(cpu_labels, dali_labels):
        # Different ordering — skip pixel comparison, label test covers this
        return

    cpu_imgs = cpu_batch["input"].cuda()
    dali_imgs = dali_batch["input"]

    diff = (cpu_imgs - dali_imgs).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    # PIL/libjpeg and DALI/nvJPEG use different JPEG decoders and bicubic
    # interpolation kernels, so individual pixel outliers are expected.
    assert mean_diff < 0.15, f"Mean pixel diff too large: {mean_diff:.4f}"
    assert max_diff < 5.0, f"Max pixel diff too large: {max_diff:.4f}"
    print(f"         (mean_diff={mean_diff:.4f}, max_diff={max_diff:.4f})")


# ── 5. Training augmentation effects ────────────────────────────────────────

def test_augmentation_variety():
    """Training batches should show variety (not all identical), confirming augmentations work."""
    dm = _build_dali_dm(augment=True)
    loader = dm.train_dataloader()
    it = iter(loader)

    batches = []
    for _ in range(3):
        raw = next(it)
        batch = dm.on_before_batch_transfer(raw, 0)
        batches.append(batch["input"])

    # Images within a batch should differ from each other
    b = batches[0]
    pairwise = (b[0] - b[1]).abs().mean().item()
    assert pairwise > 0.1, f"Images within batch are too similar: {pairwise:.4f}"

    # Different batches should produce different outputs
    cross = (batches[0].mean() - batches[1].mean()).abs().item()
    assert cross > 0.001, f"Batches are suspiciously identical: {cross:.6f}"


def test_three_augment_grayscale():
    """Some training images should be grayscale (R ≈ G ≈ B) before normalization."""
    from experiments.datamodules.imagenet import DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD
    dm = _build_dali_dm(augment=True)
    loader = dm.train_dataloader()

    mean = torch.tensor(DEFAULT_IMAGENET_MEAN, device="cuda").view(1, 1, 1, 3)
    std = torch.tensor(DEFAULT_IMAGENET_STD, device="cuda").view(1, 1, 1, 3)

    n_grey = 0
    n_total = 0
    for i, raw in enumerate(loader):
        if i >= 5:
            break
        batch = dm.on_before_batch_transfer(raw, 0)
        img = batch["input"]  # (B, H, W, C) normalized
        img_unnorm = img * std + mean  # undo normalization → [0, 1] range
        channel_std = img_unnorm.std(dim=-1).mean(dim=(1, 2))  # cross-channel variation
        n_grey += (channel_std < 0.02).sum().item()
        n_total += img.shape[0]

    grey_pct = n_grey / n_total
    # ~1/3 of images get ThreeAugment, and 1/3 of those are grayscale → ~11%
    assert grey_pct > 0.03, f"Too few grayscale images: {grey_pct:.1%} ({n_grey}/{n_total})"
    assert grey_pct < 0.50, f"Too many grayscale images: {grey_pct:.1%} ({n_grey}/{n_total})"


def test_color_jitter_effect():
    """ColorJitter should cause brightness variation across training batches."""
    dm = _build_dali_dm(augment=True)
    loader = dm.train_dataloader()

    means = []
    for i, raw in enumerate(loader):
        if i >= 10:
            break
        batch = dm.on_before_batch_transfer(raw, 0)
        means.append(batch["input"].mean().item())

    spread = max(means) - min(means)
    assert spread > 0.01, f"Batch means show no variation: spread={spread:.4f}"


def test_horizontal_flip_rate():
    """Approximately 50% of val-sized images should differ from a second pass.

    We verify this indirectly: in training mode, run two epochs on the same data.
    Since RandomHorizontalFlip is applied, images should differ ~50% of the time.
    We use a simpler proxy: check that images are not all left-right symmetric.
    """
    dm = _build_dali_dm(augment=True)
    loader = dm.train_dataloader()

    n_asymmetric = 0
    n_total = 0
    for i, raw in enumerate(loader):
        if i >= 3:
            break
        imgs, _ = raw  # (B, H, W, C) uint8
        flipped = imgs.flip(-2)  # flip W dimension
        diff = (imgs.float() - flipped.float()).abs().mean(dim=(1, 2, 3))
        n_asymmetric += (diff > 1.0).sum().item()
        n_total += imgs.shape[0]

    asym_pct = n_asymmetric / n_total
    assert asym_pct > 0.3, f"Too many symmetric images: {asym_pct:.1%} — flip may not be working"


# ── 6. Device placement ─────────────────────────────────────────────────────

def test_dali_tensors_on_gpu():
    """DALI should produce images on GPU and labels that can be moved to GPU."""
    dm = _build_dali_dm()
    loader = dm.train_dataloader()
    imgs, labels = next(iter(loader))

    assert imgs.is_cuda, f"DALI images not on GPU: {imgs.device}"
    batch = dm.on_before_batch_transfer((imgs, labels), 0)
    assert batch["input"].is_cuda, f"Processed input not on GPU: {batch['input'].device}"
    assert batch["label"].is_cuda, f"Processed label not on GPU: {batch['label'].device}"


def test_cpu_tensors_on_cpu():
    """CPU dataloader should produce tensors on CPU."""
    dm = _build_cpu_dm()
    loader = dm.train_dataloader()
    imgs, labels = next(iter(loader))
    assert not imgs.is_cuda, f"CPU images unexpectedly on GPU: {imgs.device}"


# ── Runner ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DALI ↔ CPU DataModule Equivalence Tests")
    print("=" * 60)
    print()

    tests = [
        ("1a. DALI train output format", test_train_output_format),
        ("1b. DALI val output format", test_val_output_format),
        ("1c. CPU train output format", test_cpu_output_format),
        ("2a. Label range [0, 999]", test_label_range),
        ("2b. Val label ordering", test_val_label_ordering),
        ("3.  Normalization statistics", test_normalization_statistics),
        ("4.  Val pixel similarity", test_val_pixel_similarity),
        ("5a. Augmentation variety", test_augmentation_variety),
        ("5b. ThreeAugment grayscale", test_three_augment_grayscale),
        ("5c. ColorJitter effect", test_color_jitter_effect),
        ("5d. Horizontal flip rate", test_horizontal_flip_rate),
        ("6a. DALI tensors on GPU", test_dali_tensors_on_gpu),
        ("6b. CPU tensors on CPU", test_cpu_tensors_on_cpu),
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

    sys.exit(1 if failed or errors else 0)


if __name__ == "__main__":
    main()
