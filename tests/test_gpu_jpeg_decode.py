"""Tests verifying GPU JPEG decode pipeline equivalence with the CPU/PIL path.

Validates that:
1. nvJPEG (GPU) and libjpeg (PIL) produce near-identical raw decodes
2. The deterministic val transform (Resize + CenterCrop + Normalize) produces
   outputs within acceptable tolerance across both pipelines
3. Output shapes and dtypes match
4. The _ImageNetRawBytesDataset returns valid byte tensors
5. The _raw_bytes_collate function produces correctly shaped outputs
6. The split GPU transform pipeline (per-image + batch) produces the same
   result as a single-pass pipeline

Run:
    PYTHONPATH=. python -m pytest tests/test_gpu_jpeg_decode.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms
from torchvision.io import decode_jpeg, read_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms_v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.datamodules.imagenet import (
    DEFAULT_IMAGENET_MEAN,
    DEFAULT_IMAGENET_STD,
    _ImageNetRawBytesDataset,
    _raw_bytes_collate,
)

IMAGENET_FOLDER = Path("/shared/data/image_datasets/imagenet_folder")
NEEDS_DATA = pytest.mark.skipif(
    not IMAGENET_FOLDER.exists(),
    reason="ImageNet folder data not available",
)
NEEDS_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

IMAGE_SIZE = 224
MEAN = list(DEFAULT_IMAGENET_MEAN)
STD = list(DEFAULT_IMAGENET_STD)


@pytest.fixture(scope="module")
def sample_paths():
    """Return a deterministic set of 50 validation JPEG paths."""
    val_dir = IMAGENET_FOLDER / "val"
    if not val_dir.exists():
        pytest.skip("ImageNet val folder not found")
    paths = sorted(val_dir.rglob("*.jpg"))[:50]
    if len(paths) < 10:
        pytest.skip("Not enough JPEG files found")
    return paths


# ---------------------------------------------------------------------------
# 1. Raw decode equivalence
# ---------------------------------------------------------------------------

@NEEDS_DATA
@NEEDS_CUDA
class TestRawDecodeEquivalence:
    """nvJPEG vs libjpeg produce near-identical uint8 pixel values."""

    def test_majority_pixel_identical(self, sample_paths):
        """At least 80% of images should decode to identical pixels."""
        identical = 0
        for p in sample_paths:
            pil_np = np.array(Image.open(p).convert("RGB"))
            gpu_np = (
                decode_jpeg(read_file(str(p)), device="cuda:0")
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            if np.array_equal(pil_np, gpu_np):
                identical += 1
        ratio = identical / len(sample_paths)
        assert ratio >= 0.80, f"Only {ratio:.0%} pixel-identical (expected ≥80%)"

    def test_max_pixel_diff_bounded(self, sample_paths):
        """Max per-pixel difference should stay below 50/255 for all images."""
        for p in sample_paths:
            pil_np = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
            gpu_np = (
                decode_jpeg(read_file(str(p)), device="cuda:0")
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            max_diff = np.abs(pil_np - gpu_np).max()
            assert max_diff < 50, f"{p.name}: max pixel diff {max_diff} ≥ 50"

    def test_mean_l1_negligible(self, sample_paths):
        """Mean L1 across all pixels should be < 1/255."""
        l1s = []
        for p in sample_paths:
            pil_np = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
            gpu_np = (
                decode_jpeg(read_file(str(p)), device="cuda:0")
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            l1s.append(np.abs(pil_np - gpu_np).mean())
        assert np.mean(l1s) < 1.0, f"Mean L1 = {np.mean(l1s):.3f} (expected < 1.0)"


# ---------------------------------------------------------------------------
# 2. Deterministic val transform equivalence
# ---------------------------------------------------------------------------

@NEEDS_DATA
@NEEDS_CUDA
class TestValTransformEquivalence:
    """Full validation pipeline (Resize→CenterCrop→Normalize) across both paths."""

    @pytest.fixture(scope="class")
    def pil_val_transform(self):
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE + 32, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    @pytest.fixture(scope="class")
    def gpu_per_img(self):
        return transforms_v2.Compose([
            transforms_v2.Resize(IMAGE_SIZE + 32, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms_v2.CenterCrop(IMAGE_SIZE),
        ])

    @pytest.fixture(scope="class")
    def gpu_batch(self):
        return transforms_v2.Compose([
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=MEAN, std=STD),
        ])

    def test_output_shapes_match(self, sample_paths, pil_val_transform, gpu_per_img, gpu_batch):
        p = sample_paths[0]
        pil_out = pil_val_transform(Image.open(p).convert("RGB"))
        gpu_img = decode_jpeg(read_file(str(p)), device="cuda:0")
        gpu_out = gpu_batch(gpu_per_img(gpu_img).unsqueeze(0)).squeeze(0)
        assert pil_out.shape == gpu_out.cpu().shape

    def test_output_dtype_float32(self, sample_paths, gpu_per_img, gpu_batch):
        p = sample_paths[0]
        gpu_img = decode_jpeg(read_file(str(p)), device="cuda:0")
        gpu_out = gpu_batch(gpu_per_img(gpu_img).unsqueeze(0)).squeeze(0)
        assert gpu_out.dtype == torch.float32

    def test_normalized_max_diff_bounded(self, sample_paths, pil_val_transform, gpu_per_img, gpu_batch):
        """After normalization the max diff should stay below 0.5 (in σ-space).

        This accounts for both decoder and resize interpolation differences.
        A threshold of 0.5 in normalised space corresponds to ~0.11 in [0,1]
        pixel space (std ≈ 0.22), well within acceptable tolerance.
        """
        for p in sample_paths:
            pil_out = pil_val_transform(Image.open(p).convert("RGB"))
            gpu_img = decode_jpeg(read_file(str(p)), device="cuda:0")
            gpu_out = gpu_batch(gpu_per_img(gpu_img).unsqueeze(0)).squeeze(0).cpu()
            max_diff = (pil_out - gpu_out).abs().max().item()
            assert max_diff < 0.5, f"{p.name}: normalised max diff {max_diff:.4f} ≥ 0.5"

    def test_mean_abs_diff_small(self, sample_paths, pil_val_transform, gpu_per_img, gpu_batch):
        """Mean absolute difference across all pixels and images should be tiny."""
        diffs = []
        for p in sample_paths:
            pil_out = pil_val_transform(Image.open(p).convert("RGB"))
            gpu_img = decode_jpeg(read_file(str(p)), device="cuda:0")
            gpu_out = gpu_batch(gpu_per_img(gpu_img).unsqueeze(0)).squeeze(0).cpu()
            diffs.append((pil_out - gpu_out).abs().mean().item())
        assert np.mean(diffs) < 0.01, f"Mean abs diff = {np.mean(diffs):.5f} (expected < 0.01)"


# ---------------------------------------------------------------------------
# 3. Split pipeline consistency (per-image + batch == single-pass)
# ---------------------------------------------------------------------------

@NEEDS_DATA
@NEEDS_CUDA
class TestSplitPipelineConsistency:
    """Verify that running per-image then batch ops gives the same result as
    a single composed pipeline applied image-by-image."""

    def test_split_equals_combined(self, sample_paths):
        per_img = transforms_v2.Compose([
            transforms_v2.Resize(IMAGE_SIZE + 32, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms_v2.CenterCrop(IMAGE_SIZE),
        ])
        batch_ops = transforms_v2.Compose([
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=MEAN, std=STD),
        ])
        combined = transforms_v2.Compose([
            transforms_v2.Resize(IMAGE_SIZE + 32, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms_v2.CenterCrop(IMAGE_SIZE),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=MEAN, std=STD),
        ])

        for p in sample_paths[:10]:
            gpu_img = decode_jpeg(read_file(str(p)), device="cuda:0")

            split_out = batch_ops(per_img(gpu_img).unsqueeze(0)).squeeze(0)
            combined_out = combined(gpu_img)

            torch.testing.assert_close(split_out, combined_out, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 4. Dataset and collate utilities
# ---------------------------------------------------------------------------

@NEEDS_DATA
class TestRawBytesDatasetAndCollate:
    """Verify _ImageNetRawBytesDataset and _raw_bytes_collate work correctly."""

    def test_dataset_returns_bytes_tensor(self):
        ds = _ImageNetRawBytesDataset(IMAGENET_FOLDER, split="validation")
        assert len(ds) > 0
        raw_bytes, label = ds[0]
        assert isinstance(raw_bytes, torch.Tensor)
        assert raw_bytes.dtype == torch.uint8
        assert raw_bytes.ndim == 1
        assert isinstance(label, int)

    def test_dataset_label_in_range(self):
        ds = _ImageNetRawBytesDataset(IMAGENET_FOLDER, split="validation")
        _, label = ds[0]
        assert 0 <= label < 1000

    def test_collate_output_structure(self):
        ds = _ImageNetRawBytesDataset(IMAGENET_FOLDER, split="validation")
        batch = [ds[i] for i in range(4)]
        bytes_list, labels = _raw_bytes_collate(batch)
        assert isinstance(bytes_list, list)
        assert len(bytes_list) == 4
        assert labels.shape == (4,)
        assert labels.dtype == torch.long

    @NEEDS_CUDA
    def test_collated_bytes_decodable_on_gpu(self):
        ds = _ImageNetRawBytesDataset(IMAGENET_FOLDER, split="validation")
        batch = [ds[i] for i in range(4)]
        bytes_list, _ = _raw_bytes_collate(batch)
        for raw in bytes_list:
            img = decode_jpeg(raw, device="cuda:0")
            assert img.ndim == 3
            assert img.shape[0] == 3
            assert img.dtype == torch.uint8


# ---------------------------------------------------------------------------
# 5. Augmentation noise dominates decoder noise
# ---------------------------------------------------------------------------

@NEEDS_DATA
@NEEDS_CUDA
class TestAugmentationDominatesDecoderNoise:
    """Confirm that training augmentations produce far more variance than
    the nvJPEG-vs-libjpeg decoder difference."""

    def test_crop_noise_exceeds_decoder_noise(self, sample_paths):
        resize = transforms.Compose([
            transforms.Resize(IMAGE_SIZE + 32, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        crop = transforms.RandomCrop(IMAGE_SIZE)

        decoder_l1s = []
        crop_l1s = []

        for p in sample_paths[:20]:
            pil_img = Image.open(p).convert("RGB")
            pil_np = np.array(pil_img, dtype=np.float32)
            gpu_np = (
                decode_jpeg(read_file(str(p)), device="cuda:0")
                .permute(1, 2, 0).cpu().numpy().astype(np.float32)
            )
            decoder_l1s.append(np.abs(pil_np - gpu_np).mean() / 255.0)

            resized = resize(pil_img)
            c1, c2 = crop(resized), crop(resized)
            crop_l1s.append((c1 - c2).abs().mean().item())

        mean_decoder = np.mean(decoder_l1s)
        mean_crop = np.mean(crop_l1s)
        ratio = mean_crop / max(mean_decoder, 1e-10)
        assert ratio > 100, (
            f"Expected crop noise ≫ decoder noise, got ratio {ratio:.0f}x "
            f"(crop L1={mean_crop:.5f}, decoder L1={mean_decoder:.5f})"
        )
