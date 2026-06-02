# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for DALI RandAugment — verifies exact replication of timm's implementation.

We implement RandAugment inside DALI to match timm's ``rand_augment_transform``
(used by the ViT-5 reference for finetuning with config ``rand-m9-mstd0.5-inc1``).
Since timm operates on PIL images (CPU) and DALI on GPU tensors, the underlying
pixel operations differ at the library level.  These tests verify consistency
at every layer of abstraction:

**Layer 1 — Config parsing** (``TestConfigParsing``):
    The timm config string ``rand-m9-mstd0.5-inc1`` is parsed into the same
    semantic parameters (n, m, mstd, inc, p) that timm extracts internally.

**Layer 2 — Magnitude mapping** (``TestMagnitudeMapping``):
    timm uses a continuous magnitude scale with ``_LEVEL_DENOM = 10``.  DALI
    discretizes into ``num_magnitude_bins`` (default 31).  We verify that
    ``m_dali = round(m_timm * 30 / 10)`` produces the same effective parameter
    values (rotation degrees, shear factor, brightness multiplier, etc.).

**Layer 3 — Operation suite** (``TestSuiteComposition``):
    The DALI augmentation list is checked against timm's
    ``_RAND_INCREASING_TRANSFORMS``: same 15 operations, same per-op
    ``randomly_negate`` flags, same magnitude ranges.

**Layer 4 — Pixel-level consistency** (``TestTimmDALIPixelConsistency``):
    Individual augmentations are applied to the *same* uint8 image at the
    *same* magnitude by both timm (PIL) and DALI, and the outputs are compared.
    Deterministic ops (invert, posterize, solarize, solarize_add) are expected
    to match pixel-exactly or within 1 intensity level.  Stochastic-sign ops
    (brightness, contrast, color) are tested with a controlled positive sign
    and compared within a small tolerance.  ``equalize`` is excluded because
    PIL and DALI (OpenCV) use different histogram CDF formulas (documented in
    DALI's source).

**Layer 5 — Full-pipeline integration** (``TestDALIPipelineIntegration``):
    End-to-end DALI pipelines are built with RandAugment, ThreeAugment, and
    the full ``DALIImageNetFusedDataModule``, verifying they produce tensors of
    the correct shape and dtype without errors.

**Layer 6 — Statistical distribution** (``TestFullPipelineStatistics``):
    Both timm and DALI RandAugment are run over many images and the per-channel
    output statistics (mean, std) are compared to verify they operate in the
    same distributional regime.
"""

from typing import ClassVar

import numpy as np
import pytest

from experiments.datamodules.utils.dali_rand_augment import (
    _TIMM_LEVEL_DENOM,
    _timm_m_to_dali,
    _timm_mstd_to_dali,
    get_timm_default_suite,
    get_timm_increasing_suite,
    parse_rand_augment_config,
)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 — Config string parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigParsing:
    """Verify timm config strings are decomposed into the correct parameters."""

    def test_full_config(self):
        cfg = parse_rand_augment_config("rand-m9-mstd0.5-inc1")
        assert cfg == {"n": 2, "m": 9, "mstd": 0.5, "inc": True, "p": 0.5}

    def test_defaults(self):
        cfg = parse_rand_augment_config("rand")
        assert cfg == {"n": 2, "m": 10, "mstd": 0.0, "inc": False, "p": 0.5}

    def test_custom_n_and_p(self):
        cfg = parse_rand_augment_config("rand-n3-m7-p0.8")
        assert cfg["n"] == 3
        assert cfg["m"] == 7
        assert cfg["p"] == 0.8

    def test_mstd_inf(self):
        cfg = parse_rand_augment_config("rand-m9-mstd999")
        assert cfg["mstd"] == float("inf")

    def test_invalid_prefix(self):
        with pytest.raises(ValueError):
            parse_rand_augment_config("auto-m9")


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — Magnitude scale mapping
# ═══════════════════════════════════════════════════════════════════════════


class TestMagnitudeMapping:
    """Verify that timm magnitudes map to DALI bin indices that produce
    the same effective augmentation parameters.

    timm computes ``param = (level / 10) * range_max``.
    DALI computes  ``param = linspace(lo, hi, 31)[bin]``.
    With ``bin = round(level * 30 / 10)`` the two must agree.
    """

    def test_m9_to_dali_31bins(self):
        assert _timm_m_to_dali(9, 31) == 27

    def test_m0_to_dali(self):
        assert _timm_m_to_dali(0, 31) == 0

    def test_m10_to_dali(self):
        assert _timm_m_to_dali(10, 31) == 30

    def test_m5_to_dali(self):
        assert _timm_m_to_dali(5, 31) == 15

    def test_mstd_scaling(self):
        assert _timm_mstd_to_dali(0.5, 31) == pytest.approx(1.5)

    def test_magnitude_ranges_match_timm(self):
        num_bins = 31
        m_dali = _timm_m_to_dali(9, num_bins)
        timm_ratio = 9.0 / _TIMM_LEVEL_DENOM

        cases = [
            ("rotate", np.linspace(0, 30, num_bins), timm_ratio * 30),
            ("shear", np.linspace(0, 0.3, num_bins), timm_ratio * 0.3),
            ("enhance", np.linspace(0, 0.9, num_bins), timm_ratio * 0.9),
            ("translate_rel", np.linspace(0, 0.45, num_bins), timm_ratio * 0.45),
            ("solarize_add", np.linspace(0, 110, num_bins), timm_ratio * 110),
        ]
        for name, mags, expected in cases:
            assert mags[m_dali] == pytest.approx(expected), (
                f"{name}: DALI bin {m_dali} gives {mags[m_dali]}, expected {expected}"
            )

        # Solarize threshold: int rounding can differ by at most 1
        solarize_mags = np.linspace(256, 0, num_bins)
        timm_thresh = 256 - int(timm_ratio * 256)
        assert abs(solarize_mags[m_dali] - timm_thresh) < 2


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 — Operation suite composition
# ═══════════════════════════════════════════════════════════════════════════


class TestSuiteComposition:
    """Verify the DALI augmentation list matches timm's ``_RAND_INCREASING_TRANSFORMS``."""

    DALI_TO_TIMM: ClassVar[dict[str, str]] = {
        "auto_contrast": "AutoContrast",
        "equalize": "Equalize",
        "invert": "Invert",
        "rotate": "Rotate",
        "posterize": "PosterizeIncreasing",
        "solarize": "SolarizeIncreasing",
        "solarize_add": "SolarizeAdd",
        "color": "ColorIncreasing",
        "contrast": "ContrastIncreasing",
        "brightness": "BrightnessIncreasing",
        "sharpness": "SharpnessIncreasing",
        "shear_x": "ShearX",
        "shear_y": "ShearY",
        "translate_x": "TranslateXRel",
        "translate_y": "TranslateYRel",
    }

    def test_increasing_suite_has_15_ops(self):
        suite = get_timm_increasing_suite(use_shape=True, max_translate_rel=0.45)
        assert len(suite) == 15

    def test_increasing_suite_names_match(self):
        suite = get_timm_increasing_suite(use_shape=True, max_translate_rel=0.45)
        dali_names = {aug.name for aug in suite}
        assert dali_names == set(self.DALI_TO_TIMM.keys())

    def test_no_identity(self):
        suite = get_timm_increasing_suite(use_shape=True, max_translate_rel=0.45)
        assert "identity" not in {aug.name for aug in suite}

    def test_has_invert_and_solarize_add(self):
        names = {a.name for a in get_timm_increasing_suite(use_shape=True)}
        assert "invert" in names
        assert "solarize_add" in names

    def test_default_suite_also_has_15_ops(self):
        assert len(get_timm_default_suite(use_shape=True, max_translate_rel=0.45)) == 15

    def test_randomly_negate_flags(self):
        suite = get_timm_increasing_suite(use_shape=True, max_translate_rel=0.45)
        should_negate = {
            "shear_x",
            "shear_y",
            "translate_x",
            "translate_y",
            "rotate",
            "brightness",
            "contrast",
            "color",
            "sharpness",
        }
        should_not = {
            "posterize",
            "solarize",
            "solarize_add",
            "equalize",
            "auto_contrast",
            "invert",
        }
        for aug in suite:
            if aug.name in should_negate:
                assert aug.randomly_negate, f"{aug.name} should negate"
            elif aug.name in should_not:
                assert not aug.randomly_negate, f"{aug.name} should not negate"

    def test_posterize_range_decreasing(self):
        posterize = next(a for a in get_timm_increasing_suite(use_shape=True) if a.name == "posterize")
        assert posterize.mag_range == (4, 0)

    def test_solarize_range_decreasing(self):
        solarize = next(a for a in get_timm_increasing_suite(use_shape=True) if a.name == "solarize")
        assert solarize.mag_range == (256, 0)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4 — Pixel-level consistency (timm vs DALI, per operation)
# ═══════════════════════════════════════════════════════════════════════════

_DALI_AVAILABLE = False
try:
    import nvidia.dali  # noqa: F401

    _DALI_AVAILABLE = True
except ImportError:
    pass


@pytest.fixture
def test_image():
    """Deterministic 224x224 RGB uint8 image with varied pixel values."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_image_dir(tmp_path):
    """Tiny ImageNet-like folder with random JPEG images for pipeline tests."""
    from PIL import Image

    for split in ("train", "val"):
        class_dir = tmp_path / split / "n01440764"
        class_dir.mkdir(parents=True)
        for i in range(8):
            img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
            img.save(class_dir / f"img_{i:04d}.JPEG")
    return tmp_path


def _get_dali_aug(name):
    """Look up a single augmentation by name from the increasing suite."""
    suite = get_timm_increasing_suite(use_shape=True, max_translate_rel=0.45)
    matches = [a for a in suite if a.name == name]
    assert len(matches) == 1, f"No unique match for {name}"
    return matches[0]


def _apply_dali_op(img_np, augmentation, magnitude_bin=None, num_magnitude_bins=31, force_positive_sign=True):
    """Apply a single DALI augmentation to a numpy uint8 HWC image on GPU.

    For ops with ``randomly_negate=True``, ``force_positive_sign`` controls
    the sign so comparisons with timm are deterministic.
    """
    from nvidia.dali import fn, pipeline_def, types
    from nvidia.dali.auto_aug.core import signed_bin

    has_mag = augmentation.mag_range is not None
    needs_sign = has_mag and augmentation.randomly_negate

    sign_val = 0 if force_positive_sign else 1

    @pipeline_def(batch_size=1, num_threads=1, device_id=0, enable_conditionals=True)
    def pipe():
        images = fn.external_source(name="input", layout="HWC")
        images = images.gpu()
        if has_mag:
            if needs_sign:
                sign = fn.random.uniform(values=[sign_val], dtype=types.INT32)
                mag_node = fn.random.uniform(values=[magnitude_bin], dtype=types.INT32)
                mag = signed_bin(mag_node, random_sign=sign)
            else:
                mag = magnitude_bin
            result = augmentation(
                images,
                magnitude_bin=mag,
                num_magnitude_bins=num_magnitude_bins,
            )
        else:
            result = augmentation(images, magnitude_bin=None, num_magnitude_bins=None)
        return result

    p = pipe()
    p.build()
    p.feed_input("input", [img_np])
    output = p.run()
    return output[0].as_cpu().as_array()[0]


@pytest.mark.skipif(not _DALI_AVAILABLE, reason="DALI not installed")
class TestTimmDALIPixelConsistency:
    """Compare individual augmentations applied to the same image by both
    timm (PIL) and DALI at the same magnitude.

    Deterministic operations (no random sign) are compared pixel-exactly or
    within 1 intensity level.  Stochastic-sign operations are tested with
    a controlled positive sign.

    ``equalize`` is skipped because PIL and DALI intentionally use different
    histogram equalization formulas (PIL vs OpenCV), as documented in DALI
    source: ``augmentations.py:equalize``.
    """

    M_DALI = _timm_m_to_dali(9, 31)  # bin 27

    # -- Deterministic pixel ops (no random sign) --

    def test_invert_pixel_exact(self, test_image):
        """``255 - pixel`` — trivially identical between PIL and DALI."""
        from PIL import Image, ImageOps

        timm_out = np.array(ImageOps.invert(Image.fromarray(test_image)))
        dali_out = _apply_dali_op(test_image, _get_dali_aug("invert"))
        np.testing.assert_array_equal(timm_out, dali_out)

    def test_posterize_pixel_exact(self, test_image):
        """At m=9 (increasing): both keep only the top 1 bit -> ``pixel & 0x80``."""
        from PIL import Image, ImageOps

        bits_to_keep = 4 - int(9.0 / _TIMM_LEVEL_DENOM * 4)  # = 1
        timm_out = np.array(ImageOps.posterize(Image.fromarray(test_image), bits_to_keep))
        dali_out = _apply_dali_op(test_image, _get_dali_aug("posterize"), magnitude_bin=self.M_DALI)
        np.testing.assert_array_equal(timm_out, dali_out)

    def test_solarize_near_exact(self, test_image):
        """Threshold inversion.  timm uses ``int()`` truncation, DALI uses
        ``float`` comparison against uint8; effective threshold matches for
        integer pixel values so results should be identical or differ by at
        most 1 intensity level at the boundary.
        """
        timm_thresh = 256 - int(9.0 / _TIMM_LEVEL_DENOM * 256)  # 26
        timm_out = test_image.copy()
        mask = timm_out >= timm_thresh
        timm_out[mask] = 255 - timm_out[mask]

        dali_out = _apply_dali_op(test_image, _get_dali_aug("solarize"), magnitude_bin=self.M_DALI)

        diff = np.abs(timm_out.astype(np.int16) - dali_out.astype(np.int16))
        assert diff.max() <= 1, f"Max pixel diff {diff.max()}"
        pct_diff = (diff > 0).sum() / diff.size
        assert pct_diff < 0.01, f"{pct_diff:.4%} of pixels differ (expect <1%)"

    def test_solarize_add_near_exact(self, test_image):
        """Add shift to pixels below 128; clamp to 255."""
        shift = int(min(128, int(9.0 / _TIMM_LEVEL_DENOM * 110)))  # 99
        timm_out = test_image.copy().astype(np.int16)
        below_mask = test_image < 128
        timm_out[below_mask] = np.minimum(timm_out[below_mask] + shift, 255)
        timm_out = timm_out.astype(np.uint8)

        dali_out = _apply_dali_op(test_image, _get_dali_aug("solarize_add"), magnitude_bin=self.M_DALI)

        diff = np.abs(timm_out.astype(np.int16) - dali_out.astype(np.int16))
        assert diff.max() <= 1, f"Max pixel diff {diff.max()}"
        pct_diff = (diff > 0).sum() / diff.size
        assert pct_diff < 0.01, f"{pct_diff:.4%} of pixels differ (expect <1%)"

    def test_auto_contrast_close(self, test_image):
        """Per-channel stretch to [0, 255].  PIL and DALI use the same formula
        but different rounding, so we allow a small tolerance.
        """
        from PIL import Image, ImageOps

        timm_out = np.array(ImageOps.autocontrast(Image.fromarray(test_image)))
        dali_out = _apply_dali_op(test_image, _get_dali_aug("auto_contrast"))

        diff = np.abs(timm_out.astype(np.int16) - dali_out.astype(np.int16))
        assert diff.mean() < 2.0, f"Mean diff {diff.mean():.2f}"

    # -- Stochastic-sign ops (controlled positive sign) --

    def test_brightness_positive_sign(self, test_image):
        """Brightness with positive sign at m=9: factor = 1 + 0.81 = 1.81.
        PIL and DALI both multiply by the factor; small rounding differences
        are expected from uint8 clamping.
        """
        from PIL import Image, ImageEnhance

        factor = 1.0 + 9.0 / _TIMM_LEVEL_DENOM * 0.9  # 1.81
        timm_out = np.array(ImageEnhance.Brightness(Image.fromarray(test_image)).enhance(factor))
        dali_out = _apply_dali_op(
            test_image, _get_dali_aug("brightness"), magnitude_bin=self.M_DALI, force_positive_sign=True
        )

        diff = np.abs(timm_out.astype(np.int16) - dali_out.astype(np.int16))
        assert diff.mean() < 3.0, f"Mean diff {diff.mean():.2f}"
        assert diff.max() <= 10, f"Max diff {diff.max()}"

    def test_color_positive_sign(self, test_image):
        """Color (saturation) with positive sign at m=9: factor = 1.81."""
        from PIL import Image, ImageEnhance

        factor = 1.0 + 9.0 / _TIMM_LEVEL_DENOM * 0.9
        timm_out = np.array(ImageEnhance.Color(Image.fromarray(test_image)).enhance(factor))
        dali_out = _apply_dali_op(
            test_image, _get_dali_aug("color"), magnitude_bin=self.M_DALI, force_positive_sign=True
        )

        diff = np.abs(timm_out.astype(np.int16) - dali_out.astype(np.int16))
        assert diff.mean() < 5.0, f"Mean diff {diff.mean():.2f}"

    def test_contrast_positive_sign(self, test_image):
        """Contrast with positive sign at m=9: factor = 1.81."""
        from PIL import Image, ImageEnhance

        factor = 1.0 + 9.0 / _TIMM_LEVEL_DENOM * 0.9
        timm_out = np.array(ImageEnhance.Contrast(Image.fromarray(test_image)).enhance(factor))
        dali_out = _apply_dali_op(
            test_image, _get_dali_aug("contrast"), magnitude_bin=self.M_DALI, force_positive_sign=True
        )

        diff = np.abs(timm_out.astype(np.int16) - dali_out.astype(np.int16))
        assert diff.mean() < 5.0, f"Mean diff {diff.mean():.2f}"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 5 — Full-pipeline integration
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _DALI_AVAILABLE, reason="DALI not installed")
class TestDALIPipelineIntegration:
    """Build and run complete DALI pipelines to verify they produce tensors
    of the correct shape and dtype without errors.
    """

    def test_pipeline_with_rand_augment(self, dummy_image_dir):
        from nvidia.dali import fn, pipeline_def, types

        from experiments.datamodules.utils.dali_rand_augment import dali_rand_augment

        @pipeline_def(batch_size=4, num_threads=2, device_id=0, enable_conditionals=True)
        def test_pipe():
            jpegs, labels = fn.readers.file(
                file_root=str(dummy_image_dir / "train"),
                random_shuffle=True,
                name="reader",
            )
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
            images = fn.resize(images, size=(224, 224))
            images = dali_rand_augment(images, "rand-m9-mstd0.5-inc1", shape=(224, 224))
            images = fn.crop_mirror_normalize(
                images,
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            )
            return images, labels

        pipe = test_pipe()
        pipe.build()
        imgs = pipe.run()[0].as_cpu().as_array()
        assert imgs.shape == (4, 3, 224, 224) and imgs.dtype == np.float32

    def test_pipeline_without_rand_augment(self, dummy_image_dir):
        from nvidia.dali import fn, pipeline_def, types

        @pipeline_def(batch_size=4, num_threads=2, device_id=0, enable_conditionals=True)
        def test_pipe():
            jpegs, labels = fn.readers.file(
                file_root=str(dummy_image_dir / "train"),
                random_shuffle=True,
                name="reader",
            )
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
            images = fn.resize(images, size=(224, 224))
            images = fn.crop_mirror_normalize(
                images,
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            )
            return images, labels

        pipe = test_pipe()
        pipe.build()
        assert pipe.run()[0].as_cpu().as_array().shape == (4, 3, 224, 224)

    def test_three_augment_plus_rand_augment(self, dummy_image_dir):
        """ThreeAugment and RandAugment applied sequentially (not mutually exclusive)."""
        from nvidia.dali import fn, pipeline_def, types

        from experiments.datamodules.dali_imagenet_fused import _solarize
        from experiments.datamodules.utils.dali_rand_augment import dali_rand_augment

        @pipeline_def(batch_size=4, num_threads=2, device_id=0, enable_conditionals=True)
        def test_pipe():
            jpegs, labels = fn.readers.file(
                file_root=str(dummy_image_dir / "train"),
                random_shuffle=True,
                name="reader",
            )
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
            images = fn.resize(images, size=(224, 224))
            coin = fn.random.uniform(range=(0.0, 1.0))
            if coin < (1.0 / 3.0):
                grey = fn.color_space_conversion(images, image_type=types.RGB, output_type=types.GRAY)
                images = fn.cat(grey, grey, grey, axis=2)
            else:
                if coin < (2.0 / 3.0):
                    images = _solarize(images)
                else:
                    images = fn.gaussian_blur(images, sigma=fn.random.uniform(range=(0.1, 2.0)), window_size=5)
            images = dali_rand_augment(images, "rand-m9-mstd0.5-inc1", shape=(224, 224))
            images = fn.crop_mirror_normalize(
                images,
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            )
            return images, labels

        pipe = test_pipe()
        pipe.build()
        assert pipe.run()[0].as_cpu().as_array().shape == (4, 3, 224, 224)

    def test_fixed_magnitude_no_mstd(self, dummy_image_dir):
        from nvidia.dali import fn, pipeline_def, types

        from experiments.datamodules.utils.dali_rand_augment import dali_rand_augment

        @pipeline_def(batch_size=4, num_threads=2, device_id=0, enable_conditionals=True)
        def test_pipe():
            jpegs, labels = fn.readers.file(
                file_root=str(dummy_image_dir / "train"),
                random_shuffle=True,
                name="reader",
            )
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
            images = fn.resize(images, size=(224, 224))
            images = dali_rand_augment(images, "rand-m9-inc1", shape=(224, 224))
            images = fn.crop_mirror_normalize(
                images,
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            )
            return images, labels

        pipe = test_pipe()
        pipe.build()
        assert pipe.run()[0].as_cpu().as_array().shape == (4, 3, 224, 224)

    def test_datamodule_finetune_config(self, dummy_image_dir):
        from experiments.datamodules.dali_imagenet_fused import AugmentConfig, DALIImageNetFusedDataModule, MixupConfig

        dm = DALIImageNetFusedDataModule(
            data_dir=str(dummy_image_dir),
            imagefolder_dir=str(dummy_image_dir),
            batch_size=4,
            num_workers=2,
            seed=42,
            image_size=224,
            final_image_size=224,
            num_classes=1000,
            task="classification",
            eval_crop_ratio=1.0,
            augment_cfg=AugmentConfig(
                use_three_augment=False,
                color_jitter=0.3,
                rand_augment="rand-m9-mstd0.5-inc1",
                random_erasing_prob=0.0,
            ),
            mixup_cfg=MixupConfig(mixup=0.8, cutmix=1.0, smoothing=0.1),
            device_id=0,
        )
        assert dm._rand_augment_config == "rand-m9-mstd0.5-inc1"

    def test_datamodule_pretrain_config(self, dummy_image_dir):
        from experiments.datamodules.dali_imagenet_fused import AugmentConfig, DALIImageNetFusedDataModule, MixupConfig

        dm = DALIImageNetFusedDataModule(
            data_dir=str(dummy_image_dir),
            imagefolder_dir=str(dummy_image_dir),
            batch_size=4,
            num_workers=2,
            seed=42,
            image_size=224,
            final_image_size=224,
            num_classes=1000,
            task="classification",
            augment_cfg=AugmentConfig(use_three_augment=True, color_jitter=0.3),
            mixup_cfg=MixupConfig(mixup=0.8, cutmix=1.0, smoothing=0.0),
            device_id=0,
        )
        assert dm._use_three_augment is True
        assert dm._rand_augment_config == ""


# ═══════════════════════════════════════════════════════════════════════════
# Layer 6 — Statistical distribution (timm vs DALI over many samples)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _DALI_AVAILABLE, reason="DALI not installed")
class TestFullPipelineStatistics:
    """Run both timm and DALI RandAugment over many random images and
    compare per-channel output statistics.

    This is the highest-level consistency check: it verifies that the two
    backends produce outputs in the same distributional regime, even though
    individual pixel values will differ due to different underlying libraries
    (PIL vs DALI/CUDA).
    """

    NUM_IMAGES = 64

    def _timm_augment_batch(self, images_np):
        """Apply timm's RandAugment to a batch of uint8 HWC numpy images."""
        from PIL import Image
        from timm.data.auto_augment import rand_augment_transform

        transform = rand_augment_transform(
            config_str="rand-m9-mstd0.5-inc1",
            hparams={"img_mean": (128, 128, 128)},
        )
        results = []
        for img_np in images_np:
            pil_img = Image.fromarray(img_np)
            aug_pil = transform(pil_img)
            results.append(np.array(aug_pil))
        return np.stack(results)

    def _dali_augment_batch(self, images_np):
        """Apply DALI's RandAugment to a batch of uint8 HWC numpy images."""
        from nvidia.dali import fn, pipeline_def

        from experiments.datamodules.utils.dali_rand_augment import dali_rand_augment

        batch_size = len(images_np)

        @pipeline_def(batch_size=batch_size, num_threads=2, device_id=0, enable_conditionals=True)
        def pipe():
            images = fn.external_source(name="input", layout="HWC")
            images = images.gpu()
            images = dali_rand_augment(images, "rand-m9-mstd0.5-inc1", shape=(224, 224))
            return images

        p = pipe()
        p.build()
        p.feed_input("input", list(images_np))
        output = p.run()
        return output[0].as_cpu().as_array()

    def test_per_channel_mean_similar(self, test_image):
        """Per-channel means of augmented batches should be within ~20 intensity
        levels of each other.  (The same source image is replicated to form a
        batch; randomness comes solely from the augmentation.)
        """
        batch = np.stack([test_image] * self.NUM_IMAGES)
        timm_out = self._timm_augment_batch(batch)
        dali_out = self._dali_augment_batch(batch)

        timm_means = timm_out.mean(axis=(0, 1, 2))
        dali_means = dali_out.mean(axis=(0, 1, 2))

        for ch in range(3):
            diff = abs(timm_means[ch] - dali_means[ch])
            assert diff < 20, (
                f"Channel {ch} mean diff {diff:.1f} (timm={timm_means[ch]:.1f}, dali={dali_means[ch]:.1f})"
            )

    def test_per_channel_std_similar(self, test_image):
        """Per-channel standard deviations should be in the same ballpark."""
        batch = np.stack([test_image] * self.NUM_IMAGES)
        timm_out = self._timm_augment_batch(batch)
        dali_out = self._dali_augment_batch(batch)

        timm_stds = timm_out.std(axis=(0, 1, 2))
        dali_stds = dali_out.std(axis=(0, 1, 2))

        for ch in range(3):
            ratio = timm_stds[ch] / max(dali_stds[ch], 1e-6)
            assert 0.5 < ratio < 2.0, (
                f"Channel {ch} std ratio {ratio:.2f} (timm={timm_stds[ch]:.1f}, dali={dali_stds[ch]:.1f})"
            )
