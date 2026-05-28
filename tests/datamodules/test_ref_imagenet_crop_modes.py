# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``ImageNetDataModule.train_crop_mode``.

A second ``train_crop_mode`` option was added so the JiT-exact diffusion
configs can opt into the deterministic ADM-style center crop that JiT
uses, instead of the legacy ``RandomResizedCrop(scale=(0.08, 1.0))``.

These tests pin the new flag's behaviour at the transform level (we don't
need a real dataset to verify the pipeline composition) and also exercise
the JiT-style crop end-to-end on a synthetic PIL image.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from timm.data.transforms import RandomResizedCropAndInterpolation
from torchvision import transforms

from experiments.datamodules._deprecated.ref_imagenet import (
    ImageNetDataModule,
    _center_crop_arr_adm,
    _CenterCropArrADM,
)


def _make_dm(**overrides) -> ImageNetDataModule:
    """Build a minimal ImageNetDataModule without invoking HF downloads."""
    kwargs = {
        "data_dir": "/tmp/nonexistent",
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "seed": 0,
        "image_size": 64,
        "task": "generation",
        "drop_labels": False,
    }
    kwargs.update(overrides)
    return ImageNetDataModule(**kwargs)


# =============================================================================
# Construction-time validation
# =============================================================================


def test_default_train_crop_mode_is_random_resized() -> None:
    """Backward compatibility: the legacy default is preserved."""
    dm = _make_dm()
    assert dm.train_crop_mode == "random_resized"


def test_explicit_train_crop_mode_center() -> None:
    """The new ``"center"`` value is accepted and stored."""
    dm = _make_dm(train_crop_mode="center")
    assert dm.train_crop_mode == "center"


def test_invalid_train_crop_mode_raises() -> None:
    """Unknown values fail fast with a clear message."""
    with pytest.raises(ValueError, match="train_crop_mode"):
        _make_dm(train_crop_mode="bogus")


# =============================================================================
# Transform composition
# =============================================================================


def _first_op_type(transform: transforms.Compose) -> type:
    """Return the type of the first op inside a torchvision ``Compose``."""
    return type(transform.transforms[0])


def test_random_resized_mode_uses_random_resized_crop() -> None:
    dm = _make_dm(train_crop_mode="random_resized")
    train_tf = dm._build_transform(train=True)
    assert _first_op_type(train_tf) is RandomResizedCropAndInterpolation


def test_center_mode_uses_adm_center_crop() -> None:
    """The new mode swaps the leading op for ``_CenterCropArrADM``."""
    dm = _make_dm(train_crop_mode="center")
    train_tf = dm._build_transform(train=True)
    first = train_tf.transforms[0]
    assert isinstance(first, _CenterCropArrADM)
    assert first.image_size == dm.image_size


def test_center_mode_does_not_affect_val_transform() -> None:
    """Val pipeline is unchanged regardless of ``train_crop_mode``."""
    dm_random = _make_dm(train_crop_mode="random_resized")
    dm_center = _make_dm(train_crop_mode="center")
    # Val transform uses Resize + CenterCrop irrespective of train_crop_mode.
    val_tf_random = dm_random._build_transform(train=False)
    val_tf_center = dm_center._build_transform(train=False)
    types_random = [type(op).__name__ for op in val_tf_random.transforms]
    types_center = [type(op).__name__ for op in val_tf_center.transforms]
    assert types_random == types_center


def test_center_mode_keeps_horizontal_flip() -> None:
    """The flip is still in the train pipeline — only the leading crop changed."""
    dm = _make_dm(train_crop_mode="center")
    train_tf = dm._build_transform(train=True)
    type_names = [type(op).__name__ for op in train_tf.transforms]
    assert "RandomHorizontalFlip" in type_names


# =============================================================================
# ADM center crop math
# =============================================================================


def test_center_crop_arr_produces_exact_target_size() -> None:
    """ADM crop yields exactly ``(image_size, image_size, 3)`` for any input."""
    rng = np.random.default_rng(0)
    for src_size in [(73, 130), (300, 200), (1000, 1000), (1023, 257)]:
        arr = rng.integers(0, 255, size=(src_size[0], src_size[1], 3), dtype=np.uint8)
        pil = Image.fromarray(arr)
        out = _center_crop_arr_adm(pil, image_size=64)
        assert out.size == (64, 64), f"got {out.size} for source {src_size}"


def test_center_crop_arr_is_deterministic() -> None:
    """Repeated calls on the same input give bit-identical bytes."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(512, 768, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    a = np.asarray(_center_crop_arr_adm(pil, 256))
    b = np.asarray(_center_crop_arr_adm(pil, 256))
    assert np.array_equal(a, b)


def test_center_crop_callable_class_is_picklable() -> None:
    """``_CenterCropArrADM`` must pickle so dataloader workers can fork it."""
    import pickle

    op = _CenterCropArrADM(image_size=128)
    payload = pickle.dumps(op)
    op_restored = pickle.loads(payload)
    assert op_restored.image_size == 128
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(300, 200, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    out_original = np.asarray(op(pil))
    out_restored = np.asarray(op_restored(pil))
    assert np.array_equal(out_original, out_restored)


# =============================================================================
# End-to-end transform pipeline (no actual dataset access)
# =============================================================================


def test_center_mode_pipeline_produces_normalised_tensor() -> None:
    """A full ``train_crop_mode="center"`` pipeline gives the right tensor shape and range."""
    dm = _make_dm(train_crop_mode="center", image_size=64)
    train_tf = dm._build_transform(train=True)
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(300, 200, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    out = train_tf(pil)
    # Generation task normalises to [-1, 1].
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 64, 64)
    assert out.dtype == torch.float32
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6
