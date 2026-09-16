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

"""CPU-only motion recall tests; no image downloads or CUDA extensions required."""

from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset

from experiments.datamodules import spatial_recall_dataset as motion
from nvsubquadratic.lazy_config import LazyConfig


def images():
    image = torch.zeros(1, 4, 4)
    image[:, 1:3, 1] = 1
    image[:, 2, 2] = 0.5
    return TensorDataset(image.unsqueeze(0).repeat(4, 1, 1, 1), torch.zeros(4))


def dataset(seed=7, **overrides):
    kwargs = {
        "base_dataset": images(),
        "digit_size": 4,
        "block_size": 6,
        "canvas_size": 12,
        "generator": torch.Generator().manual_seed(seed),
    }
    kwargs.update(overrides)
    return motion.SpatialRecall3DMotionDataset(**kwargs)


@pytest.mark.parametrize("placement", ["fixed", "random"])
@pytest.mark.parametrize("spin", [False, True])
def test_copy_target_is_present_and_readout_is_reserved(placement, spin):
    ds = dataset(placement=placement, spin=spin, readout_value=-2)
    for _ in range(4):
        canvas, target = ds[0]
        assert canvas.shape == (1, 12, 12, 12)
        assert target.shape == (1, 6, 6, 6)
        assert torch.isfinite(canvas).all()
        assert torch.all(canvas[:, 6:, 6:, 6:] == -2)
        assert torch.all(target.amax(dim=(0, 2, 3)) > 0)
        if placement == "fixed":
            torch.testing.assert_close(canvas[:, :6, :6, :6], target)
        else:
            # Locate the exact target without relying on the sampled origin.
            windows = canvas.unfold(1, 6, 1).unfold(2, 6, 1).unfold(3, 6, 1)
            assert (windows == target[:, None, None, None]).all(dim=(-1, -2, -3)).any()


def test_random_positions_never_overlap_readout():
    positions = dataset(placement="random").valid_positions
    assert positions.min() == 0
    assert positions.max() == 6
    assert not (positions > 0).all(dim=1).any()
    assert positions.shape == (7**3 - 6**3, 3)


def test_equal_seeds_reproduce_sample_sequences():
    left, right = dataset(placement="random"), dataset(placement="random")
    for i in range(4):
        for a, b in zip(left[i], right[i]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(dataset(seed=7)[0][1], dataset(seed=23)[0][1])


@pytest.mark.parametrize("length", [1, 2, 3, 6, 16])
@pytest.mark.parametrize("max_step", [0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3])
@pytest.mark.parametrize("limit", [0, 2, 20])
def test_sweeps_are_bounded_monotone_and_obey_rounded_step_cap(length, max_step, limit):
    for seed in range(10):
        offsets = motion._random_sweep_offsets(length, limit, max_step, torch.Generator().manual_seed(seed))
        steps = offsets.diff()
        assert offsets.dtype == torch.long
        assert torch.all((offsets >= 0) & (offsets <= limit))
        assert torch.all(steps >= 0) or torch.all(steps <= 0)
        if steps.numel():
            assert steps.abs().max() <= torch.ceil(torch.tensor(max_step))


def test_no_motion_or_spin_preserves_each_source_frame():
    image = images()[0][0]
    block = motion._make_motion_block(image, 4, 4, torch.Generator().manual_seed(0), 0, False)
    torch.testing.assert_close(block, image[:, None].expand(1, 4, 4, 4))


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_blank_images_remain_background(value):
    image = torch.full((1, 4, 4), value)
    block = motion._make_motion_block(image, 4, 6, torch.Generator().manual_seed(0), 2, True)
    assert torch.all(block == value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"digit_size": 0},
        {"digit_size": 7},
        {"block_size": 0},
        {"canvas_size": 11},
        {"placement": "invalid"},
        {"max_step": -1},
        {"max_step": float("nan")},
        {"max_step": float("inf")},
    ],
)
def test_invalid_geometry_and_motion_fail_early(overrides):
    with pytest.raises(ValueError):
        dataset(**overrides)


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, channels=1, num_workers=0):
        super().__init__()
        self.batch_size, self.num_workers, self.pin_memory, self.seed = 2, num_workers, False, 7
        self.input_channels, self.output_channels = channels, 10

    def setup(self, stage=None):
        source = images()
        source = TensorDataset(source.tensors[0].repeat(1, self.input_channels, 1, 1), source.tensors[1])
        if stage in (None, "fit"):
            self.train_dataset = self.val_dataset = source
        if stage in (None, "test"):
            self.test_dataset = source


def datamodule(data_type="volume", **overrides):
    kwargs = {
        "base_datamodule_cfg": LazyConfig(SyntheticDataModule)(),
        "digit_size": 4,
        "block_size": 6,
        "canvas_size": 12,
        "data_type": data_type,
    }
    kwargs.update(overrides)
    return motion.SpatialRecall3DMotionDataModule(**kwargs)


@pytest.mark.parametrize("data_type", ["volume", "sequence"])
def test_datamodule_split_loaders_and_batch_layout(data_type):
    dm = datamodule(data_type)
    with pytest.raises(RuntimeError):
        dm.train_dataloader()
    dm.setup()
    assert len({split.generator.initial_seed() for split in (dm.train_dataset, dm.val_dataset, dm.test_dataset)}) == 3
    for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
        batch = dm.on_before_batch_transfer(next(iter(loader)), 0)
        assert batch["condition"] is None
        if data_type == "volume":
            assert batch["input"].shape == (2, 12, 12, 12, 1)
            assert batch["label"].shape == (2, 6, 6, 6, 1)
        else:
            assert batch["input"].shape == (2, 12**3, 1)
            assert batch["label"].shape == (2, 6**3, 1)


def test_validate_stage_creates_validation_loader(monkeypatch):
    from experiments.datamodules import emnist

    # Exercise the documented base module's real stage dispatch without downloads.
    monkeypatch.setattr(emnist.datasets, "EMNIST", lambda *args, **kwargs: images())
    dm = datamodule(
        base_datamodule_cfg=LazyConfig(emnist.EMNISTDataModule)(
            data_dir="unused",
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            data_type="image",
            permuted=False,
            seed=7,
            use_test_as_val=True,
        )
    )
    dm.setup("validate")
    assert dm.train_dataset is None
    assert len(dm.val_dataloader()) == 2
    canvas, target = next(iter(dm.val_dataloader()))
    assert canvas.shape == (2, 1, 12, 12, 12)
    assert target.shape == (2, 1, 6, 6, 6)


def test_datamodule_validation_precedes_base_instantiation():
    with pytest.raises(ValueError):
        datamodule(canvas_size=6)
    with pytest.raises(ValueError):
        datamodule(data_type="invalid")


def test_worker_rngs_use_unique_reproducible_pytorch_seeds(monkeypatch):
    def sample(seed):
        ds = dataset()
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(dataset=ds, seed=seed))
        motion._seed_motion_worker(0)
        assert ds.generator.initial_seed() == seed
        return ds[0][1]

    torch.testing.assert_close(sample(101), sample(101))
    assert not torch.equal(sample(101), sample(102))
    dm = datamodule()
    dm.setup("fit")
    assert dm.train_dataloader().worker_init_fn is motion._seed_motion_worker


@pytest.mark.parametrize("num_workers", [0, 2])
def test_rank_motion_streams_are_distinct_and_reproducible(num_workers):
    def samples(rank):
        dm = datamodule(base_datamodule_cfg=LazyConfig(SyntheticDataModule)(num_workers=num_workers))
        dm.trainer = SimpleNamespace(global_rank=rank)
        dm.setup()
        assert dm.seed == dm._base_datamodule.seed == 7
        result = []
        for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
            # Consume real worker processes when num_workers > 0.
            result.append(torch.cat([target for _, target in loader]))
        return result

    rank0, repeated, rank1 = samples(0), samples(0), samples(1)
    for a, b, c in zip(rank0, repeated, rank1):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert not torch.equal(a, c)


@pytest.mark.parametrize("data_type", ["volume", "sequence"])
def test_rgb_channel_metadata_matches_batches_and_projection(data_type):
    dm = datamodule(data_type, base_datamodule_cfg=LazyConfig(SyntheticDataModule)(channels=3))
    # Metadata must be usable to build the network before setup.
    assert dm.input_channels == dm.output_channels == 3
    projection = torch.nn.Linear(dm.input_channels, 8)
    dm.setup("fit")
    batch = dm.on_before_batch_transfer(next(iter(dm.train_dataloader())), 0)
    assert batch["input"].shape[-1] == batch["label"].shape[-1] == 3
    assert projection(batch["input"]).shape[-1] == 8


def test_sequence_readout_gather_matches_volume_crop():
    dm = datamodule("sequence")
    S, b = dm.canvas_size, dm.block_size
    # Give every voxel a unique value to distinguish the cube from the tail.
    prediction_volume = torch.arange(S**3).reshape(1, 1, S, S, S)
    target = prediction_volume[:, :, -b:, -b:, -b:]
    batch = dm.on_before_batch_transfer((prediction_volume, target), 0)
    prediction = batch["input"]
    coords = torch.arange(S - b, S, device=prediction.device)
    d, h, w = torch.meshgrid(coords, coords, coords, indexing="ij")
    indices = (d * S * S + h * S + w).reshape(-1)
    readout = prediction.index_select(1, indices)
    torch.testing.assert_close(readout, batch["label"])
    assert not torch.equal(prediction[:, -(b**3) :], batch["label"])


def test_translation_actually_moves_the_image():
    block = motion._make_motion_block(images()[0][0], 4, 6, torch.Generator().manual_seed(7), 1, False)
    assert not torch.equal(block[:, 0], block[:, -1])
    assert torch.all(block.amax(dim=(0, 2, 3)) > 0)
