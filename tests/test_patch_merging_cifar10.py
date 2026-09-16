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


"""Download-free coverage of the two recovered CIFAR-10 recipe adapters."""

import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset


@pytest.mark.parametrize("source", ["torchvision", "huggingface"])
def test_cifar_validation_batch_contract(monkeypatch, source):
    if source == "torchvision":
        from experiments.datamodules import cifar10 as module

        samples = TensorDataset(torch.randn(4, 3, 32, 32), torch.arange(4))
        monkeypatch.setattr(module.datasets, "CIFAR10", lambda *args, **kwargs: samples)
        dm = module.CIFAR10DataModule(batch_size=2, num_workers=0, pin_memory=False)
        dm.setup("validate")
        batch = next(iter(dm.val_dataloader()))
    else:
        from experiments.datamodules import cifar10_hf as module

        samples = [{"img": Image.new("RGB", (32, 32)), "label": i} for i in range(4)]
        monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: samples)
        dm = module.CIFAR10DataModule(
            data_dir="unused", batch_size=2, num_workers=0, pin_memory=False, final_image_size=32
        )
        dm.setup("validate")
        batch = dm.on_before_batch_transfer(next(iter(dm.val_dataloader())), 0)
    assert batch["input"].shape[1:] == (32, 32, 3)
    assert batch["label"].dtype == torch.long
    assert batch["condition"] is None
    assert dm.output_channels == 10


def test_hyena_pair_has_matching_optimizer_budget():
    from examples.vit5_imagenet.v6_hierarchical.cifar10 import hyena_flat, hyena_hier

    configs = [recipe.get_config() for recipe in (hyena_flat, hyena_hier)]
    for cfg in configs:
        assert cfg.dataset.batch_size * cfg.train.accumulate_grad_steps == 256
        assert cfg.train.iterations == 19_500
        assert cfg.scheduler.warmup_iterations_percentage * cfg.train.iterations == 975


@pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.3])
def test_cifar_mixup_targets_use_explicit_smoothing(monkeypatch, smoothing):
    from experiments.datamodules import cifar10 as module

    # Identical classes make the expected target independent of the random mix ratio.
    samples = TensorDataset(torch.randn(4, 3, 32, 32), torch.full((4,), 3))
    monkeypatch.setattr(module.datasets, "CIFAR10", lambda *args, **kwargs: samples)
    dm = module.CIFAR10DataModule(batch_size=4, num_workers=0, pin_memory=False, mixup=0.8, label_smoothing=smoothing)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    expected = torch.full((4, 10), smoothing / 10)
    expected[:, 3] += 1 - smoothing
    torch.testing.assert_close(batch["label"], expected)
    assert next(iter(dm.val_dataloader()))["label"].dtype == torch.long


@pytest.mark.parametrize("smoothing", [-0.1, 1.1])
def test_cifar_rejects_invalid_smoothing(smoothing):
    from experiments.datamodules.cifar10 import CIFAR10DataModule

    with pytest.raises(ValueError, match="label_smoothing"):
        CIFAR10DataModule(label_smoothing=smoothing)


@pytest.mark.parametrize(
    "recipe", ["flat_p4", "flat_p8", "flat_p16", "hier_p4", "hier_p8", "hier_p16", "hyena_flat", "hyena_hier"]
)
def test_cifar_recipes_record_smoothing(recipe):
    from importlib import import_module

    from experiments.utils.cli import config_to_dict

    module = import_module(f"examples.vit5_imagenet.v6_hierarchical.cifar10.{recipe}")
    cfg = config_to_dict(module.get_config())
    assert cfg["dataset"]["label_smoothing"] == 0.1
