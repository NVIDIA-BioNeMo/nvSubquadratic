# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import torch
from experiments.datamodules.imagenet import ImageNetDataModule
from PIL import Image

from experiments.datamodules.mnist import MNISTDataModule


def _mnist_datamodule(task: str) -> MNISTDataModule:
    return MNISTDataModule(
        data_dir=".data/mnist",
        batch_size=1,
        data_type="image",
        num_workers=0,
        pin_memory=False,
        use_deterministic_worker_init=False,
        seed=0,
        task=task,
    )


def test_mnist_generation_transform_scales_to_minus_one_one():
    dm = _mnist_datamodule(task="generation")
    torch.manual_seed(0)
    image = Image.fromarray(np.full((28, 28), 128, dtype=np.uint8), mode="L")

    transformed = dm.transform(image)
    assert transformed.shape == (1, 28, 28)
    assert transformed.min().item() >= -1.0001
    assert transformed.max().item() <= 1.0001

    baseline = torch.full((28, 28), 128.0) / 255.0
    baseline = (baseline - 0.5) / 0.5
    assert not torch.allclose(transformed.squeeze(), baseline, atol=1e-3)


def test_imagenet_transform_keeps_dynamic_range(tmp_path):
    dm = ImageNetDataModule(
        data_dir=str(tmp_path),
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        seed=0,
        image_size=32,
        final_image_size=32,
        center_crop=True,
        drop_labels=True,
        task="generation",
    )
    transform = dm._build_transform(train=False)

    pattern = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    image = Image.fromarray(pattern, mode="RGB")
    torch.manual_seed(0)
    tensor = transform(image)

    assert tensor.shape == (3, 32, 32)
    assert tensor.min().item() >= -1.0001
    assert tensor.max().item() <= 1.0001
