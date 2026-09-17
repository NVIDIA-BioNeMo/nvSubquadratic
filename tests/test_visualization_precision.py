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

"""Visualization forwards must use the same precision as validation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib


matplotlib.use("Agg")
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins.precision import MixedPrecision, Precision

from experiments.callbacks.image_grid_val_visualization import (
    ValidationImageGridCallback,
    ValidationVolumeGridCallback,
)
from experiments.callbacks.sequence_visualization_1d import Sequence1DVisualizationCallback


class _Projection(pl.LightningModule):
    def __init__(self, layout):
        super().__init__()
        self.layout = layout
        self.projection = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.projection.weight.fill_(0.314159)
            self.projection.bias.fill_(0.123456)
        self.calls = []

    def forward(self, batch):
        x = batch["input"]
        if self.layout == "volume":
            x = x[:, -1]
        prediction = self.projection(x)
        self.calls.append((torch.is_autocast_enabled("cpu"), prediction.detach().clone()))
        return {"logits": prediction}


@pytest.mark.parametrize("layout", ["image", "volume", "sequence"])
@pytest.mark.parametrize("mixed", [False, True], ids=["fp32", "bf16"])
def test_visualization_matches_validation_precision_and_renders(layout, mixed):
    plugin = MixedPrecision("bf16-mixed", device="cpu") if mixed else Precision()
    model = _Projection(layout)
    if layout == "volume":
        x = torch.linspace(0, 1, 2 * 8 * 8).reshape(1, 2, 8, 8, 1)
        callback = ValidationVolumeGridCallback(num_samples=1, target_size=8, denormalize=False)
        log = callback._log_volume_grid
    elif layout == "sequence":
        x = torch.linspace(0, 1, 64).reshape(1, 64, 1)
        callback = Sequence1DVisualizationCallback(num_samples=1, target_size=8, denormalize=False)
        log = callback._log_visualization
    else:
        x = torch.linspace(0, 1, 64).reshape(1, 8, 8, 1)
        callback = ValidationImageGridCallback(num_samples=1, denormalize=False)
        log = callback._log_image_grid

    with torch.no_grad(), plugin.forward_context():
        reference = model({"input": x})["logits"].float()
    model.calls.clear()
    logger = MagicMock(spec=["log_image"])
    trainer = SimpleNamespace(
        precision_plugin=plugin,
        val_dataloaders=[[{"input": x, "label": reference}]],
        datamodule=SimpleNamespace(),
        logger=logger,
        global_step=0,
    )
    # Run the complete rendering path, including BF16-to-NumPy conversion.
    assert not torch.is_autocast_enabled("cpu")
    log(trainer, model, event_idx=0)
    assert not torch.is_autocast_enabled("cpu")
    assert len(model.calls) == 1
    autocast_enabled, prediction = model.calls[0]
    assert autocast_enabled == mixed
    assert prediction.dtype == (torch.bfloat16 if mixed else torch.float32)
    torch.testing.assert_close(prediction.float(), reference, rtol=0, atol=0)
    logger.log_image.assert_called_once()
