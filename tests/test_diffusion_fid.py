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

from types import SimpleNamespace

import pytest
import torch

from experiments.default_cfg import DiffusionConfig, DiffusionExperimentConfig
from experiments.lightning_wrappers.diffusion_wrapper import DiffusionWrapper
from nvsubquadratic.metrics.cleanfid import compute_folder_fid


class _IdentityBackbone(torch.nn.Module):
    hidden_dim = 4

    def forward(self, input_and_condition):
        return {"logits": input_and_condition["input"]}


def _make_cfg() -> DiffusionExperimentConfig:
    cfg = DiffusionExperimentConfig()
    cfg.diffusion = DiffusionConfig()
    cfg.diffusion.num_train_timesteps = 4
    cfg.diffusion.num_inference_steps = 2
    cfg.diffusion.log_samples = False
    cfg.diffusion.num_samples = 2
    cfg.diffusion.use_classifier_free_guidance = False
    cfg.diffusion.num_classes = None
    cfg.diffusion.fid_num_inference_steps = 1
    cfg.optimizer = SimpleNamespace(weight_decay=0.0)
    return cfg


def test_diffusion_wrapper_validation_step():
    cfg = _make_cfg()
    module = DiffusionWrapper(network=_IdentityBackbone(), cfg=cfg)
    module.log = lambda *args, **kwargs: None

    batch = {"input": torch.zeros(2, 4, 4, 1)}
    loss = module.validation_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)


def test_compute_folder_fid_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_folder_fid(tmp_path / "missing", dataset_name="imagenet2012", dataset_resolution=64)
