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

"""Unit tests for the DiffusionWrapper with and without classifier-free guidance."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from experiments.default_cfg import DiffusionConfig, DiffusionExperimentConfig
from experiments.lightning_wrappers.diffusion_wrapper import DiffusionWrapper


class RecordingDenoiser(torch.nn.Module):
    """Minimal denoiser that records every conditioning vector it receives."""

    def __init__(self, hidden_dim: int, channels: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.calls: list[torch.Tensor] = []
        # Register a parameter so Lightning can infer the device even though the model is almost empty.
        self.offset = torch.nn.Parameter(torch.zeros(1))

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        condition = data["condition"]
        # Store a detached copy so tests can inspect the unconditional vs. conditional branches.
        self.calls.append(condition.detach().cpu())
        logits = torch.tanh(data["input"] + self.offset)
        return {"logits": logits}


def _make_wrapper(
    *,
    num_classes: int | None,
    use_cfg: bool,
    guidance_scale: float = 1.0,
    condition_dropout_prob: float = 0.0,
    num_inference_steps: int = 2,
    diffusion_overrides: dict[str, Any] | None = None,
) -> DiffusionWrapper:
    """Helper to instantiate a DiffusionWrapper with a lightweight recording network."""

    diff_cfg = DiffusionConfig()
    diff_cfg.num_train_timesteps = 8
    diff_cfg.num_inference_steps = num_inference_steps
    diff_cfg.num_samples = 2
    diff_cfg.log_samples = False
    diff_cfg.ema_enabled = False
    diff_cfg.use_classifier_free_guidance = use_cfg
    diff_cfg.guidance_scale = guidance_scale
    diff_cfg.condition_dropout_prob = condition_dropout_prob
    diff_cfg.num_classes = num_classes
    if diffusion_overrides:
        for key, value in diffusion_overrides.items():
            setattr(diff_cfg, key, value)

    exp_cfg = DiffusionExperimentConfig(diffusion=diff_cfg)
    exp_cfg.train.iterations = 10
    exp_cfg.train.batch_size = 1
    exp_cfg.train.grad_clip = 0.0

    network = RecordingDenoiser(hidden_dim=8, channels=3)
    wrapper = DiffusionWrapper(network=network, cfg=exp_cfg)
    wrapper.to(torch.device("cpu"))
    return wrapper


def test_class_conditioning_requires_labels() -> None:
    """Ensure the wrapper refuses to run class-conditioned diffusion without labels."""
    wrapper = _make_wrapper(num_classes=10, use_cfg=True, guidance_scale=3.0)
    images = torch.randn(2, 4, 4, 3)
    batch = {"input": images}
    with pytest.raises(RuntimeError, match="Class conditioning requires datamodule"):
        wrapper._shared_step(batch)  # type: ignore[arg-type]


def test_cfg_sampling_invokes_both_branches() -> None:
    """Classifier-free guidance should evaluate unconditional and conditional paths per step."""
    torch.manual_seed(0)
    wrapper = _make_wrapper(num_classes=4, use_cfg=True, guidance_scale=2.5, num_inference_steps=2)
    wrapper.example_input_shape = torch.Size((4, 4, 3))

    samples = wrapper.sample(num_samples=1, labels=torch.tensor([1]))
    assert samples.shape == (1, 4, 4, 3)
    # Two timesteps * two passes (unconditional + conditional).
    assert len(wrapper.network.calls) == 4  # type: ignore[attr-defined]
    unconditional_condition = wrapper.network.calls[0]  # type: ignore[attr-defined]
    conditional_condition = wrapper.network.calls[1]  # type: ignore[attr-defined]
    assert not torch.allclose(unconditional_condition, conditional_condition)


def test_sampling_without_guidance_uses_single_branch() -> None:
    """When guidance is disabled the denoiser should run exactly once per timestep."""
    wrapper = _make_wrapper(num_classes=4, use_cfg=False, num_inference_steps=3)
    wrapper.example_input_shape = torch.Size((4, 4, 3))
    wrapper.network.calls.clear()  # type: ignore[attr-defined]

    samples = wrapper.sample(num_samples=1, labels=torch.tensor([2]))
    assert samples.shape == (1, 4, 4, 3)
    # Three timesteps, single pass each.
    assert len(wrapper.network.calls) == 3  # type: ignore[attr-defined]


