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

import sys
import types
from pathlib import Path

import torch

from nvsubquadratic.networks.huggingface_diffusers import (
    DiffusersDiTWrapper,
    DiffusersUVitWrapper,
    HuggingFaceDiTConfig,
    HuggingFaceUVitConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _DummyWrapper:
    """Minimal stub mimicking the Lightning diffusion wrapper interface."""

    def __init__(self):
        def _condition(_, timesteps: torch.LongTensor) -> torch.Tensor:
            # Mirror the Lightning behaviour by returning an embedding tensor; we only care about
            # forwarding the timesteps to registered callbacks.
            return torch.zeros((timesteps.shape[0], 4), dtype=torch.float32, device=timesteps.device)

        self._condition_from_timesteps = types.MethodType(_condition, self)


def _tiny_config() -> HuggingFaceDiTConfig:
    return HuggingFaceDiTConfig(
        sample_size=8,
        patch_size=2,
        in_channels=1,
        out_channels=1,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        num_embeds_ada_norm=32,
        activation_fn="gelu-approximate",
        norm_type="ada_norm_zero",
    )


def test_hf_wrapper_tracks_timesteps():
    model = DiffusersDiTWrapper(_tiny_config(), in_channels=1, out_channels=1)
    dummy_wrapper = _DummyWrapper()

    model.hf_register_diffusion_wrapper(dummy_wrapper)

    timesteps = torch.tensor([3, 7], dtype=torch.long)
    _ = dummy_wrapper._condition_from_timesteps(timesteps)

    assert model._latest_timesteps is not None
    assert torch.equal(model._latest_timesteps, timesteps)


def test_hf_wrapper_forward_matches_input_shape():
    model = DiffusersDiTWrapper(_tiny_config(), in_channels=1, out_channels=1)
    dummy_wrapper = _DummyWrapper()
    model.hf_register_diffusion_wrapper(dummy_wrapper)

    batch_size = 2
    inputs = torch.randn(
        batch_size, model.hf_config.sample_size, model.hf_config.sample_size, model.hf_config.in_channels
    )
    dummy_condition = torch.zeros(batch_size, model.hidden_dim)

    timesteps = torch.tensor([5, 11], dtype=torch.long)
    model._latest_timesteps = timesteps

    outputs = model({"input": inputs, "condition": dummy_condition})

    assert "logits" in outputs
    assert outputs["logits"].shape == inputs.shape


def test_hf_wrapper_accepts_channel_overrides():
    cfg = _tiny_config()
    cfg.in_channels = 2
    model = DiffusersDiTWrapper(cfg, in_channels=3, out_channels=5)
    assert model.hf_config.in_channels == 3
    assert model.hf_config.out_channels == 5


def test_extra_repr_handles_dict_config():
    cfg = _tiny_config()
    repr_str = repr(DiffusersDiTWrapper(cfg, in_channels=1, out_channels=1))
    assert "DiffusersDiTWrapper" in repr_str


def test_specialised_config_variants():
    dit_cfg = HuggingFaceDiTConfig()
    uvit_cfg = HuggingFaceUVitConfig()
    assert dit_cfg.num_layers > 0
    assert uvit_cfg.hidden_size > 0


def test_uvit_wrapper_builds_with_defaults():
    cfg = HuggingFaceUVitConfig(
        sample_size=4,
        in_channels=1,
        out_channels=1,
        hidden_size=64,
        cond_embed_dim=16,
        encoder_hidden_size=16,
        block_out_channels=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        layer_norm_eps=1e-5,
        micro_cond_encode_dim=8,
        micro_cond_embed_dim=16,
        codebook_size=32,
        vocab_size=33,
    )

    model = DiffusersUVitWrapper(cfg)
    dummy_wrapper = _DummyWrapper()
    model.hf_register_diffusion_wrapper(dummy_wrapper)

    timesteps = torch.tensor([1, 2], dtype=torch.long)
    dummy_wrapper._condition_from_timesteps(timesteps)

    inputs = torch.randn(2, cfg.sample_size, cfg.sample_size, cfg.in_channels)
    cond = torch.zeros(2, cfg.cond_embed_dim)
    outputs = model({"input": inputs, "condition": cond})

    assert "logits" in outputs
    assert outputs["logits"].shape[0] == inputs.shape[0]
