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

"""CPU checks for the dense, prefix-free vision encoder retrofit asset.

Run from the repository root with:
    python -m pytest -o addopts='' skills/hyenand-retrofit/evals/test_vision_encoder.py
"""

import importlib.util
import io
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nvsubquadratic.modules.film import KernelFiLMGenerator
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer


_ASSET_PATH = Path(__file__).resolve().parents[1] / "assets" / "vision_encoder.py"
_SPEC = importlib.util.spec_from_file_location("hyenand_retrofit_vision_encoder", _ASSET_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ASSET = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ASSET
_SPEC.loader.exec_module(_ASSET)
Hyena2DTokenMixer = _ASSET.Hyena2DTokenMixer

_CHANNELS = 8
_GRID = (3, 5)
_MAX_GRID = (8, 8)


@pytest.fixture(autouse=True)
def _reproducible_randomness():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield


def _make_layer(film_conditioning="none"):
    return Hyena2DTokenMixer(
        _CHANNELS,
        max_grid_hw=_MAX_GRID,
        film_conditioning=film_conditioning,
        fft_backend="torch_fft",
        kernel_hidden_dim=8,
    )


def _only_module(model, module_type):
    matches = [module for module in model.modules() if isinstance(module, module_type)]
    assert len(matches) == 1
    return matches[0]


def _activate_film(model):
    # Identity initialization intentionally hides conditioning sensitivity until
    # the FiLM output head changes. Perturb that head, not the spatial inputs.
    generator = _only_module(model, KernelFiLMGenerator)
    with torch.no_grad():
        generator.mlp[-1].weight.normal_(std=0.1)
    return generator


def test_rectangular_grid_reaches_the_spatial_mixer_without_new_parameters():
    layer = _make_layer("patch_mean")
    parameters_before = {name: id(value) for name, value in layer.named_parameters()}
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-3)
    optimizer_ids = {id(value) for group in optimizer.param_groups for value in group["params"]}
    seen_shapes = []
    inner = _only_module(layer, QKVSequenceMixer)
    handle = inner.register_forward_pre_hook(lambda _module, args: seen_shapes.append(tuple(args[0].shape)))
    try:
        for height, width in (_GRID, (5, 3), (4, 7)):
            x = torch.randn(2, height * width, _CHANNELS)
            y = layer(x, grid_hw=(height, width))
            assert isinstance(y, torch.Tensor)
            assert y.shape == x.shape
            assert torch.isfinite(y).all()
    finally:
        handle.remove()
    assert seen_shapes == [(2, 3, 5, _CHANNELS), (2, 5, 3, _CHANNELS), (2, 4, 7, _CHANNELS)]
    assert {name: id(value) for name, value in layer.named_parameters()} == parameters_before
    assert {id(value) for value in layer.parameters()} == optimizer_ids


@pytest.mark.parametrize("film_conditioning", ["none", "patch_mean"])
def test_samples_are_independent_with_and_without_active_film(film_conditioning):
    layer = _make_layer(film_conditioning).eval()
    if film_conditioning == "patch_mean":
        _activate_film(layer)
    x = torch.randn(2, 15, _CHANNELS)
    x[1] += 2.0
    with torch.no_grad():
        together = layer(x, grid_hw=_GRID)
        separately = torch.cat([layer(sample, grid_hw=_GRID) for sample in x.split(1)], dim=0)
    torch.testing.assert_close(together, separately, rtol=2e-5, atol=2e-5)


def test_film_is_wired_and_identity_initialization_can_become_sample_dependent():
    plain = _make_layer()
    assert not any(isinstance(module, KernelFiLMGenerator) for module in plain.modules())
    layer = _make_layer("patch_mean").eval()
    generator = _only_module(layer, KernelFiLMGenerator)
    kernel = _only_module(layer, SIRENKernelND)
    x = torch.randn(2, 15, _CHANNELS)
    x[1] += 2 * torch.linspace(-1, 1, _CHANNELS)
    expected_conditioning = x.mean(dim=1)
    for gamma, beta in generator(expected_conditioning):
        torch.testing.assert_close(gamma, torch.ones_like(gamma), rtol=0, atol=0)
        torch.testing.assert_close(beta, torch.zeros_like(beta), rtol=0, atol=0)

    seen_conditions = []
    seen_kernels = []
    condition_handle = generator.register_forward_pre_hook(
        lambda _module, args: seen_conditions.append(args[0].detach().clone())
    )
    kernel_handle = kernel.register_forward_hook(
        lambda _module, _args, output: seen_kernels.append(output[0].detach().clone())
    )
    try:
        with torch.no_grad():
            before = layer(x, grid_hw=_GRID)
            _activate_film(layer)
            after = layer(x, grid_hw=_GRID)
    finally:
        condition_handle.remove()
        kernel_handle.remove()
    assert len(seen_conditions) == len(seen_kernels) == 2
    for conditioning in seen_conditions:
        torch.testing.assert_close(conditioning, expected_conditioning)
    assert seen_kernels[0].shape[0] == x.shape[0]
    torch.testing.assert_close(seen_kernels[0][0], seen_kernels[0][1], rtol=0, atol=0)
    assert not torch.allclose(seen_kernels[1][0], seen_kernels[1][1])
    assert not torch.allclose(before, after)


@pytest.mark.parametrize("film_conditioning", ["none", "patch_mean"])
def test_backward_grid_history_and_trained_checkpoint_round_trip(film_conditioning):
    layer = _make_layer(film_conditioning)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-2)
    x = torch.randn(2, 15, _CHANNELS, requires_grad=True)
    target = torch.randn_like(x)
    loss = F.mse_loss(layer(x, grid_hw=_GRID), target)
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    for name, parameter in layer.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"Disconnected trainable parameter: {name}"
            assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"
    optimizer.step()
    if film_conditioning == "patch_mean":
        generator = _only_module(layer, KernelFiLMGenerator)
        assert torch.count_nonzero(generator.mlp[-1].weight) > 0

    layer.eval()
    with torch.no_grad():
        expected = layer(x.detach(), grid_hw=_GRID).clone()
        # A cache may update for other shapes, but must not change the function
        # on an already-seen input or become hidden checkpoint state.
        for height, width in ((5, 2), (2, 7), _MAX_GRID):
            layer(torch.randn(2, height * width, _CHANNELS), grid_hw=(height, width))
            torch.testing.assert_close(layer(x.detach(), grid_hw=_GRID), expected, rtol=0, atol=0)

    checkpoint = io.BytesIO()
    torch.save(layer.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = _make_layer(film_conditioning)
    result = restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x.detach(), grid_hw=_GRID), expected, rtol=0, atol=0)


@pytest.mark.parametrize("film_conditioning", ["none", "patch_mean"])
def test_cpu_bfloat16_autocast_preserves_shortcut_parameter_and_gradients(film_conditioning):
    layer = _make_layer(film_conditioning)
    shortcut = layer.mixer.mixer.global_conv.shortcut
    x = torch.randn(2, 15, _CHANNELS, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = layer(x, grid_hw=_GRID)
        loss = F.mse_loss(output.float(), torch.randn_like(x))
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    loss.backward()
    assert layer.mixer.mixer.global_conv.shortcut is shortcut
    assert shortcut.dtype == torch.float32
    assert shortcut.grad is not None and torch.isfinite(shortcut.grad).all()
    assert torch.count_nonzero(shortcut.grad) > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0


def test_projection_transfer_preserves_weights_and_parameter_identity():
    layer = _make_layer()
    qkv = nn.Linear(_CHANNELS, 3 * _CHANNELS)
    out_proj = nn.Linear(_CHANNELS, _CHANNELS)
    before_ids = {name: id(value) for name, value in layer.named_parameters()}
    layer.copy_projections_from(qkv, out_proj)
    mixer = _only_module(layer, QKVSequenceMixer)
    for source, destination in ((qkv, mixer.qkv_proj), (out_proj, mixer.out_proj)):
        torch.testing.assert_close(destination.weight, source.weight, rtol=0, atol=0)
        torch.testing.assert_close(destination.bias, source.bias, rtol=0, atol=0)
    assert {name: id(value) for name, value in layer.named_parameters()} == before_ids
    with pytest.raises((ValueError, TypeError)):
        layer.copy_projections_from(nn.Linear(_CHANNELS, 2 * _CHANNELS), out_proj)
    with pytest.raises((ValueError, TypeError)):
        layer.copy_projections_from(nn.Linear(_CHANNELS, 3 * _CHANNELS, bias=False), out_proj)
    with pytest.raises((ValueError, TypeError)):
        layer.copy_projections_from(qkv.double(), out_proj)
    # A bad second projection must not leave the first projection partly updated.
    saved_qkv = mixer.qkv_proj.weight.detach().clone()
    with pytest.raises((ValueError, TypeError)):
        layer.copy_projections_from(nn.Linear(_CHANNELS, 3 * _CHANNELS), nn.Linear(_CHANNELS, _CHANNELS + 1))
    torch.testing.assert_close(mixer.qkv_proj.weight, saved_qkv, rtol=0, atol=0)


def test_invalid_geometry_prefix_tokens_and_attention_options_are_rejected():
    with pytest.raises(TypeError):
        Hyena2DTokenMixer(_CHANNELS)  # Bounds must be an explicit architecture choice.
    for max_grid_hw in ((1, 8), (8, 1), (8,), (8, 8.0)):
        with pytest.raises((ValueError, TypeError)):
            Hyena2DTokenMixer(_CHANNELS, max_grid_hw=max_grid_hw)
    layer = _make_layer()
    x = torch.randn(2, 15, _CHANNELS)
    for grid_hw in ((0, 5), (3,), (3, 5, 1), (3.0, 5), (3, 6)):
        with pytest.raises((ValueError, TypeError)):
            layer(x, grid_hw=grid_hw)
    for height, width in ((9, 3), (3, 9)):
        with pytest.raises(ValueError, match="exceeds max_grid_hw"):
            layer(torch.randn(2, height * width, _CHANNELS), grid_hw=(height, width))
    with pytest.raises((ValueError, TypeError)):
        layer(torch.randn(2, 16, _CHANNELS), grid_hw=_GRID)  # A CLS token is not a grid site.
    with pytest.raises((ValueError, TypeError)):
        layer(torch.randn(2, 15, _CHANNELS + 1), grid_hw=_GRID)
    with pytest.raises(TypeError):
        layer(x, grid_hw=_GRID, attn_mask=torch.zeros(15, 15))
    with pytest.raises(TypeError):
        layer(x, grid_hw=_GRID, is_causal=True)


class _DenseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(_CHANNELS, 3 * _CHANNELS)
        self.out_proj = nn.Linear(_CHANNELS, _CHANNELS)

    def forward(self, x, *, grid_hw):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        mixed = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1))
        return self.out_proj(mixed.squeeze(1))


class _DenseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixers = nn.ModuleList([_DenseAttention(), _DenseAttention()])
        self.norms = nn.ModuleList([nn.LayerNorm(_CHANNELS), nn.LayerNorm(_CHANNELS)])

    def forward(self, x, *, grid_hw):
        for norm, mixer in zip(self.norms, self.mixers):
            x = x + mixer(norm(x), grid_hw=grid_hw)
        return x


def test_dense_encoder_really_replaces_one_attention_layer():
    encoder = _DenseEncoder()
    original = encoder.mixers[1]
    replacement = _make_layer("patch_mean")
    replacement.copy_projections_from(original.qkv, original.out_proj)
    encoder.mixers[1] = replacement
    assert sum(isinstance(module, Hyena2DTokenMixer) for module in encoder.modules()) == 1
    assert sum(isinstance(module, _DenseAttention) for module in encoder.modules()) == 1
    x = torch.randn(2, 15, _CHANNELS)
    output = encoder(x, grid_hw=_GRID)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
