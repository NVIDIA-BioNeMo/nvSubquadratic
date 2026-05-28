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

# TODO: Add license header here


"""Tests for torch.compile compatibility with Hyena models.

Verifies that torch.compile produces identical outputs and gradients compared to eager mode.
"""

import pytest
import torch
import torch.nn as nn

import nvsubquadratic.ops.fftconv as _fftconv
from examples.mnist_classification.ccnn_4_160_hyena_rope_qknorm import get_config
from experiments.utils.cli import apply_config_overrides
from nvsubquadratic.lazy_config import instantiate


_ALLCLOSE_RTOL = 1e-4
_ALLCLOSE_ATOL = 1e-4
_GRAD_RTOL = 5e-4
_GRAD_ATOL = 5e-4


@pytest.fixture(autouse=True)
def _enable_compile_compatible_fft():
    """Enable real-arithmetic complex multiply so torch.compile/Inductor works.

    Triton cannot codegen complex64 kernels, so the default in-place
    ``fft_x.mul_(fft_kernel)`` breaks during Inductor lowering.
    """
    prev = _fftconv.COMPILE_COMPATIBLE
    _fftconv.COMPILE_COMPATIBLE = True
    yield
    _fftconv.COMPILE_COMPATIBLE = prev


@pytest.fixture
def mnist_hyena_config():
    """Load and resolve MNIST Hyena configuration."""
    config = get_config()
    return apply_config_overrides(config, [])


@pytest.fixture
def mnist_hyena_model(mnist_hyena_config, device):
    """Create MNIST Hyena model instance."""
    torch.manual_seed(42)
    datamodule = instantiate(mnist_hyena_config.dataset)
    model = instantiate(
        mnist_hyena_config.net,
        in_channels=datamodule.input_channels,
        out_channels=datamodule.output_channels,
    )
    return model.to(device)


@pytest.fixture
def sample_mnist_input(device):
    """Sample MNIST input tensor (B, H, W, C)."""
    torch.manual_seed(42)
    return torch.randn(4, 28, 28, 1, device=device)


@pytest.fixture
def sample_mnist_target(device):
    """Sample MNIST target labels."""
    torch.manual_seed(43)
    return torch.randint(0, 10, (4,), device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compile_forward_pass_equality(mnist_hyena_model, sample_mnist_input):
    """Verify forward pass outputs are identical with and without torch.compile."""
    mnist_hyena_model.eval()
    torch.manual_seed(42)

    with torch.no_grad():
        output_no_compile = mnist_hyena_model({"input": sample_mnist_input.clone(), "condition": None})

    compiled_model = torch.compile(mnist_hyena_model)
    torch.manual_seed(42)

    with torch.no_grad():
        # warmup
        _ = compiled_model({"input": sample_mnist_input.clone(), "condition": None})

        output_with_compile = compiled_model({"input": sample_mnist_input.clone(), "condition": None})

    max_diff = (output_no_compile["logits"] - output_with_compile["logits"]).abs().max().item()
    assert torch.allclose(
        output_no_compile["logits"], output_with_compile["logits"], rtol=_ALLCLOSE_RTOL, atol=_ALLCLOSE_ATOL
    ), (
        f"Forward outputs differ: max_diff={max_diff:.6e}, "
        f"shapes=({output_no_compile['logits'].shape}, {output_with_compile['logits'].shape})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.xfail(
    reason="Triton Inductor cannot codegen complex64 in backward graph (rfft/irfft gradients)",
    raises=Exception,
    strict=False,
)
def test_compile_backward_pass_equality(mnist_hyena_model, sample_mnist_input, sample_mnist_target):
    """Verify backward pass gradients are identical with and without torch.compile."""
    loss_fn = nn.CrossEntropyLoss()
    mnist_hyena_model.eval()

    # Eager mode backward pass
    torch.manual_seed(42)
    mnist_hyena_model.zero_grad()
    output_no_compile = mnist_hyena_model({"input": sample_mnist_input.clone(), "condition": None})
    loss_no_compile = loss_fn(output_no_compile["logits"], sample_mnist_target.clone())
    loss_no_compile.backward()

    grads_no_compile = {
        name: param.grad.clone() if param.grad is not None else None
        for name, param in mnist_hyena_model.named_parameters()
    }

    # Compiled mode backward pass
    torch.manual_seed(42)
    mnist_hyena_model.zero_grad()
    compiled_model = torch.compile(mnist_hyena_model)

    # warmup with backward pass
    warmup_output = compiled_model({"input": sample_mnist_input.clone(), "condition": None})
    warmup_loss = loss_fn(warmup_output["logits"], sample_mnist_target.clone())
    warmup_loss.backward()

    # actual measurement
    torch.manual_seed(42)
    mnist_hyena_model.zero_grad()
    output_with_compile = compiled_model({"input": sample_mnist_input.clone(), "condition": None})
    loss_with_compile = loss_fn(output_with_compile["logits"], sample_mnist_target.clone())
    loss_with_compile.backward()

    grads_with_compile = {
        name: param.grad.clone() if param.grad is not None else None
        for name, param in mnist_hyena_model.named_parameters()
    }

    max_output_diff = (output_no_compile["logits"] - output_with_compile["logits"]).abs().max().item()
    assert torch.allclose(
        output_no_compile["logits"], output_with_compile["logits"], rtol=_ALLCLOSE_RTOL, atol=_ALLCLOSE_ATOL
    ), f"Forward outputs differ: max_diff={max_output_diff:.6e}"

    assert torch.allclose(loss_no_compile, loss_with_compile, rtol=_ALLCLOSE_RTOL, atol=_ALLCLOSE_ATOL), (
        f"Losses differ: {loss_no_compile.item():.6f} vs {loss_with_compile.item():.6f}"
    )

    mismatched_grads = []
    for name in grads_no_compile:
        g1, g2 = grads_no_compile[name], grads_with_compile[name]
        if (g1 is None) != (g2 is None):
            mismatched_grads.append(f"{name}: gradient existence mismatch")
        elif g1 is not None and not torch.allclose(g1, g2, rtol=_GRAD_RTOL, atol=_GRAD_ATOL):
            max_diff = (g1 - g2).abs().max().item()
            mismatched_grads.append(f"{name}: max_diff={max_diff:.6e}")

    assert not mismatched_grads, f"Gradients differ for {len(mismatched_grads)} parameters:\n" + "\n".join(
        mismatched_grads[:10]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.xfail(
    reason="Triton Inductor cannot codegen complex64 in backward graph (rfft/irfft gradients)",
    raises=Exception,
    strict=False,
)
def test_compile_multiple_forward_passes(mnist_hyena_model, sample_mnist_input):
    """Verify compiled model produces consistent outputs across multiple forward passes."""
    mnist_hyena_model.eval()
    compiled_model = torch.compile(mnist_hyena_model)

    # warmup (without no_grad, so backward graph is traced → hits complex64)
    _ = compiled_model({"input": sample_mnist_input.clone(), "condition": None})

    with torch.no_grad():
        outputs = [compiled_model({"input": sample_mnist_input.clone(), "condition": None}) for _ in range(3)]

    for i, output in enumerate(outputs[1:], 1):
        max_diff = (outputs[0]["logits"] - output["logits"]).abs().max().item()
        assert torch.allclose(outputs[0]["logits"], output["logits"], rtol=_ALLCLOSE_RTOL, atol=_ALLCLOSE_ATOL), (
            f"Output mismatch at iteration {i}: max_diff={max_diff:.6e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size", [1, 4])
def test_compile_with_different_batch_sizes(mnist_hyena_config, batch_size, device):
    """Verify torch.compile works correctly with different batch sizes."""
    torch.manual_seed(42)
    datamodule = instantiate(mnist_hyena_config.dataset)
    model = instantiate(
        mnist_hyena_config.net,
        in_channels=datamodule.input_channels,
        out_channels=datamodule.output_channels,
    )
    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(batch_size, 28, 28, 1, device=device)

    with torch.no_grad():
        output_no_compile = model({"input": input_tensor.clone(), "condition": None})

    compiled_model = torch.compile(model)

    with torch.no_grad():
        # warmup
        _ = compiled_model({"input": input_tensor.clone(), "condition": None})

        output_with_compile = compiled_model({"input": input_tensor.clone(), "condition": None})

    max_diff = (output_no_compile["logits"] - output_with_compile["logits"]).abs().max().item()
    assert torch.allclose(
        output_no_compile["logits"], output_with_compile["logits"], rtol=_ALLCLOSE_RTOL, atol=_ALLCLOSE_ATOL
    ), f"Outputs differ for batch_size={batch_size}: max_diff={max_diff:.6e}"
