# TODO: Add license header here


"""Tests for torch.compile compatibility with Hyena models.

Verifies that torch.compile produces identical outputs and gradients compared to eager mode.
"""

import pytest
import torch
import torch.nn as nn

from examples.mnist_classification.ccnn_4_160_hyena_rope_qknorm import get_config
from experiments.utils.cli import apply_config_overrides
from nvsubquadratic.lazy_config import instantiate


_ALLCLOSE_RTOL = 1e-4
_ALLCLOSE_ATOL = 1e-5


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
        output_no_compile = mnist_hyena_model(sample_mnist_input.clone())

    compiled_model = torch.compile(mnist_hyena_model)
    torch.manual_seed(42)

    with torch.no_grad():
        output_with_compile = compiled_model(sample_mnist_input.clone())

    max_diff = (output_no_compile - output_with_compile).abs().max().item()
    assert torch.allclose(output_no_compile, output_with_compile, rtol=_ALLCLOSE_RTOL, atol=_ALLCLOSE_ATOL), (
        f"Forward outputs differ: max_diff={max_diff:.6e}, "
        f"shapes=({output_no_compile.shape}, {output_with_compile.shape})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compile_backward_pass_equality(mnist_hyena_model, sample_mnist_input, sample_mnist_target, device):
    """Verify backward pass gradients are identical with and without torch.compile."""
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(42)

    mnist_hyena_model.eval()
    output_no_compile = mnist_hyena_model(sample_mnist_input.clone())
    loss_no_compile = loss_fn(output_no_compile, sample_mnist_target.clone())
    loss_no_compile.backward()

    grads_no_compile = {
        name: param.grad.clone() if param.grad is not None else None
        for name, param in mnist_hyena_model.named_parameters()
    }

    torch.manual_seed(42)
    config = apply_config_overrides(get_config(), [])
    datamodule = instantiate(config.dataset)
    model_with_compile = instantiate(
        config.net,
        in_channels=datamodule.input_channels,
        out_channels=datamodule.output_channels,
    )
    model_with_compile.load_state_dict(mnist_hyena_model.state_dict())
    model_with_compile = model_with_compile.to(device)
    model_with_compile.eval()

    compiled_model = torch.compile(model_with_compile)
    output_with_compile = compiled_model(sample_mnist_input.clone())
    loss_with_compile = loss_fn(output_with_compile, sample_mnist_target.clone())
    loss_with_compile.backward()

    grads_with_compile = {
        name: param.grad.clone() if param.grad is not None else None
        for name, param in model_with_compile.named_parameters()
    }

    max_output_diff = (output_no_compile - output_with_compile).abs().max().item()
    assert torch.allclose(output_no_compile, output_with_compile, rtol=_ALLCLOSE_RTOL, atol=_ALLCLOSE_ATOL), (
        f"Forward outputs differ: max_diff={max_output_diff:.6e}"
    )

    assert torch.allclose(loss_no_compile, loss_with_compile, rtol=_ALLCLOSE_RTOL, atol=_ALLCLOSE_ATOL), (
        f"Losses differ: {loss_no_compile.item():.6f} vs {loss_with_compile.item():.6f}"
    )

    mismatched_grads = []
    for name in grads_no_compile:
        g1, g2 = grads_no_compile[name], grads_with_compile[name]
        if (g1 is None) != (g2 is None):
            mismatched_grads.append(f"{name}: gradient existence mismatch")
        elif g1 is not None and not torch.allclose(g1, g2, rtol=_ALLCLOSE_RTOL, atol=_ALLCLOSE_ATOL):
            max_diff = (g1 - g2).abs().max().item()
            mismatched_grads.append(f"{name}: max_diff={max_diff:.6e}")

    assert not mismatched_grads, (
        f"Gradients differ for {len(mismatched_grads)} parameters:\n" + "\n".join(mismatched_grads[:10])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compile_multiple_forward_passes(mnist_hyena_model, sample_mnist_input):
    """Verify compiled model produces consistent outputs across multiple forward passes."""
    mnist_hyena_model.eval()
    compiled_model = torch.compile(mnist_hyena_model)

    with torch.no_grad():
        outputs = [compiled_model(sample_mnist_input.clone()) for _ in range(3)]

    for i, output in enumerate(outputs[1:], 1):
        max_diff = (outputs[0] - output).abs().max().item()
        assert torch.allclose(outputs[0], output, rtol=_ALLCLOSE_RTOL, atol=_ALLCLOSE_ATOL), (
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
        output_no_compile = model(input_tensor.clone())
        output_with_compile = torch.compile(model)(input_tensor.clone())

    max_diff = (output_no_compile - output_with_compile).abs().max().item()
    assert torch.allclose(output_no_compile, output_with_compile, rtol=_ALLCLOSE_RTOL, atol=_ALLCLOSE_ATOL), (
        f"Outputs differ for batch_size={batch_size}: max_diff={max_diff:.6e}"
    )
