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

"""Tests for torch.compile compatibility with Hyena models.

Verifies that torch.compile produces identical outputs and gradients compared to eager mode.
"""

import pytest
import torch
import torch.nn as nn

import nvsubquadratic.ops.fftconv as _fftconv
from experiments.datamodules.mnist import MNISTDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from experiments.utils.cli import apply_config_overrides
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig, instantiate
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.classification_resnet import ClassificationResNet
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


_ALLCLOSE_RTOL = 1e-4
_ALLCLOSE_ATOL = 1e-4
_GRAD_RTOL = 5e-4
_GRAD_ATOL = 5e-4


def _build_mnist_hyena_config() -> ExperimentConfig:
    """Build a small MNIST-shaped Hyena classification config for the compile tests.

    Relocated verbatim from the former ``examples/mnist_classification`` recipe so
    this test is self-contained and does not depend on an example config. The
    model (4 blocks, hidden dim 160, 2D circular Hyena) is intentionally tiny —
    it is a ``torch.compile`` compatibility vehicle, not a trained model.
    """
    config = ExperimentConfig()

    config.dataset = LazyConfig(MNISTDataModule)(
        data_dir=".data/mnist",
        data_type="image",
        batch_size=128,
        num_workers=0,
        pin_memory=False,
        use_deterministic_worker_init=True,
        seed=config.seed,
        task="classification",
    )

    config.net = LazyConfig(ClassificationResNet)(
        in_channels=1,
        out_channels=10,
        num_blocks=4,
        hidden_dim=160,
        data_dim=2,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.in_channels}", out_features="${net.hidden_dim}"),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.hidden_dim}", out_features="${net.out_channels}"),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim="${net.data_dim}",
                        hidden_dim="${net.hidden_dim}",
                        kernel_cfg=LazyConfig(SIRENKernelND)(
                            data_dim="${net.data_dim}",
                            out_dim="${net.hidden_dim}",
                            mlp_hidden_dim=32,
                            num_layers=3,
                            embedding_dim=32,
                            omega_0=100.0,
                            L_cache=32,
                            use_bias=True,
                            hidden_omega_0=1.0,
                        ),
                        mask_cfg=LazyConfig(GaussianModulationND)(
                            data_dim="${net.data_dim}",
                            num_channels="${net.hidden_dim}",
                            min_attenuation_at_step=0.1,
                            max_attenuation_at_limit=0.95,
                            init_extent=1.0,
                            parametrization="direct",
                        ),
                        grid_type="single",
                        fft_padding="circular",
                    ),
                    short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                        in_channels="3 * ${net.hidden_dim}",
                        out_channels="3 * ${net.hidden_dim}",
                        kernel_size=3,
                        groups="3 * ${net.hidden_dim}",
                        padding=1,
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
                    pixelhyena_norm_cfg=LazyConfig(torch.nn.GroupNorm)(num_groups=1, num_channels="${net.hidden_dim}"),
                    qk_norm_cfg=LazyConfig(L2Norm)(),
                ),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p="${net.block_cfg.dropout_cfg.p}"),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.1),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
    )

    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)()
    config.optimizer = LazyConfig(torch.optim.AdamW)(params=PLACEHOLDER, lr=0.001, weight_decay=0.01)
    config.train = TrainConfig(batch_size="${dataset.batch_size}", iterations=100_000, grad_clip=10.0)
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
    )
    config.wandb = WandbConfig(job_group="mnist_classification_compile_test", project="nvsubquadratic")

    return config


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
    """Build and resolve the self-contained compile-test Hyena configuration."""
    config = _build_mnist_hyena_config()
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
