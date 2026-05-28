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

import torch

from nvsubquadratic.modules.grn import GlobalResponseNorm


def test_identity_behavior() -> None:
    """Test that GRN is (approximately) identity at initialization.
    With gamma and beta initialized to zero, the forward should reduce to x.
    """
    torch.manual_seed(0)
    # Test 1D temporal input: [B, T, C]
    b, t, c = 2, 4, 8
    layer = GlobalResponseNorm(dim=c)
    x = torch.randn(b, t, c)
    y = layer(x)
    assert y.shape == x.shape
    # Allow for potential floating-point noise (should be strictly equal here).
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)

    # Test 2D spatial input: [B, H, W, C]
    b, h, w, c = 2, 3, 3, 4
    layer = GlobalResponseNorm(dim=c)
    x = torch.randn(b, h, w, c)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)


def test_shape_preservation() -> None:
    """Test that GRN preserves input shape for common layouts."""
    torch.manual_seed(1)
    # [B, T, C]
    b, t, c = 3, 5, 7
    layer = GlobalResponseNorm(dim=c)
    x = torch.randn(b, t, c)
    y = layer(x)
    assert y.shape == x.shape

    # [B, H, W, C]
    b, h, w, c = 1, 4, 4, 6
    layer = GlobalResponseNorm(dim=c)
    x = torch.randn(b, h, w, c)
    y = layer(x)
    assert y.shape == x.shape


def test_gradient_flow() -> None:
    """Test that gradients flow through gamma and beta parameters."""
    torch.manual_seed(2)
    b, t, c = 2, 3, 5
    layer = GlobalResponseNorm(dim=c)
    x = torch.randn(b, t, c, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert layer.gamma.grad is not None, "gamma.grad should not be None"
    assert layer.beta.grad is not None, "beta.grad should not be None"
    # Gradients should be non-zero for a generic random input.
    assert layer.gamma.grad.abs().sum().item() > 0.0
    assert layer.beta.grad.abs().sum().item() > 0.0


def test_flop_count() -> None:
    """Test that the flop count matches the 6 * T * C formula."""
    dim = 8
    num_tokens = 4
    layer = GlobalResponseNorm(dim=dim)
    flops = layer.flop_count(num_tokens=num_tokens)
    assert flops == 6 * num_tokens * dim


if __name__ == "__main__":
    # Run basic self-tests when this module is executed directly.
    test_identity_behavior()
    test_shape_preservation()
    test_gradient_flow()
    test_flop_count()
    print("All tests passed!")
