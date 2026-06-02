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

"""Tests for chunked (memory-efficient) FFT convolution operators.

These tests verify that:
1. Chunked implementations produce identical results to standard implementations
2. Drop-in replacement API works correctly (auto-selects based on global flag)
3. Context manager and global configuration work as expected
4. Edge cases (H not divisible by chunk_size, H <= chunk_size) are handled
"""

import pytest
import torch

from nvsubquadratic.ops.fftconv import (
    causal_fftconv1d_fp32_bhl as causal_fftconv1d_std,
)
from nvsubquadratic.ops.fftconv import (
    fftconv1d_fp32_bhl as fftconv1d_std,
)
from nvsubquadratic.ops.fftconv import (
    fftconv2d_fp32_bhl as fftconv2d_std,
)
from nvsubquadratic.ops.fftconv import (
    fftconv3d_fp32_bhl as fftconv3d_std,
)
from nvsubquadratic.ops.fftconv_chunked import (
    # Explicit chunked functions
    causal_fftconv1d_fp32_bhl_chunked,
    chunking_enabled,
    fftconv1d_fp32_bhl_chunked,
    fftconv2d_fp32_bhl_chunked,
    fftconv3d_fp32_bhl_chunked,
    # Configuration
    get_default_chunk_size,
    is_chunking_enabled,
    set_chunking_enabled,
    set_default_chunk_size,
)
from nvsubquadratic.ops.fftconv_chunked import (
    # Drop-in replacements (auto-select based on global flag)
    fftconv1d_fp32_bhl as fftconv1d_bhl_dropin,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv2d_fp32_bhl as fftconv2d_bhl_dropin,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv3d_fp32_bhl as fftconv3d_bhl_dropin,
)


# Tolerances for FP32 comparisons
# Output should be exact (same computation order within chunks)
# Gradients may have small diffs due to accumulation order
ATOL_OUTPUT = 2e-5
ATOL_GRAD = 5e-4  # More lenient for gradients with large reductions


@pytest.fixture
def device():
    """Return CUDA device if available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestChunkedFFTConv1D:
    """Tests for 1D chunked FFT convolution."""

    @pytest.mark.parametrize(
        "B,H,L,K,chunk_size",
        [
            (2, 64, 256, 32, 128),
            (2, 64, 256, 32, 32),
            (2, 128, 512, 64, 64),
            (2, 256, 1024, 128, 128),
            (2, 256, 1024, 128, 64),
        ],
    )
    def test_fftconv1d_forward_backward(self, device, B, H, L, K, chunk_size):
        """Test that chunked 1D FFT conv matches standard implementation."""
        torch.manual_seed(42)

        x = torch.randn(B, H, L, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32, requires_grad=True)
        shortcut = torch.randn(H, device=device, dtype=torch.float32, requires_grad=True)

        # Standard
        x_std = x.detach().clone().requires_grad_(True)
        k_std = kernel.detach().clone().requires_grad_(True)
        s_std = shortcut.detach().clone().requires_grad_(True)

        y_std = fftconv1d_std(x_std, k_std, s_std)
        y_std.sum().backward()

        # Chunked
        x_chunk = x.detach().clone().requires_grad_(True)
        k_chunk = kernel.detach().clone().requires_grad_(True)
        s_chunk = shortcut.detach().clone().requires_grad_(True)

        y_chunk = fftconv1d_fp32_bhl_chunked(x_chunk, k_chunk, s_chunk, chunk_size=chunk_size)
        y_chunk.sum().backward()

        # Compare
        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_OUTPUT, rtol=0)
        torch.testing.assert_close(x_chunk.grad, x_std.grad, atol=ATOL_GRAD, rtol=0)
        torch.testing.assert_close(k_chunk.grad, k_std.grad, atol=ATOL_GRAD, rtol=0)
        torch.testing.assert_close(s_chunk.grad, s_std.grad, atol=ATOL_GRAD, rtol=0)

    @pytest.mark.parametrize("chunk_size", [128, 64, 32])
    def test_causal_fftconv1d(self, device, chunk_size):
        """Test causal 1D FFT conv chunking."""
        torch.manual_seed(42)
        B, H, L, K = 2, 128, 512, 64

        x = torch.randn(B, H, L, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, K, device=device, dtype=torch.float32, requires_grad=True)
        shortcut = torch.randn(H, device=device, dtype=torch.float32, requires_grad=True)

        # Standard
        x_std = x.detach().clone().requires_grad_(True)
        k_std = kernel.detach().clone().requires_grad_(True)
        s_std = shortcut.detach().clone().requires_grad_(True)

        y_std = causal_fftconv1d_std(x_std, k_std, s_std)
        y_std.sum().backward()

        # Chunked
        x_chunk = x.detach().clone().requires_grad_(True)
        k_chunk = kernel.detach().clone().requires_grad_(True)
        s_chunk = shortcut.detach().clone().requires_grad_(True)

        y_chunk = causal_fftconv1d_fp32_bhl_chunked(x_chunk, k_chunk, s_chunk, chunk_size=chunk_size)
        y_chunk.sum().backward()

        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_OUTPUT, rtol=0)
        torch.testing.assert_close(x_chunk.grad, x_std.grad, atol=ATOL_GRAD, rtol=0)


class TestChunkedFFTConv2D:
    """Tests for 2D chunked FFT convolution."""

    @pytest.mark.parametrize(
        "B,H,X,Y,Kx,Ky,chunk_size",
        [
            (2, 64, 64, 64, 32, 32, 128),
            (2, 128, 64, 64, 32, 32, 64),
            (2, 256, 64, 64, 32, 32, 128),
            (2, 256, 64, 64, 32, 32, 64),
        ],
    )
    def test_fftconv2d_forward_backward(self, device, B, H, X, Y, Kx, Ky, chunk_size):
        """Test that chunked 2D FFT conv matches standard implementation."""
        torch.manual_seed(42)

        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)
        shortcut = torch.randn(H, device=device, dtype=torch.float32, requires_grad=True)

        # Standard
        x_std = x.detach().clone().requires_grad_(True)
        k_std = kernel.detach().clone().requires_grad_(True)
        s_std = shortcut.detach().clone().requires_grad_(True)

        y_std = fftconv2d_std(x_std, k_std, s_std)
        y_std.sum().backward()

        # Chunked
        x_chunk = x.detach().clone().requires_grad_(True)
        k_chunk = kernel.detach().clone().requires_grad_(True)
        s_chunk = shortcut.detach().clone().requires_grad_(True)

        y_chunk = fftconv2d_fp32_bhl_chunked(x_chunk, k_chunk, s_chunk, chunk_size=chunk_size)
        y_chunk.sum().backward()

        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_OUTPUT, rtol=0)
        torch.testing.assert_close(x_chunk.grad, x_std.grad, atol=ATOL_GRAD, rtol=0)
        torch.testing.assert_close(k_chunk.grad, k_std.grad, atol=ATOL_GRAD, rtol=0)
        torch.testing.assert_close(s_chunk.grad, s_std.grad, atol=ATOL_GRAD, rtol=0)


class TestChunkedFFTConv3D:
    """Tests for 3D chunked FFT convolution."""

    @pytest.mark.parametrize(
        "B,H,X,Y,Z,Kx,Ky,Kz,chunk_size",
        [
            (2, 64, 8, 64, 64, 8, 64, 64, 128),  # Full kernel, H <= chunk
            (2, 128, 8, 64, 64, 4, 32, 32, 64),  # Half kernel
            (2, 256, 8, 64, 64, 8, 64, 64, 128),  # Many channels
            (2, 256, 8, 64, 64, 8, 64, 64, 64),  # Many channels, smaller chunk
            (4, 64, 32, 32, 32, 16, 16, 16, 32),  # Cube
        ],
    )
    def test_fftconv3d_forward_backward(self, device, B, H, X, Y, Z, Kx, Ky, Kz, chunk_size):
        """Test that chunked 3D FFT conv matches standard implementation."""
        torch.manual_seed(42)

        x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32, requires_grad=True)
        shortcut = torch.randn(H, device=device, dtype=torch.float32, requires_grad=True)

        # Standard
        x_std = x.detach().clone().requires_grad_(True)
        k_std = kernel.detach().clone().requires_grad_(True)
        s_std = shortcut.detach().clone().requires_grad_(True)

        y_std = fftconv3d_std(x_std, k_std, s_std)
        y_std.sum().backward()

        # Chunked
        x_chunk = x.detach().clone().requires_grad_(True)
        k_chunk = kernel.detach().clone().requires_grad_(True)
        s_chunk = shortcut.detach().clone().requires_grad_(True)

        y_chunk = fftconv3d_fp32_bhl_chunked(x_chunk, k_chunk, s_chunk, chunk_size=chunk_size)
        y_chunk.sum().backward()

        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_OUTPUT, rtol=0)
        torch.testing.assert_close(x_chunk.grad, x_std.grad, atol=ATOL_GRAD, rtol=0)
        torch.testing.assert_close(k_chunk.grad, k_std.grad, atol=ATOL_GRAD, rtol=0)
        torch.testing.assert_close(s_chunk.grad, s_std.grad, atol=ATOL_GRAD, rtol=0)

    def test_fftconv3d_no_shortcut(self, device):
        """Test 3D FFT conv without shortcut."""
        torch.manual_seed(42)
        B, H, X, Y, Z = 2, 256, 8, 64, 64
        Kx, Ky, Kz = 8, 64, 64

        x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, Kx, Ky, Kz, device=device, dtype=torch.float32, requires_grad=True)

        # Standard
        x_std = x.detach().clone().requires_grad_(True)
        k_std = kernel.detach().clone().requires_grad_(True)

        y_std = fftconv3d_std(x_std, k_std, None)
        y_std.sum().backward()

        # Chunked
        x_chunk = x.detach().clone().requires_grad_(True)
        k_chunk = kernel.detach().clone().requires_grad_(True)

        y_chunk = fftconv3d_fp32_bhl_chunked(x_chunk, k_chunk, None, chunk_size=64)
        y_chunk.sum().backward()

        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_OUTPUT, rtol=0)
        torch.testing.assert_close(x_chunk.grad, x_std.grad, atol=ATOL_GRAD, rtol=0)
        torch.testing.assert_close(k_chunk.grad, k_std.grad, atol=ATOL_GRAD, rtol=0)


class TestChunkSizeConfig:
    """Tests for chunk size configuration."""

    def test_default_chunk_size(self):
        """Test default chunk size is 128."""
        assert get_default_chunk_size() == 128

    def test_set_chunk_size(self):
        """Test setting chunk size."""
        original = get_default_chunk_size()
        try:
            set_default_chunk_size(64)
            assert get_default_chunk_size() == 64

            set_default_chunk_size(32)
            assert get_default_chunk_size() == 32
        finally:
            set_default_chunk_size(original)

    def test_bypass_when_h_less_than_chunk(self, device):
        """Test that chunking is bypassed when H <= chunk_size."""
        torch.manual_seed(42)

        # H=64, chunk=128 -> should bypass chunking (exact same result)
        x = torch.randn(2, 64, 8, 64, 64, device=device, dtype=torch.float32)
        kernel = torch.randn(1, 64, 8, 64, 64, device=device, dtype=torch.float32)
        shortcut = torch.randn(64, device=device, dtype=torch.float32)

        y_std = fftconv3d_std(x, kernel, shortcut)
        y_chunk = fftconv3d_fp32_bhl_chunked(x, kernel, shortcut, chunk_size=128)

        # Should be exactly equal (same code path)
        torch.testing.assert_close(y_chunk, y_std, atol=0, rtol=0)

    def test_h_not_divisible_by_chunk(self, device):
        """Test chunking works when H is not evenly divisible by chunk_size."""
        torch.manual_seed(42)

        # H=100, chunk=64 -> will have chunks of [64, 36]
        B, H, X, Y = 2, 100, 32, 32
        Kx, Ky = 16, 16

        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)
        shortcut = torch.randn(H, device=device, dtype=torch.float32, requires_grad=True)

        # Standard
        x_std = x.detach().clone().requires_grad_(True)
        k_std = kernel.detach().clone().requires_grad_(True)
        s_std = shortcut.detach().clone().requires_grad_(True)

        y_std = fftconv2d_std(x_std, k_std, s_std)
        y_std.sum().backward()

        # Chunked with non-divisible chunk size
        x_chunk = x.detach().clone().requires_grad_(True)
        k_chunk = kernel.detach().clone().requires_grad_(True)
        s_chunk = shortcut.detach().clone().requires_grad_(True)

        y_chunk = fftconv2d_fp32_bhl_chunked(x_chunk, k_chunk, s_chunk, chunk_size=64)
        y_chunk.sum().backward()

        torch.testing.assert_close(y_chunk, y_std, atol=ATOL_OUTPUT, rtol=0)
        torch.testing.assert_close(x_chunk.grad, x_std.grad, atol=ATOL_GRAD, rtol=0)
        torch.testing.assert_close(k_chunk.grad, k_std.grad, atol=ATOL_GRAD, rtol=0)
        torch.testing.assert_close(s_chunk.grad, s_std.grad, atol=ATOL_GRAD, rtol=0)


class TestDropInReplacementAPI:
    """Tests for drop-in replacement functions that auto-select based on global flag."""

    def test_chunking_enabled_flag(self):
        """Test is_chunking_enabled / set_chunking_enabled."""
        original = is_chunking_enabled()
        try:
            set_chunking_enabled(False)
            assert is_chunking_enabled() is False

            set_chunking_enabled(True)
            assert is_chunking_enabled() is True
        finally:
            set_chunking_enabled(original)

    def test_dropin_uses_standard_when_disabled(self, device):
        """Test drop-in replacement uses standard impl when chunking disabled."""
        torch.manual_seed(42)
        B, H, X, Y = 2, 128, 32, 32

        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, 16, 16, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_std = fftconv2d_std(x, kernel, shortcut)

        # Disable chunking - should use standard implementation
        with chunking_enabled(False):
            y_dropin = fftconv2d_bhl_dropin(x, kernel, shortcut)

        # Should be exactly equal (same code path)
        torch.testing.assert_close(y_dropin, y_std, atol=0, rtol=0)

    def test_dropin_uses_chunked_when_enabled(self, device):
        """Test drop-in replacement uses chunked impl when chunking enabled."""
        torch.manual_seed(42)
        B, H, X, Y = 2, 128, 32, 32

        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        kernel = torch.randn(1, H, 16, 16, device=device, dtype=torch.float32)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        y_std = fftconv2d_std(x, kernel, shortcut)

        # Enable chunking with specific chunk size
        with chunking_enabled(True, chunk_size=32):
            y_dropin = fftconv2d_bhl_dropin(x, kernel, shortcut)

        # Should match (chunked produces same result)
        torch.testing.assert_close(y_dropin, y_std, atol=ATOL_OUTPUT, rtol=0)

    def test_context_manager_restores_state(self):
        """Test chunking_enabled context manager restores original state."""
        original_enabled = is_chunking_enabled()
        original_chunk = get_default_chunk_size()

        # Modify inside context
        with chunking_enabled(not original_enabled, chunk_size=17):
            assert is_chunking_enabled() != original_enabled
            assert get_default_chunk_size() == 17

        # Should be restored
        assert is_chunking_enabled() == original_enabled
        assert get_default_chunk_size() == original_chunk

    def test_context_manager_restores_on_exception(self):
        """Test context manager restores state even on exception."""
        original_enabled = is_chunking_enabled()
        original_chunk = get_default_chunk_size()

        try:
            with chunking_enabled(not original_enabled, chunk_size=23):
                raise ValueError("test exception")
        except ValueError:
            pass

        # Should be restored despite exception
        assert is_chunking_enabled() == original_enabled
        assert get_default_chunk_size() == original_chunk

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_dropin_all_dimensions(self, device, dim):
        """Test drop-in replacements work for all dimensions (1D, 2D, 3D)."""
        torch.manual_seed(42)

        if dim == 1:
            x = torch.randn(2, 64, 256, device=device, dtype=torch.float32)
            kernel = torch.randn(1, 64, 32, device=device, dtype=torch.float32)
            std_fn = fftconv1d_std
            dropin_fn = fftconv1d_bhl_dropin
        elif dim == 2:
            x = torch.randn(2, 64, 32, 32, device=device, dtype=torch.float32)
            kernel = torch.randn(1, 64, 16, 16, device=device, dtype=torch.float32)
            std_fn = fftconv2d_std
            dropin_fn = fftconv2d_bhl_dropin
        else:  # dim == 3
            x = torch.randn(2, 64, 8, 32, 32, device=device, dtype=torch.float32)
            kernel = torch.randn(1, 64, 4, 16, 16, device=device, dtype=torch.float32)
            std_fn = fftconv3d_std
            dropin_fn = fftconv3d_bhl_dropin

        shortcut = torch.randn(64, device=device, dtype=torch.float32)

        y_std = std_fn(x, kernel, shortcut)

        with chunking_enabled(True, chunk_size=32):
            y_dropin = dropin_fn(x, kernel, shortcut)

        torch.testing.assert_close(y_dropin, y_std, atol=ATOL_OUTPUT, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
