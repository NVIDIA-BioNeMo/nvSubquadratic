# TODO: Add license header here


"""Memory-efficient (chunked) FFT convolution operators.

This module provides memory-efficient variants of FFT convolutions that reduce
peak GPU memory usage by processing channels in chunks.

Memory problem
--------------
FFT convolutions are memory-intensive because:
1. They require FP32 precision for complex FFT operations
2. They create large temporary complex tensors: fft(x), fft(kernel)
3. Complex tensors are 2x the size of real tensors

For a tensor [B, H, X, Y] with FFT size [Fx, Fy], the intermediate complex
tensors are roughly [B, H, Fx, Fy//2+1] * 16 bytes (complex64).

Solution: Channel chunking
--------------------------
Instead of computing FFT on all channels simultaneously, we process channels
in smaller chunks. This reduces peak memory because the large complex FFT
intermediates only exist for a subset of channels at a time.

Note: PyTorch's FFT autograd already recomputes FFTs during backward (doesn't
store them), so this is NOT activation checkpointing. The savings come from
reducing peak temporary memory during both forward and backward passes.

Performance characteristics (3D, B=2, H=256, spatial=8x64x64):
- chunk=128 (default): ~11% overhead, ~26% memory savings (recommended)
- chunk=64:            ~27% overhead, ~35% memory savings

Memory savings depend on:
- Total channels (H): More channels = more benefit from chunking
- Chunk size: Smaller chunks = lower peak memory but more overhead
- Spatial dimensions: Larger spatial dims = more benefit from chunking

Usage
-----
Explicit chunked functions::

    from nvsubquadratic.ops.fftconv_chunked import fftconv2d_fp32_bhl_chunked

    # Process in chunks of 128 channels (default)
    y = fftconv2d_fp32_bhl_chunked(x, kernel, shortcut)

    # Process in chunks of 64 channels (more memory-efficient)
    y = fftconv2d_fp32_bhl_chunked(x, kernel, shortcut, chunk_size=64)

Drop-in replacements that automatically use chunking when enabled::

    from nvsubquadratic.ops.fftconv_chunked import fftconv2d_fp32_bhl, set_chunking_enabled

    set_chunking_enabled(True)   # Enable chunked processing
    y = fftconv2d_fp32_bhl(x, kernel, shortcut)  # Uses chunking

Control chunking via global flags::

    from nvsubquadratic.ops.fftconv_chunked import (
        set_chunking_enabled,
        set_default_chunk_size,
    )

    set_chunking_enabled(True)    # Enable chunked processing
    set_default_chunk_size(64)    # Set chunk size
"""

from __future__ import annotations

import functools
from contextlib import contextmanager

import torch
from einops import rearrange


__all__ = [  # noqa: RUF022
    # Control
    "enable_chunking",
    "set_chunking_enabled",
    "is_chunking_enabled",
    "chunking_enabled",
    "set_default_chunk_size",
    "get_default_chunk_size",
    # Chunked FFT convolutions (explicit)
    "fftconv1d_fp32_bhl_chunked",
    "fftconv2d_fp32_bhl_chunked",
    "fftconv3d_fp32_bhl_chunked",
    "causal_fftconv1d_fp32_bhl_chunked",
    # Drop-in replacements (use chunking when enabled)
    "causal_fftconv1d_fp32_bhl",
    "causal_fftconv1d_fp32_bhl_w_reshape",
    "causal_fftconv1d_fp32_blh",
    "fftconv1d_fp32_bhl",
    "fftconv1d_fp32_bhl_w_reshape",
    "fftconv1d_fp32_blh",
    "fftconv2d_fp32_bhl",
    "fftconv2d_fp32_bhl_w_reshape",
    "fftconv2d_fp32_blh",
    "fftconv3d_fp32_bhl",
    "fftconv3d_fp32_bhl_w_reshape",
    "fftconv3d_fp32_blh",
]

# =============================================================================
# Global configuration
# =============================================================================

_CHUNKING_ENABLED = True
_DEFAULT_CHUNK_SIZE = 128  # Default to 128 for best speed/memory trade-off (~11% overhead, ~26% savings)


def set_chunking_enabled(enabled: bool) -> None:
    """Set whether memory-efficient chunked FFT conv is enabled globally.

    When enabled, FFT convolutions process channels in chunks to reduce peak
    memory. When disabled, they use the standard (faster) implementation.

    Args:
        enabled: If True, use chunked processing for lower memory usage.
    """
    global _CHUNKING_ENABLED
    _CHUNKING_ENABLED = enabled


def is_chunking_enabled() -> bool:
    """Return whether chunked FFT conv is currently enabled."""
    return _CHUNKING_ENABLED


def set_default_chunk_size(chunk_size: int) -> None:
    """Set the default chunk size for chunked FFT convolutions.

    Smaller chunk sizes use less memory but have more overhead.
    Recommended: 128 for best speed/memory trade-off (~11% overhead, ~26% savings).
    Use 64 for more aggressive memory savings (~27% overhead, ~35% savings).

    Args:
        chunk_size: Number of channels to process at once.
    """
    global _DEFAULT_CHUNK_SIZE
    assert chunk_size > 0, f"chunk_size must be positive, got {chunk_size}"
    _DEFAULT_CHUNK_SIZE = chunk_size


def get_default_chunk_size() -> int:
    """Return the default chunk size for chunked FFT convolutions."""
    return _DEFAULT_CHUNK_SIZE


@contextmanager
def chunking_enabled(enabled: bool = True, chunk_size: int | None = None):
    """Context manager to temporarily enable/disable chunked FFT conv.

    Example::

        from nvsubquadratic.ops.fftconv_chunked import (
            fftconv2d_fp32_bhl,
            chunking_enabled,
        )

        # Use chunking with chunk_size=32
        with chunking_enabled(True, chunk_size=32):
            y = fftconv2d_fp32_bhl(x, kernel)

        # Disable chunking (use standard impl)
        with chunking_enabled(False):
            y = fftconv2d_fp32_bhl(x, kernel)

    Args:
        enabled: Whether to enable chunking within this context.
        chunk_size: Optional chunk size override for this context.

    Yields:
        None
    """
    global _CHUNKING_ENABLED, _DEFAULT_CHUNK_SIZE
    old_enabled = _CHUNKING_ENABLED
    old_chunk_size = _DEFAULT_CHUNK_SIZE

    _CHUNKING_ENABLED = enabled
    if chunk_size is not None:
        _DEFAULT_CHUNK_SIZE = chunk_size

    try:
        yield
    finally:
        _CHUNKING_ENABLED = old_enabled
        _DEFAULT_CHUNK_SIZE = old_chunk_size


def enable_chunking(module_or_flag=None, chunk_size: int | None = None):
    """Enable chunked FFT conv globally, as decorator, or as context manager.

    Can be used as:
    1. A function to set global state: ``enable_chunking(True)``
    2. A class decorator: ``@enable_chunking`` on an nn.Module
    3. A context manager: ``with enable_chunking():``

    Args:
        module_or_flag: Either a bool, module class, or None.
        chunk_size: Optional chunk size to set.

    Returns:
        Decorated class, context manager, or None depending on usage.
    """
    if module_or_flag is None:
        return chunking_enabled(True, chunk_size=chunk_size)
    elif isinstance(module_or_flag, bool):
        set_chunking_enabled(module_or_flag)
        if chunk_size is not None:
            set_default_chunk_size(chunk_size)
        return None
    elif isinstance(module_or_flag, type) and issubclass(module_or_flag, torch.nn.Module):
        original_forward = module_or_flag.forward

        @functools.wraps(original_forward)
        def chunked_forward(self, *args, **kwargs):
            with chunking_enabled(True, chunk_size=chunk_size):
                return original_forward(self, *args, **kwargs)

        module_or_flag.forward = chunked_forward
        return module_or_flag
    else:
        raise TypeError(f"Expected bool, nn.Module class, or None. Got {type(module_or_flag).__name__}")


# =============================================================================
# Chunked FFT convolution implementations
# =============================================================================


def fftconv1d_fp32_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """1D FFT convolution (BHL layout) with channel chunking.

    Args:
        x: Input tensor [B, H, L], float32
        kernel: Kernel tensor [1|B, H, K], float32
        shortcut: Optional per-channel scale [H], float32
        chunk_size: Channels per chunk (None = use global default)

    Returns:
        Output tensor [B, H, L]
    """
    from nvsubquadratic.ops.fftconv import fftconv1d_fp32_bhl as _fftconv1d_bhl_std

    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE

    _B, H, _L = x.shape

    if H <= chunk_size:
        return _fftconv1d_bhl_std(x, kernel, shortcut)

    outputs = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        x_chunk = x[:, start:end, :]
        k_chunk = kernel[:, start:end, :]
        s_chunk = shortcut[start:end] if shortcut is not None else None

        y_chunk = _fftconv1d_bhl_std(x_chunk, k_chunk, s_chunk)
        outputs.append(y_chunk)

    return torch.cat(outputs, dim=1)


def causal_fftconv1d_fp32_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """1D causal FFT convolution (BHL layout) with channel chunking.

    Args:
        x: Input tensor [B, H, L], float32
        kernel: Kernel tensor [1|B, H, K], float32
        shortcut: Optional per-channel scale [H], float32
        chunk_size: Channels per chunk (None = use global default)

    Returns:
        Output tensor [B, H, L]
    """
    from nvsubquadratic.ops.fftconv import causal_fftconv1d_fp32_bhl as _causal_fftconv1d_bhl_std

    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE

    _B, H, _L = x.shape

    if H <= chunk_size:
        return _causal_fftconv1d_bhl_std(x, kernel, shortcut)

    outputs = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        x_chunk = x[:, start:end, :]
        k_chunk = kernel[:, start:end, :]
        s_chunk = shortcut[start:end] if shortcut is not None else None

        y_chunk = _causal_fftconv1d_bhl_std(x_chunk, k_chunk, s_chunk)
        outputs.append(y_chunk)

    return torch.cat(outputs, dim=1)


def fftconv2d_fp32_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """2D FFT convolution (BHL layout) with channel chunking.

    Reduces peak memory by processing channels in chunks instead of all at once.
    Typical savings: ~26% memory with ~11% overhead (chunk=128).

    Args:
        x: Input tensor [B, H, X, Y], float32
        kernel: Kernel tensor [1|B, H, K_x, K_y], float32
        shortcut: Optional per-channel scale [H], float32
        chunk_size: Channels per chunk (None = use global default of 128)

    Returns:
        Output tensor [B, H, X, Y]
    """
    from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl as _fftconv2d_bhl_std

    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE

    _B, H, _X_in, _Y_in = x.shape

    if H <= chunk_size:
        return _fftconv2d_bhl_std(x, kernel, shortcut)

    outputs = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        y_chunk = _fftconv2d_bhl_std(
            x[:, start:end],
            kernel[:, start:end],
            shortcut[start:end] if shortcut is not None else None,
        )
        outputs.append(y_chunk)

    return torch.cat(outputs, dim=1)


def fftconv3d_fp32_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """3D FFT convolution (BHL layout) with channel chunking.

    Reduces peak memory by processing channels in chunks.

    Performance characteristics (B=2, H=256, 8x64x64 spatial):
    - chunk=128 (default): ~11% overhead, ~26% memory savings
    - chunk=64:            ~27% overhead, ~35% memory savings

    Args:
        x: Input tensor [B, H, X, Y, Z], float32
        kernel: Kernel tensor [1|B, H, K_x, K_y, K_z], float32
        shortcut: Optional per-channel scale [H], float32
        chunk_size: Channels per chunk (None = use global default of 128)

    Returns:
        Output tensor [B, H, X, Y, Z]
    """
    from nvsubquadratic.ops.fftconv import fftconv3d_fp32_bhl as _fftconv3d_bhl_std

    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE

    _B, H, _X_in, _Y_in, _Z_in = x.shape

    if H <= chunk_size:
        return _fftconv3d_bhl_std(x, kernel, shortcut)

    outputs = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        y_chunk = _fftconv3d_bhl_std(
            x[:, start:end],
            kernel[:, start:end],
            shortcut[start:end] if shortcut is not None else None,
        )
        outputs.append(y_chunk)

    return torch.cat(outputs, dim=1)


# =============================================================================
# Drop-in replacement functions (auto-select based on global flag)
# =============================================================================


def fftconv1d_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution (BHL layout).

    Drop-in replacement that uses chunking when ``is_chunking_enabled()`` is True.
    Control via ``set_chunking_enabled()`` or the ``chunking_enabled()`` context manager.

    Args:
        x: Input tensor [B, H, L]
        kernel: Kernel tensor [1|B, H, K]
        shortcut: Optional per-channel scale [H]

    Returns:
        Output tensor [B, H, L]
    """
    if _CHUNKING_ENABLED:
        return fftconv1d_fp32_bhl_chunked(x, kernel, shortcut)
    else:
        from nvsubquadratic.ops.fftconv import fftconv1d_fp32_bhl as _std

        return _std(x, kernel, shortcut)


def causal_fftconv1d_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D causal FFT convolution (BHL layout).

    Drop-in replacement that uses chunking when ``is_chunking_enabled()`` is True.
    Control via ``set_chunking_enabled()`` or the ``chunking_enabled()`` context manager.

    Args:
        x: Input tensor [B, H, L]
        kernel: Kernel tensor [1|B, H, K]
        shortcut: Optional per-channel scale [H]

    Returns:
        Output tensor [B, H, L] (causal: output[i] depends only on input[0..i])
    """
    if _CHUNKING_ENABLED:
        return causal_fftconv1d_fp32_bhl_chunked(x, kernel, shortcut)
    else:
        from nvsubquadratic.ops.fftconv import causal_fftconv1d_fp32_bhl as _std

        return _std(x, kernel, shortcut)


def fftconv2d_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution (BHL layout).

    Drop-in replacement that uses chunking when ``is_chunking_enabled()`` is True.
    Control via ``set_chunking_enabled()`` or the ``chunking_enabled()`` context manager.

    Args:
        x: Input tensor [B, H, X, Y]
        kernel: Kernel tensor [1|B, H, K_x, K_y]
        shortcut: Optional per-channel scale [H]

    Returns:
        Output tensor [B, H, X, Y]
    """
    if _CHUNKING_ENABLED:
        return fftconv2d_fp32_bhl_chunked(x, kernel, shortcut)
    else:
        from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl as _std

        return _std(x, kernel, shortcut)


def fftconv3d_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D FFT convolution (BHL layout).

    Drop-in replacement that uses chunking when ``is_chunking_enabled()`` is True.
    Control via ``set_chunking_enabled()`` or the ``chunking_enabled()`` context manager.

    Args:
        x: Input tensor [B, H, X, Y, Z]
        kernel: Kernel tensor [1|B, H, K_x, K_y, K_z]
        shortcut: Optional per-channel scale [H]

    Returns:
        Output tensor [B, H, X, Y, Z]
    """
    if _CHUNKING_ENABLED:
        return fftconv3d_fp32_bhl_chunked(x, kernel, shortcut)
    else:
        from nvsubquadratic.ops.fftconv import fftconv3d_fp32_bhl as _std

        return _std(x, kernel, shortcut)


# BLH layout wrappers (channels-last)
def fftconv1d_fp32_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution (BLH layout, channels-last).

    Args:
        x: Input tensor [B, L, H]
        kernel: Kernel tensor [B, K, H]
        shortcut: Optional per-channel scale [H]

    Returns:
        Output tensor [B, L, H]
    """
    x_bhl = rearrange(x, "b l h -> b h l")
    k_bhl = rearrange(kernel, "b l h -> b h l")
    y_bhl = fftconv1d_fp32_bhl(x_bhl, k_bhl, shortcut)
    return rearrange(y_bhl, "b h l -> b l h")


def causal_fftconv1d_fp32_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D causal FFT convolution (BLH layout, channels-last).

    Args:
        x: Input tensor [B, L, H]
        kernel: Kernel tensor [B, K, H]
        shortcut: Optional per-channel scale [H]

    Returns:
        Output tensor [B, L, H] (causal: output[i] depends only on input[0..i])
    """
    x_bhl = rearrange(x, "b l h -> b h l")
    k_bhl = rearrange(kernel, "b l h -> b h l")
    y_bhl = causal_fftconv1d_fp32_bhl(x_bhl, k_bhl, shortcut)
    return rearrange(y_bhl, "b h l -> b l h")


def fftconv2d_fp32_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution (BLH layout, channels-last).

    Args:
        x: Input tensor [B, X, Y, H]
        kernel: Kernel tensor [B, K_x, K_y, H]
        shortcut: Optional per-channel scale [H]

    Returns:
        Output tensor [B, X, Y, H]
    """
    x_bhl = rearrange(x, "b x y h -> b h x y")
    k_bhl = rearrange(kernel, "b x y h -> b h x y")
    y_bhl = fftconv2d_fp32_bhl(x_bhl, k_bhl, shortcut)
    return rearrange(y_bhl, "b h x y -> b x y h")


def fftconv3d_fp32_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D FFT convolution (BLH layout, channels-last).

    Args:
        x: Input tensor [B, X, Y, Z, H]
        kernel: Kernel tensor [B, K_x, K_y, K_z, H]
        shortcut: Optional per-channel scale [H]

    Returns:
        Output tensor [B, X, Y, Z, H]
    """
    x_bhl = rearrange(x, "b x y z h -> b h x y z")
    k_bhl = rearrange(kernel, "b x y z h -> b h x y z")
    y_bhl = fftconv3d_fp32_bhl(x_bhl, k_bhl, shortcut)
    return rearrange(y_bhl, "b h x y z -> b x y z h")


# w_reshape aliases: BLH input → internal BHL conv → BLH output
# These match the naming convention in nvsubquadratic.ops.fftconv
fftconv1d_fp32_bhl_w_reshape = fftconv1d_fp32_blh
fftconv2d_fp32_bhl_w_reshape = fftconv2d_fp32_blh
fftconv3d_fp32_bhl_w_reshape = fftconv3d_fp32_blh
causal_fftconv1d_fp32_bhl_w_reshape = causal_fftconv1d_fp32_blh


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import gc

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}")

    # Test correctness
    print("\n=== Testing correctness ===")
    B, H, X, Y = 8, 128, 64, 64
    K = 32

    x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
    kernel = torch.randn(1, H, K, K, device=device, dtype=torch.float32, requires_grad=True)
    shortcut = torch.randn(H, device=device, dtype=torch.float32, requires_grad=True)

    # Reference
    from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl as fftconv2d_bhl_std

    y_std = fftconv2d_bhl_std(x, kernel, shortcut)
    loss_std = y_std.sum()
    loss_std.backward()
    grad_x_std = x.grad.clone()
    grad_k_std = kernel.grad.clone()
    grad_s_std = shortcut.grad.clone()

    x.grad = None
    kernel.grad = None
    shortcut.grad = None

    # Chunked
    y_chunked = fftconv2d_fp32_bhl_chunked(x, kernel, shortcut, chunk_size=32)
    loss_chunked = y_chunked.sum()
    loss_chunked.backward()

    print(f"Output diff: {(y_std - y_chunked).abs().max().item():.2e}")
    print(f"Grad x diff: {(grad_x_std - x.grad).abs().max().item():.2e}")
    print(f"Grad k diff: {(grad_k_std - kernel.grad).abs().max().item():.2e}")
    print(f"Grad s diff: {(grad_s_std - shortcut.grad).abs().max().item():.2e}")

    # Memory comparison
    if device == "cuda":
        print("\n=== Memory comparison ===")
        B, H, X, Y = 16, 256, 128, 128
        K = 64

        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        kernel = torch.randn(1, H, K, K, device=device, dtype=torch.float32, requires_grad=True)
        shortcut = torch.randn(H, device=device, dtype=torch.float32)

        # Standard
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        y = fftconv2d_bhl_std(x, kernel, shortcut)
        y.sum().backward()
        peak_std = torch.cuda.max_memory_allocated() / 1024 / 1024

        x.grad = None
        kernel.grad = None

        print(f"Standard: {peak_std:.1f} MB")

        for cs in [128, 64, 32, 16]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            y = fftconv2d_fp32_bhl_chunked(x, kernel, shortcut, chunk_size=cs)
            y.sum().backward()
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            savings = peak_std - peak
            pct = 100 * savings / peak_std

            print(f"chunk_size={cs:3d}: {peak:.1f} MB (savings: {savings:+.1f} MB, {pct:+.1f}%)")

            x.grad = None
            kernel.grad = None

    # Test 3D
    print("\n=== Testing 3D ===")
    B, H, X, Y, Z = 2, 64, 32, 32, 32
    K = 16

    x = torch.randn(B, H, X, Y, Z, device=device, dtype=torch.float32, requires_grad=True)
    kernel = torch.randn(1, H, K, K, K, device=device, dtype=torch.float32, requires_grad=True)

    from nvsubquadratic.ops.fftconv import fftconv3d_fp32_bhl as fftconv3d_bhl_std

    y_std = fftconv3d_bhl_std(x, kernel, None)
    y_std.sum().backward()
    grad_std = x.grad.clone()

    x.grad = None
    kernel.grad = None

    y_chunked = fftconv3d_fp32_bhl_chunked(x, kernel, None, chunk_size=16)
    y_chunked.sum().backward()

    print(f"3D output diff: {(y_std - y_chunked).abs().max().item():.2e}")
    print(f"3D grad diff: {(grad_std - x.grad).abs().max().item():.2e}")

    print("\nAll tests passed!")
