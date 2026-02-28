"""Quick test: torch.compile fftconv2d_bhl with batch-dependent kernels (FiLM scenario)."""

import torch

from nvsubquadratic.ops.fftconv import fftconv2d_bhl


def test_compile_batch_dependent():
    B, H, X, Y = 4, 384, 15, 14
    x = torch.randn(B, H, X, Y, device="cuda", dtype=torch.float32)
    kernel = torch.randn(B, H, X, Y, device="cuda", dtype=torch.float32)
    shortcut = torch.randn(H, device="cuda", dtype=torch.float32)

    # Eager baseline
    y_eager = fftconv2d_bhl(x, kernel, shortcut)
    print(f"Eager OK — shape={y_eager.shape}, max={y_eager.abs().max().item():.4f}")

    # Compiled
    compiled_fn = torch.compile(fftconv2d_bhl, mode="max-autotune")
    y_compiled = compiled_fn(x, kernel, shortcut)
    print(f"Compiled OK — shape={y_compiled.shape}, max={y_compiled.abs().max().item():.4f}")

    diff = (y_eager - y_compiled).abs().max().item()
    print(f"Max diff (eager vs compiled): {diff:.2e}")

    # Backward through compiled graph
    x_g = x.detach().requires_grad_(True)
    k_g = kernel.detach().requires_grad_(True)
    y = compiled_fn(x_g, k_g, shortcut)
    y.sum().backward()
    print(f"Backward OK — grad_x max={x_g.grad.abs().max().item():.4f}, grad_k max={k_g.grad.abs().max().item():.4f}")

    print("\nAll checks PASSED.")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    test_compile_batch_dependent()
