"""Focused GPU correctness + benchmark for MLP: torch vs QuACK.

Matches the well/euler training config:
  dim=384, activation="glu", batch_size=24, seq_len=1024, bf16
"""

import time

import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.mlp import MLP


DROPOUT_CFG = LazyConfig(torch.nn.Dropout)(p=0.0)

DIM = 384
ACT = "glu"
B = 24
S = 1024
DTYPE = torch.bfloat16


def make_mlp(backend="torch"):
    """Construct an MLP instance for benchmarking."""
    return MLP(
        dim=DIM,
        activation=ACT,
        dropout_cfg=DROPOUT_CFG,
        expansion_factor=1.0,
        bias=False,
        backend=backend,
    )


def benchmark_fn(fn, warmup=20, iters=200):
    """Return median milliseconds per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main():
    """Run MLP forward/backward benchmarks comparing torch and QuACK backends."""
    device = "cuda"
    print(f"GPU:   {torch.cuda.get_device_name()}")
    print(f"SM:    {torch.cuda.get_device_capability()}")
    import quack

    print(f"QuACK: {quack.__version__}")
    print(f"Config: dim={DIM}, act={ACT}, B={B}, S={S}, dtype=bf16")
    print()

    # ── 1. Build MLPs with shared weights ─────────────────────────────────
    torch.manual_seed(42)
    mlp_t = make_mlp(backend="torch").to(device, DTYPE)
    mlp_q = make_mlp(backend="quack").to(device, DTYPE)
    with torch.no_grad():
        mlp_q.layer1.weight.copy_(mlp_t.layer1.weight)
        mlp_q.layer2.weight.copy_(mlp_t.layer2.weight)

    # ── 2. Forward correctness ────────────────────────────────────────────
    print("=== Forward correctness ===")
    x = torch.randn(B, S, DIM, device=device, dtype=DTYPE)
    with torch.no_grad():
        y_torch = mlp_t(x)
        y_quack = mlp_q(x)
    abs_diff = (y_quack - y_torch).abs()
    rel_diff = abs_diff / (y_torch.abs().clamp(min=1e-6))
    print(f"  max abs diff:  {abs_diff.max().item():.6f}")
    print(f"  mean abs diff: {abs_diff.mean().item():.6f}")
    print(f"  max rel diff:  {rel_diff.max().item():.6f}")
    print(f"  mean rel diff: {rel_diff.mean().item():.6f}")
    fwd_ok = abs_diff.max().item() < 0.05
    print(f"  Status: {'PASS' if fwd_ok else 'FAIL'}")
    assert fwd_ok, f"Forward correctness failed: max abs diff = {abs_diff.max().item()}"

    # ── 3. Backward correctness ───────────────────────────────────────────
    print("\n=== Backward correctness ===")
    x_t = x.clone().requires_grad_(True)
    mlp_t(x_t).sum().backward()

    x_q = x.clone().requires_grad_(True)
    mlp_q(x_q).sum().backward()

    grad_abs = (x_q.grad - x_t.grad).abs()
    grad_rel = grad_abs / (x_t.grad.abs().clamp(min=1e-6))
    print(f"  max abs diff:  {grad_abs.max().item():.6f}")
    print(f"  mean abs diff: {grad_abs.mean().item():.6f}")
    print(f"  max rel diff:  {grad_rel.max().item():.6f}")
    print(f"  mean rel diff: {grad_rel.mean().item():.6f}")
    bwd_ok = grad_abs.max().item() < 0.1
    print(f"  Status: {'PASS' if bwd_ok else 'FAIL'}")
    assert bwd_ok, f"Backward correctness failed: max abs diff = {grad_abs.max().item()}"

    # ── 4. Benchmark ──────────────────────────────────────────────────────
    print("\n=== Benchmark (median of 200 iters, 20 warmup) ===")
    print(f"  Shape: ({B}, {S}, {DIM}) bf16\n")

    x_bench = torch.randn(B, S, DIM, device=device, dtype=DTYPE)

    # Forward only
    t_fwd_torch = benchmark_fn(lambda: mlp_t(x_bench))
    t_fwd_quack = benchmark_fn(lambda: mlp_q(x_bench))
    print("  Forward only:")
    print(f"    torch:   {t_fwd_torch:.3f} ms")
    print(f"    quack:   {t_fwd_quack:.3f} ms")
    print(f"    speedup: {t_fwd_torch / t_fwd_quack:.2f}x")

    # Forward + backward
    def torch_fb():
        xt = x_bench.clone().requires_grad_(True)
        mlp_t(xt).sum().backward()

    def quack_fb():
        xq = x_bench.clone().requires_grad_(True)
        mlp_q(xq).sum().backward()

    t_fb_torch = benchmark_fn(torch_fb)
    t_fb_quack = benchmark_fn(quack_fb)
    print("\n  Forward + Backward:")
    print(f"    torch:   {t_fb_torch:.3f} ms")
    print(f"    quack:   {t_fb_quack:.3f} ms")
    print(f"    speedup: {t_fb_torch / t_fb_quack:.2f}x")

    print("\nAll tests PASSED!")


if __name__ == "__main__":
    main()
