#!/usr/bin/env python
"""Benchmark Hyena block stack: forward + backward with torch.compile.

Compares the optimized channels-first norm path against the baseline
movedim-based path.  Reports it/s and peak GPU memory for each
(patch_size, batch_size) combination.

Usage (must run on a GPU node, e.g. via srun):
    python scripts/benchmark_hyena_permutations.py \
        --patch-sizes 16 4 \
        --batch-sizes 1 4 16 32 64 \
        --num-blocks 12 \
        --warmup 10 \
        --steps 50
"""

import argparse
import gc
import sys
import time

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.utils.qk_norm import L2Norm


# ── Model hyperparameters (matching active_matter Hyena config) ──────────────
HIDDEN_DIM = 384
IMG_SIZE = 256
DATA_DIM = 2
OMEGA_0 = 30.0


# ── Baseline (old) norm implementation ───────────────────────────────────────


def _baseline_apply_norm_bchw(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Old movedim-based norm path (pre-optimization)."""
    if isinstance(norm, nn.GroupNorm):
        return norm(x)
    if isinstance(norm, L2Norm):
        return norm(x.movedim(1, -1)).movedim(-1, 1)
    # RMSNorm path: movedim + reshape to 2D, norm, view + movedim back
    shape = x.shape  # [B, C, *spatial]
    x = x.movedim(1, -1).reshape(-1, shape[1])
    x = norm(x)
    return x.view(shape[0], *shape[2:], shape[1]).movedim(-1, 1)


# ── Mode switching ───────────────────────────────────────────────────────────

import nvsubquadratic.modules.hyena_nd as _hmod


_OPTIMIZED_NORM = _hmod._apply_norm_bchw  # save optimized version at import time


def set_mode(mode: str):
    """Swap the norm dispatch function used inside Hyena.forward."""
    if mode == "baseline":
        _hmod._apply_norm_bchw = _baseline_apply_norm_bchw
    elif mode == "optimized":
        _hmod._apply_norm_bchw = _OPTIMIZED_NORM
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Model construction ───────────────────────────────────────────────────────


class BlockStack(nn.Module):
    """Stack of ResidualBlocks for benchmarking (no Patchify/Unpatchify)."""

    def __init__(self, blocks, out_norm):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = out_norm

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x, condition=None)
        return self.out_norm(x)


def make_block_cfg(hidden_dim: int, seq_len: int) -> LazyConfig:
    """Build a LazyConfig for one ResidualBlock(QKVSequenceMixer(Hyena) + MLP)."""
    norm_cfg = LazyConfig(RMSNorm)(dim=hidden_dim)

    return LazyConfig(ResidualBlock)(
        sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
            hidden_dim=hidden_dim,
            mixer_cfg=LazyConfig(Hyena)(
                global_conv_cfg=LazyConfig(CKConvND)(
                    data_dim=DATA_DIM,
                    hidden_dim=hidden_dim,
                    fft_padding="circular",
                    use_fp16_fft=False,
                    kernel_cfg=LazyConfig(SIRENKernelND)(
                        data_dim=DATA_DIM,
                        out_dim=hidden_dim,
                        mlp_hidden_dim=64,
                        num_layers=3,
                        embedding_dim=64,
                        omega_0=OMEGA_0,
                        L_cache=seq_len,
                        use_bias=True,
                        hidden_omega_0=1.0,
                    ),
                    mask_cfg=LazyConfig(nn.Identity)(),
                    grid_type="single",
                ),
                short_conv_cfg=LazyConfig(nn.Conv2d)(
                    in_channels=3 * hidden_dim,
                    out_channels=3 * hidden_dim,
                    kernel_size=3,
                    groups=3 * hidden_dim,
                    padding=1,
                    bias=False,
                ),
                gate_nonlinear_cfg=LazyConfig(nn.SiLU)(),
                gate_nonlinear_2_cfg=LazyConfig(nn.Sigmoid)(),
                pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim),
                output_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim),
                qk_norm_cfg=LazyConfig(L2Norm)(),
                use_rope=False,
            ),
        ),
        sequence_mixer_norm_cfg=norm_cfg,
        condition_mixer_cfg=LazyConfig(nn.Identity)(),
        condition_mixer_norm_cfg=LazyConfig(nn.Identity)(),
        mlp_cfg=LazyConfig(MLP)(
            dim=hidden_dim,
            activation="glu",
            expansion_factor=1.0,
            dropout_cfg=LazyConfig(nn.Dropout)(p=0.0),
        ),
        mlp_norm_cfg=norm_cfg,
        dropout_cfg=LazyConfig(nn.Dropout)(p=0.0),
    )


def build_model(num_blocks: int, patch_size: int) -> BlockStack:
    """Build a BlockStack matching the active_matter Hyena architecture."""
    seq_len = IMG_SIZE // patch_size
    block_cfg = make_block_cfg(HIDDEN_DIM, seq_len)
    blocks = [instantiate(block_cfg) for _ in range(num_blocks)]
    out_norm = RMSNorm(dim=HIDDEN_DIM)
    return BlockStack(blocks, out_norm)


# ── Correctness check ───────────────────────────────────────────────────────


def check_correctness(num_blocks: int, patch_size: int, batch_size: int = 2):
    """Verify that baseline and optimized produce identical outputs."""
    seq_len = IMG_SIZE // patch_size
    print(f"\n{'=' * 60}")
    print(f"Correctness check: p{patch_size} ({seq_len}x{seq_len}), batch={batch_size}")
    print(f"{'=' * 60}")

    seed = 42
    x = torch.randn(
        batch_size,
        seq_len,
        seq_len,
        HIDDEN_DIM,
        device="cuda",
        dtype=torch.float32,
    )

    results = {}
    for mode in ["baseline", "optimized"]:
        set_mode(mode)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model = build_model(num_blocks, patch_size).cuda().float()
        model.eval()
        with torch.no_grad():
            out = model(x)
        results[mode] = out.clone()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    diff = (results["baseline"] - results["optimized"]).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (results["baseline"].abs() + 1e-8)).mean().item()

    print(f"  Max  abs diff: {max_diff:.2e}")
    print(f"  Mean abs diff: {mean_diff:.2e}")
    print(f"  Mean rel diff: {rel_diff:.2e}")

    atol = 1e-5
    passed = max_diff < atol
    print(f"  Result: {'PASS' if passed else 'FAIL'} (atol={atol})")
    if not passed:
        print("  WARNING: outputs differ beyond tolerance!")
    return passed


# ── Benchmark ────────────────────────────────────────────────────────────────


def benchmark_one(
    model: nn.Module,
    batch_size: int,
    seq_h: int,
    seq_w: int,
    warmup: int,
    steps: int,
    dtype: torch.dtype,
) -> dict:
    """Run forward+backward, return it/s and peak memory."""
    x = torch.randn(batch_size, seq_h, seq_w, HIDDEN_DIM, device="cuda", dtype=dtype)

    # Warmup (includes torch.compile tracing)
    for _ in range(warmup):
        out = model(x)
        out.sum().backward()
        model.zero_grad(set_to_none=True)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(steps):
        out = model(x)
        out.sum().backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    it_per_sec = steps / elapsed

    del x
    return {"it_s": it_per_sec, "peak_mem_gb": peak_mem_gb}


def run_benchmarks(args):
    """Run the full benchmark sweep and print results."""
    compile_mode = "max-autotune-no-cudagraphs"
    dtype = torch.bfloat16
    results = []

    for patch_size in args.patch_sizes:
        seq_len = IMG_SIZE // patch_size
        tokens = seq_len * seq_len

        for mode in ["baseline", "optimized"]:
            set_mode(mode)
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()

            print(f"\n--- Building model: p{patch_size} ({seq_len}x{seq_len} = {tokens} tokens), mode={mode} ---")
            model = build_model(args.num_blocks, patch_size).cuda().to(dtype)
            compiled = torch.compile(model, mode=compile_mode)

            for batch_size in args.batch_sizes:
                label = f"p{patch_size:>2d}  b{batch_size:>3d}  {mode:<10s}"
                try:
                    res = benchmark_one(
                        compiled,
                        batch_size,
                        seq_len,
                        seq_len,
                        args.warmup,
                        args.steps,
                        dtype,
                    )
                    results.append(
                        {
                            "patch": patch_size,
                            "tokens": tokens,
                            "batch": batch_size,
                            "mode": mode,
                            **res,
                        }
                    )
                    print(f"  {label}  {res['it_s']:7.2f} it/s   {res['peak_mem_gb']:6.2f} GB")
                except torch.cuda.OutOfMemoryError:
                    results.append(
                        {
                            "patch": patch_size,
                            "tokens": tokens,
                            "batch": batch_size,
                            "mode": mode,
                            "it_s": None,
                            "peak_mem_gb": None,
                        }
                    )
                    print(f"  {label}  OOM")
                    gc.collect()
                    torch.cuda.empty_cache()

            del model, compiled
            gc.collect()
            torch.cuda.empty_cache()

    # ── Summary table ────────────────────────────────────────────────────
    print("\n")
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    header = f"{'Patch':>5s} {'Tokens':>7s} {'Batch':>5s} | {'baseline it/s':>14s} {'opt it/s':>10s} {'speedup':>8s} | {'base mem':>9s} {'opt mem':>9s} {'mem diff':>9s}"
    print(header)
    print("-" * 90)

    # Group results by (patch, batch)
    from collections import defaultdict

    grouped = defaultdict(dict)
    for r in results:
        key = (r["patch"], r["batch"])
        grouped[key][r["mode"]] = r

    for (patch, batch), modes in sorted(grouped.items()):
        base = modes.get("baseline", {})
        opt = modes.get("optimized", {})
        base_its = base.get("it_s")
        opt_its = opt.get("it_s")
        base_mem = base.get("peak_mem_gb")
        opt_mem = opt.get("peak_mem_gb")
        tokens = base.get("tokens", opt.get("tokens", "?"))

        base_its_s = f"{base_its:>14.2f}" if base_its else f"{'OOM':>14s}"
        opt_its_s = f"{opt_its:>10.2f}" if opt_its else f"{'OOM':>10s}"
        if base_its and opt_its:
            speedup = opt_its / base_its
            speedup_s = f"{speedup:>7.2f}x"
        else:
            speedup_s = f"{'—':>8s}"

        base_mem_s = f"{base_mem:>8.2f}G" if base_mem else f"{'OOM':>9s}"
        opt_mem_s = f"{opt_mem:>8.2f}G" if opt_mem else f"{'OOM':>9s}"
        if base_mem and opt_mem:
            mem_diff = opt_mem - base_mem
            mem_diff_s = f"{mem_diff:>+8.2f}G"
        else:
            mem_diff_s = f"{'—':>9s}"

        print(
            f"{patch:>5d} {tokens:>7d} {batch:>5d} | {base_its_s} {opt_its_s} {speedup_s} | {base_mem_s} {opt_mem_s} {mem_diff_s}"
        )

    print("=" * 90)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hyena norm permutation optimization")
    parser.add_argument(
        "--patch-sizes", type=int, nargs="+", default=[16, 4], help="Patch sizes to benchmark (default: 16 4)"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 32, 64],
        help="Batch sizes to sweep (default: 1 4 16 32 64)",
    )
    parser.add_argument("--num-blocks", type=int, default=12, help="Number of ResidualBlocks (default: 12)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations including compile (default: 10)")
    parser.add_argument("--steps", type=int, default=50, help="Timed iterations (default: 50)")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip the correctness check")
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Blocks: {args.num_blocks}, Hidden: {HIDDEN_DIM}, Img: {IMG_SIZE}")
    print(f"Patch sizes: {args.patch_sizes}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Warmup: {args.warmup}, Steps: {args.steps}")
    print("Compile mode: max-autotune-no-cudagraphs")
    print("Dtype: bfloat16")

    if not args.skip_correctness:
        all_pass = True
        for ps in args.patch_sizes:
            if not check_correctness(args.num_blocks, ps):
                all_pass = False
        if not all_pass:
            print("\nERROR: Correctness check failed! Aborting benchmark.")
            sys.exit(1)
        print("\nAll correctness checks passed.")

    run_benchmarks(args)


if __name__ == "__main__":
    main()
