# Hyena Permutation Optimization — v2 Tracker

**Branch**: `dwromero/hyena-permutation-opt` (based on `wessels/well-v2-base-configs`)

## Summary

**Conclusion: The Phase 1+2 permutation optimizations are counterproductive
when `torch.compile` is in use.**  The compiler already fuses and optimizes
permutation operations; our manual elimination + required `@torch.compiler.disable`
decorators actually *hurt* performance by introducing graph breaks.

The v1 tracker showed apparent speedups (up to 53%), but these were artifacts
of **insufficient warmup** (200-300 iterations).  With longer runs (2000 iters,
5+ epochs), `torch.compile(mode="max-autotune-no-cudagraphs")` reaches a much
higher steady-state speed on the baseline — surpassing the "optimized" variants.

______________________________________________________________________

## v2 Benchmark Setup (improvements over v1)

| Parameter       | v1                           | v2                              |
| --------------- | ---------------------------- | ------------------------------- |
| Iterations      | 200-300                      | **2000** (5-10 epochs)          |
| Data location   | Network filesystem           | **Local NVMe** (`/scratch`)     |
| Triton cache    | Stale caches on disk         | **Clean** (old caches removed)  |
| Concurrent jobs | Sequential                   | Matched (both run concurrently) |
| Node            | Various                      | **b65c909e-01** (pinned)        |
| GPU             | 1× H100 80 GB                | Same                            |
| Compile mode    | `max-autotune-no-cudagraphs` | Same                            |
| Dataset         | `active_matter`              | Same                            |

______________________________________________________________________

## Decorator variants tested

The Phase 1 optimization requires `_apply_norm_bchw` to be excluded from
`torch.compile` to avoid an `InductorError: KeyError 'complex64'` when the
compiler tries to fuse norm ops with FFT ops.  Three approaches were tested:

| Decorator                                  | Behavior                                  | Result                                      |
| ------------------------------------------ | ----------------------------------------- | ------------------------------------------- |
| None                                       | Full compilation, norms fuse with FFT     | **Crashes** with `complex64` error          |
| `@torch.compiler.disable(recursive=False)` | Dispatch runs eager; inner norms compiled | **36 graph breaks**, compiles but slower    |
| `@torch.compiler.disable`                  | Entire function in eager mode             | 1 opaque call per norm, fewest graph breaks |

______________________________________________________________________

## v2 Results — p16 b32

| Config                               | Peak it/s | vs Baseline | Solo?   | v_num | SLURM Job |
| ------------------------------------ | --------- | ----------- | ------- | ----- | --------- |
| **Baseline** (no changes)            | **5.07**  | 1.00x       | No      | 6fa0  | 39381     |
| Phase 1+2, `recursive=False`         | 4.27      | 0.84x       | No      | jyca  | 39385     |
| Phase 1+2, `@torch.compiler.disable` | 4.53      | 0.89x       | **Yes** | 2l7j  | 39388     |

**The baseline is fastest.**  The optimized variants are 11-16% slower even
though the `recursive=True` variant ran solo (no contention from concurrent p4).

## v2 Results — p4 b64

| Config                       | Peak it/s | vs Baseline | Solo? | v_num | SLURM Job |
| ---------------------------- | --------- | ----------- | ----- | ----- | --------- |
| **Baseline** (no changes)    | **2.21**  | 1.00x       | No    | z8au  | 39382     |
| Phase 1+2, `recursive=False` | 1.78      | 0.81x       | No    | z4rb  | 39386     |

**The baseline is fastest here too.**  19% slower with the "optimization."

______________________________________________________________________

## Why the v1 results were misleading

1. **Insufficient warmup**: `torch.compile(mode="max-autotune-no-cudagraphs")`
   needs hundreds of iterations to fully warm up.  The first ~100 steps include
   Triton autotuning, kernel compilation, and JIT overhead.

   - v1 baseline p16: 3.08 it/s (300 iters, not stabilized)
   - v2 baseline p16: **5.07 it/s** (2000 iters, fully stabilized) — **+65%**

1. **The "speedup" was faster warmup**: The optimized code reached its lower
   steady state faster because `@torch.compiler.disable` on norms reduced the
   amount of code the compiler needed to JIT.  This produced higher *cumulative*
   and *apparent steady-state* numbers at 200-300 steps, but the baseline
   eventually surpasses it.

1. **Compiler already optimizes permutations**: `torch.compile` with
   `max-autotune` can fuse `movedim`, `reshape`, and `rearrange` operations
   into surrounding kernels.  Manually eliminating them and replacing with
   compiler-opaque eager functions actually removes optimization opportunities.

______________________________________________________________________

## Impact of data staging

Pre-staging `active_matter` (49 GB) to local NVMe (`/scratch/dwromero`) vs
reading from the network filesystem improved data loading speed, especially
visible in the first epoch where the pipeline fills up.  All v2 runs used
local staging.

______________________________________________________________________

## Recommendation

1. **Do not merge the Phase 1+2 changes** for `torch.compile` workloads.
1. The changes *might* still help in **eager mode** (without compile), but
   the current workflow always uses `torch.compile`.
1. **Data staging** (`local_staging_dir=/scratch/dwromero`) should be enabled
   for all benchmarks and training runs — this is a genuine improvement
   independent of the permutation optimization.
1. Further investigation could focus on:
   - Filing a PyTorch bug for the `complex64` Inductor fusion issue
   - Profiling to understand what the compiler does with the baseline's
     movedim operations (likely fuses them away)
   - Phase 3 changes (full BCHW block loop) which could avoid needing
     `@torch.compiler.disable` entirely — but the evidence suggests the
     compiler handles this well already.

______________________________________________________________________

## Full v1 → v2 speed comparison

| Config        | v1 it/s (300 iters) | v2 it/s (2000 iters) | Change     |
| ------------- | ------------------- | -------------------- | ---------- |
| Baseline p16  | 3.08                | **5.07**             | +65%       |
| Phase 1+2 p16 | 4.71                | **4.27-4.53**        | -4% to -9% |
| Baseline p4   | ~1.7                | **2.21**             | +30%       |
| Phase 1+2 p4  | ~1.7                | **1.78**             | +5%        |

______________________________________________________________________

## WandB v2 runs

| SLURM Job | v_num | Config                              |
| --------- | ----- | ----------------------------------- |
| 39381     | 6fa0  | Baseline p16 b32                    |
| 39382     | z8au  | Baseline p4 b64                     |
| 39385     | jyca  | Phase 1+2 p16 b32 (recursive=False) |
| 39386     | z4rb  | Phase 1+2 p4 b64 (recursive=False)  |
| 39388     | 2l7j  | Phase 1+2 p16 b32 (recursive=True)  |
