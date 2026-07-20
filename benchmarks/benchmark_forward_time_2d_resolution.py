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

"""Forward-time vs 2D-resolution benchmark (the 2D analogue of Figure 1, right).

The paper's Figure 1 (right) plots single-operator forward time against **1D
sequence length** (4K->1M) for HyenaND (nSubQ) / attention / Mamba2, showing
attention's O(L^2) wall while HyenaND scales as O(L log L).  This script draws
the same comparison with a **2D context** on the x-axis: it sweeps a square
spatial resolution ``R = H = W`` so the token count ``L = R^2`` grows, and times
the forward pass of a single mixer *layer* at each resolution.

    R          64      128     256     512     1024
    L = R^2    4096    16384   65536   262144  1048576   (== 4K 16K 64K 256K 1M)

Three mixers are compared at a shared ``hidden_dim`` (channels-last
``[B, R, R, hidden_dim]`` input):

  * ``hyena``     -- ``QKVSequenceMixer`` wrapping ``Hyena`` with a 2D
    ``CKConvND`` on the ``subq_ops`` CUDA backend (the nSubQ fused FFT-conv).
  * ``attention`` -- ``QKVSequenceMixer`` wrapping ``Attention`` (SDPA /
    flash / cuDNN), with 2D axial RoPE.
  * ``mamba``     -- a **bare** ``Mamba`` (bidirectional Mamba2) that rasterizes
    the 2D grid into a 1D scan.  Not wrapped in ``QKVSequenceMixer`` (its
    ``forward`` takes a single tensor, not q/k/v).

The mixer config builders are reused from ``benchmark_patch_size_2d.py``.  We
time a single layer (not the 4-block ``ResidualNetwork`` that script builds):
it is faithful to Figure 1's operator-level timing, uses far less memory (so it
reaches R=1024), and removes the confounds of the residual net's projections,
MLPs and extra norms.

Output is a JSONL file (one row per ``(mixer, resolution)``); plotting is a
separate step (``scripts/visualization/visualize_forward_time_2d.py``) that
reads the JSONL, so the GPU run needs no matplotlib.

Local smoke test (any CUDA GPU, no ``subquadratic_ops_torch`` needed)::

    PYTHONPATH=. python benchmarks/benchmark_forward_time_2d_resolution.py \\
        --fft-backend torch_fft --no-compile --batch-size 1 \\
        --resolutions 8 16 32 --mixers hyena attention \\
        --num-warmup 2 --num-iters 3 --output /tmp/smoke_2d.jsonl

GB200 production run (fused nSubQ kernels, all three mixers)::

    PYTHONPATH=. python benchmarks/benchmark_forward_time_2d_resolution.py \\
        --fft-backend subq_ops --dtype bf16 --batch-size 1 --hidden-dim 256 \\
        --resolutions 64 128 256 512 1024 \\
        --output benchmarks/results/forward_time_2d_resolution.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


_BENCH_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCH_DIR.parent
for _p in (str(_BENCH_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the concrete mixer-config builders (backward-compatibly extended to take
# per-resolution kwargs).  ``benchmarks/`` has no __init__.py, hence the direct
# module import after the sys.path insert above.
from benchmark_patch_size_2d import (
    _attention_mixer_cfg,
    _hyena_mixer_cfg,
    _mamba_mixer_cfg,
)

from nvsubquadratic.lazy_config import instantiate


MIXER_CHOICES = ("attention", "hyena", "mamba")
DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ─── Module construction (fresh per resolution) ───────────────────────────────


def build_module(
    name: str,
    *,
    hidden_dim: int,
    resolution: int,
    fft_backend: str,
    grid_type: str,
    num_heads: int,
    attn_rope: bool,
    mamba_headdim: int,
    mamba_expand: int,
) -> torch.nn.Module:
    """Instantiate a single mixer layer sized for a square ``resolution`` grid.

    Only the resolution-dependent construction args change with ``resolution``:
    the SIREN kernel cache (``L_cache``) for Hyena and the RoPE tables for
    Attention.  Mamba is resolution-independent (it rasterizes at forward time).
    """
    if name == "hyena":
        cfg = _hyena_mixer_cfg(hidden_dim, fft_backend, canvas_size=resolution, grid_type=grid_type)
    elif name == "attention":
        head_dim = hidden_dim // num_heads
        use_rope = attn_rope and (head_dim % 4 == 0)
        if attn_rope and not use_rope:
            print(
                f"   [warn] disabling 2D RoPE: head_dim={head_dim} "
                f"(hidden_dim={hidden_dim} / num_heads={num_heads}) is not divisible by 4.",
                flush=True,
            )
        cfg = _attention_mixer_cfg(
            hidden_dim,
            num_heads=num_heads,
            use_rope=use_rope,
            rope_spatial_dims=(resolution, resolution) if use_rope else None,
        )
    elif name == "mamba":
        cfg = _mamba_mixer_cfg(
            hidden_dim,
            headdim=mamba_headdim,
            expand=mamba_expand,
            bidirectional=True,
        )
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown mixer '{name}'")

    return instantiate(cfg)


# ─── Timing ───────────────────────────────────────────────────────────────────


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def time_forward(
    name: str,
    resolution: int,
    *,
    hidden_dim: int,
    batch_size: int,
    dtype: torch.dtype,
    num_warmup: int,
    num_iters: int,
    compile_mode: str | None,
    fft_backend: str,
    grid_type: str,
    num_heads: int,
    attn_rope: bool,
    mamba_headdim: int,
    mamba_expand: int,
    max_seconds: float,
    device: torch.device,
) -> dict[str, Any]:
    """Build the layer, run a forward, and return a result dict.

    Returns keys ``status`` (``ok``/``oom``/``error``/``timeout``), ``ms`` and
    ``mem_gb`` (``None`` unless the point produced a usable timing).  A single
    warmup forward is wall-timed first; if it already exceeds ``max_seconds`` the
    point is marked ``timeout`` and the (expensive) timed loop is skipped, so the
    worst case is one slow forward rather than ``num_iters`` of them.
    """
    if name == "hyena" and compile_mode is not None and fft_backend == "torch_fft":
        # Only the torch.fft path needs this flag; the subq_ops custom op is
        # compile-safe on its own.
        import nvsubquadratic.ops.fftconv as _fftconv

        _fftconv.COMPILE_COMPATIBLE = True

    module = None
    x = None
    try:
        module = (
            build_module(
                name,
                hidden_dim=hidden_dim,
                resolution=resolution,
                fft_backend=fft_backend,
                grid_type=grid_type,
                num_heads=num_heads,
                attn_rope=attn_rope,
                mamba_headdim=mamba_headdim,
                mamba_expand=mamba_expand,
            )
            .to(device)
            .eval()
        )

        if compile_mode is not None:
            module = torch.compile(module, mode=compile_mode)

        x = torch.randn(batch_size, resolution, resolution, hidden_dim, device=device, dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats(device)

        # ── Single wall-timed forward (compile + timeout guard) ──────────────
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            _ = module(x)
        torch.cuda.synchronize(device)
        first_forward_s = time.perf_counter() - t0

        if first_forward_s > max_seconds:
            return {
                "status": "timeout",
                "ms": first_forward_s * 1000.0,
                "mem_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            }

        # ── Remaining warmup ──────────────────────────────────────────────────
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for _ in range(max(0, num_warmup - 1)):
                _ = module(x)
        torch.cuda.synchronize(device)

        # ── Timed iterations (CUDA events) ────────────────────────────────────
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for _ in range(num_iters):
                _ = module(x)
        end.record()
        torch.cuda.synchronize(device)

        return {
            "status": "ok",
            "ms": start.elapsed_time(end) / num_iters,
            "mem_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
    except Exception as exc:
        status = "oom" if _is_oom(exc) else "error"
        return {"status": status, "ms": None, "mem_gb": None, "detail": repr(exc)}
    finally:
        del module, x
        gc.collect()
        torch.cuda.empty_cache()
        if compile_mode is not None and hasattr(torch, "_dynamo"):
            torch._dynamo.reset()


def _predicted_ms(name: str, seq_len: int, last_seq_len: int, last_ms: float) -> float:
    """Extrapolate step time to ``seq_len`` from the last successful point.

    Uses each mixer's asymptotic law so we can *skip* (rather than launch and
    block on) points that will clearly exceed the time budget: attention grows
    as O(L^2), Hyena as O(L log L), Mamba as O(L).
    """
    r = seq_len / last_seq_len
    if name == "attention":
        return last_ms * r * r
    if name == "hyena":
        return last_ms * r * (math.log2(seq_len) / math.log2(last_seq_len))
    return last_ms * r  # mamba (linear) and any other operator


# ─── Main ─────────────────────────────────────────────────────────────────────


def _validate(args: argparse.Namespace) -> None:
    if "attention" in args.mixers:
        if args.hidden_dim % args.num_heads != 0:
            raise SystemExit(f"hidden_dim={args.hidden_dim} not divisible by num_heads={args.num_heads}.")
    if "mamba" in args.mixers:
        d_inner = args.hidden_dim * args.mamba_expand
        if d_inner % args.mamba_headdim != 0:
            raise SystemExit(
                f"Mamba2 d_inner=hidden_dim*expand={d_inner} not divisible by mamba_headdim={args.mamba_headdim}."
            )
    if args.fft_backend == "subq_ops" and "hyena" in args.mixers:
        odd = [r for r in args.resolutions if r % 2 != 0]
        if odd:
            raise SystemExit(f"subq_ops requires even resolutions; got odd {odd}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024],
        help="Square spatial resolutions R (H=W). Token count L=R^2. Swept ascending.",
    )
    parser.add_argument(
        "--mixers",
        nargs="+",
        choices=MIXER_CHOICES,
        default=list(MIXER_CHOICES),
        help="Subset of mixers to benchmark.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the timed forward pass.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Shared channel dim across all mixers.")
    parser.add_argument("--num-heads", type=int, default=8, help="Attention heads (head_dim=hidden_dim/num_heads).")
    parser.add_argument("--mamba-headdim", type=int, default=64, help="Mamba2 head dim.")
    parser.add_argument("--mamba-expand", type=int, default=2, help="Mamba2 expansion factor.")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations (also covers compile).")
    parser.add_argument("--num-iters", type=int, default=30, help="Timed iterations.")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16", help="Autocast dtype.")
    parser.add_argument(
        "--fft-backend",
        choices=["subq_ops", "torch_fft"],
        default="subq_ops",
        help="FFT-conv backend for Hyena. Use 'torch_fft' on hosts without the subq_ops CUDA kernels.",
    )
    parser.add_argument(
        "--grid-type",
        choices=["double", "single"],
        default="double",
        help="Hyena SIREN kernel grid. 'double' = non-circular / zero-padded (kernel size ~2R per axis); "
        "'single' = circular (kernel size ~R). The materialized kernel is ~4x smaller with 'single', which "
        "helps memory at high resolution. Both size the 2D FFT to ~2*next_pow2(R) from the input, so R=8192 "
        "(~16K-point FFT) sits near the cuFFTDx size ceiling either way.",
    )
    rope = parser.add_mutually_exclusive_group()
    rope.add_argument("--attn-rope", dest="attn_rope", action="store_true", help="Enable 2D axial RoPE (default).")
    rope.add_argument("--no-attn-rope", dest="attn_rope", action="store_false", help="Disable RoPE.")
    parser.set_defaults(attn_rope=True)
    compile_grp = parser.add_mutually_exclusive_group()
    compile_grp.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        help="torch.compile mode (e.g. 'max-autotune-no-cudagraphs'). Default: eager.",
    )
    compile_grp.add_argument("--no-compile", action="store_true", help="Disable torch.compile (default).")
    parser.add_argument(
        "--max-seconds-per-point",
        type=float,
        default=120.0,
        help="If a single forward exceeds this, mark the point 'timeout' and skip larger resolutions "
        "for that mixer (attention's O(L^2) explosion).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/forward_time_2d_resolution.jsonl",
        help="Output JSONL path (one row per mixer/resolution).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for the forward-time benchmark.")
    _validate(args)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    dtype = DTYPE_MAP[args.dtype]
    compile_mode = None if args.no_compile else args.compile_mode
    resolutions = sorted(args.resolutions)
    device_name = torch.cuda.get_device_name(device)

    print(f"Device: {device_name}")
    print(
        f"Settings: batch_size={args.batch_size} hidden_dim={args.hidden_dim} dtype={args.dtype} "
        f"compile={compile_mode} fft_backend={args.fft_backend} grid_type={args.grid_type} rope={args.attn_rope} "
        f"warmup={args.num_warmup} timed={args.num_iters} max_s={args.max_seconds_per_point}"
    )
    print(f"Mixers: {args.mixers}    Resolutions: {resolutions}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # rows[mixer][seq_len] = result dict, for the summary table
    rows: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in args.mixers}

    with out_path.open("w") as fh:
        for mixer in args.mixers:
            last_ok: tuple[int, float] | None = None  # (seq_len, ms) of last ok/timeout point
            for R in resolutions:
                seq_len = R * R
                label = f"[{mixer:>9s}  R={R:>4d}  L={seq_len:>9d}]"

                # Predictive skip: never launch a point that will clearly blow the budget.
                if last_ok is not None:
                    pred_ms = _predicted_ms(mixer, seq_len, last_ok[0], last_ok[1])
                    if pred_ms > args.max_seconds_per_point * 1000.0:
                        print(
                            f"\n{label}\n   [skip] predicted ~{pred_ms / 1000.0:.0f}s > budget; marking timeout.",
                            flush=True,
                        )
                        result = {"status": "timeout", "ms": None, "mem_gb": None}
                        rows[mixer][seq_len] = result
                        record = {
                            "mixer": mixer,
                            "resolution": R,
                            "seq_len": seq_len,
                            "backend": args.fft_backend if mixer == "hyena" else None,
                            "batch_size": args.batch_size,
                            "hidden_dim": args.hidden_dim,
                            "dtype": args.dtype,
                            "device": device_name,
                            **result,
                        }
                        fh.write(json.dumps(record) + "\n")
                        fh.flush()
                        continue

                print(f"\n{label}", flush=True)
                t0 = time.perf_counter()
                result = time_forward(
                    mixer,
                    R,
                    hidden_dim=args.hidden_dim,
                    batch_size=args.batch_size,
                    dtype=dtype,
                    num_warmup=args.num_warmup,
                    num_iters=args.num_iters,
                    compile_mode=compile_mode,
                    fft_backend=args.fft_backend,
                    grid_type=args.grid_type,
                    num_heads=args.num_heads,
                    attn_rope=args.attn_rope,
                    mamba_headdim=args.mamba_headdim,
                    mamba_expand=args.mamba_expand,
                    max_seconds=args.max_seconds_per_point,
                    device=device,
                )
                wall = time.perf_counter() - t0

                ms, mem = result.get("ms"), result.get("mem_gb")
                if result["status"] == "ok":
                    print(f"   ms/fwd = {ms:9.3f}  |  peak mem = {mem:6.2f} GB  |  wall = {wall:5.1f}s", flush=True)
                    last_ok = (seq_len, ms)
                elif result["status"] == "timeout":
                    shown = f"{ms / 1000.0:.1f}s" if ms is not None else "n/a"
                    print(f"   [timeout] single forward = {shown}  |  wall = {wall:5.1f}s", flush=True)
                    last_ok = (seq_len, ms) if ms is not None else last_ok
                else:
                    print(f"   [{result['status']}] {result.get('detail', '')}  |  wall = {wall:5.1f}s", flush=True)

                rows[mixer][seq_len] = result
                record = {
                    "mixer": mixer,
                    "resolution": R,
                    "seq_len": seq_len,
                    "backend": args.fft_backend if mixer == "hyena" else None,
                    "batch_size": args.batch_size,
                    "hidden_dim": args.hidden_dim,
                    "dtype": args.dtype,
                    "device": device_name,
                    "status": result["status"],
                    "ms": ms,
                    "mem_gb": mem,
                }
                if "detail" in result:
                    record["detail"] = result["detail"]
                fh.write(json.dumps(record) + "\n")
                fh.flush()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("Forward-time summary (ms/fwd, lower is better; 'oom'/'timeout'/'error' otherwise)")
    print(f"{'=' * 78}")
    header = f"{'R':>6s} {'L=R^2':>10s}  " + "  ".join(f"{m:>14s}" for m in args.mixers)
    print(header)
    print("-" * len(header))
    for R in resolutions:
        seq_len = R * R
        row = f"{R:>6d} {seq_len:>10d}  "
        for m in args.mixers:
            res = rows[m].get(seq_len)
            if res is None:
                cell = "-"
            elif res["status"] == "ok":
                cell = f"{res['ms']:.3f}"
            else:
                cell = res["status"]
            row += f"{cell:>14s}  "
        print(row.rstrip())

    print(f"\n[done] wrote {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
