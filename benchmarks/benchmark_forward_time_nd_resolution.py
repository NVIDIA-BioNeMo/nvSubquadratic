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

"""Forward-time vs resolution benchmark in 1D / 2D / 3D (the ND analogue of Figure 1, right).

Select the dimensionality with ``--data-dim {1,2,3}``; the input is
``[B, R, ...(N axes)..., C]`` and the token count is ``L = R^N``. Hyena is causal
in 1D (the fused genomics kernel) and non-causal in 2D/3D; 3D falls back to
torch_fft (subq_ops has no 3D kernel). The 2D case (the default) is described below.

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
  * ``attention`` -- ``QKVSequenceMixer`` wrapping ``Attention`` with the default
    SDPA kernel (PyTorch auto-selects flash / cuDNN / fallback), 2D axial RoPE.
  * ``flex``      -- same ``Attention`` layer with the compiled FlexAttention
    kernel (``torch.nn.attention.flex_attention``).  Requires head_dim >= 16.
  * ``fa4``       -- same ``Attention`` layer with FlashAttention-4 (the external
    ``flash-attn-4`` CuTe-DSL wheel).  Requires head_dim >= 16 and a Hopper/
    Blackwell GPU; if ``flash_attn`` is absent the points show ``unavailable``.
  * ``mamba``     -- a **bare** ``Mamba`` (bidirectional Mamba2) that rasterizes
    the 2D grid into a 1D scan.  Not wrapped in ``QKVSequenceMixer`` (its
    ``forward`` takes a single tensor, not q/k/v).

``attention``/``flex``/``fa4`` are three interchangeable attention *kernels* on
one shared q/k/v + RoPE path; run them together for an apples-to-apples flash
comparison (needs head_dim >= 16, so a wider ``hidden_dim`` than the tiny-width
16M-reach sweep).

The mixer config builders are reused from ``benchmark_patch_size_2d.py``.  We
time a single layer (not the 4-block ``ResidualNetwork`` that script builds):
it is faithful to Figure 1's operator-level timing, uses far less memory (so it
reaches R=1024), and removes the confounds of the residual net's projections,
MLPs and extra norms.

Output is a JSONL file (one row per ``(mixer, resolution)``); plotting is a
separate step (``scripts/visualization/visualize_forward_time_nd.py``) that
reads the JSONL, so the GPU run needs no matplotlib.

Local smoke test (any CUDA GPU, no ``subquadratic_ops_torch`` needed)::

    PYTHONPATH=. python benchmarks/benchmark_forward_time_nd_resolution.py \\
        --fft-backend torch_fft --no-compile --batch-size 1 \\
        --resolutions 8 16 32 --mixers hyena attention \\
        --num-warmup 2 --num-iters 3 --output /tmp/smoke_2d.jsonl

GB200 production run (fused nSubQ kernels, all three mixers)::

    PYTHONPATH=. python benchmarks/benchmark_forward_time_nd_resolution.py \\
        --fft-backend subq_ops --dtype bf16 --batch-size 1 --hidden-dim 256 \\
        --resolutions 64 128 256 512 1024 \\
        --output benchmarks/results/forward_time.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
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


MIXER_CHOICES = ("attention", "flex", "fa4", "hyena", "mamba")
# Attention kernel per mixer key: SDPA (auto cuDNN/flash), compiled FlexAttention,
# or FlashAttention-4 (external flash_attn). All share the same q/k/v + RoPE path.
_ATTN_IMPL = {"attention": "sdpa", "flex": "flex", "fa4": "fa4"}
DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# 2D axial RoPE needs head_dim divisible by 4; 1D by 2; 3D by 6.
_ROPE_DIVISOR = {1: 2, 2: 4, 3: 6}


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
    data_dim: int,
) -> torch.nn.Module:
    """Instantiate a single mixer layer sized for a ``resolution`` grid in ``data_dim`` dims.

    Only the resolution-dependent construction args change with ``resolution``:
    the SIREN kernel cache (``L_cache``) for Hyena and the RoPE tables for
    Attention.  Mamba is resolution-independent (it rasterizes at forward time).
    Hyena is causal in 1D (the fused genomics path) and non-causal in 2D/3D.
    """
    if name == "hyena":
        cfg = _hyena_mixer_cfg(
            hidden_dim,
            fft_backend,
            canvas_size=resolution,
            grid_type=grid_type,
            data_dim=data_dim,
            is_causal=(data_dim == 1),
        )
    elif name in _ATTN_IMPL:  # attention (sdpa) / flex / fa4 — shared q/k/v + RoPE path
        head_dim = hidden_dim // num_heads
        div = _ROPE_DIVISOR[data_dim]  # 1D:2, 2D:4, 3D:6
        use_rope = attn_rope and (head_dim % div == 0)
        if attn_rope and not use_rope:
            print(
                f"   [warn] disabling {data_dim}D RoPE: head_dim={head_dim} "
                f"(hidden_dim={hidden_dim} / num_heads={num_heads}) not divisible by {div}.",
                flush=True,
            )
        cfg = _attention_mixer_cfg(
            hidden_dim,
            num_heads=num_heads,
            use_rope=use_rope,
            rope_spatial_dims=(resolution,) * data_dim if use_rope else None,
            attn_impl=_ATTN_IMPL[name],
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
    data_dim: int,
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
                data_dim=data_dim,
            )
            .to(device)
            .eval()
        )

        if compile_mode is not None:
            module = torch.compile(module, mode=compile_mode)

        x = torch.randn(batch_size, *([resolution] * data_dim), hidden_dim, device=device, dtype=torch.float32)
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

        # ── Adaptive iteration counts ─────────────────────────────────────────
        # Per-forward cost spans ~1e-3 s (small hyena) to minutes (attention at
        # multi-M tokens). A fixed count would be noisy for fast ops and run for
        # hours on slow ones (30 iters x a 5-min forward = 2.5 h for one point).
        # Scale both warmup and timed counts to the measured cost, keeping each
        # point within the per-point time budget. num_warmup/num_iters are caps.
        target_timed_s = 5.0
        min_timed_iters = 3
        n_timed = round(target_timed_s / max(first_forward_s, 1e-9))
        n_timed = max(min_timed_iters, min(num_iters, n_timed))
        # Never let the timed loop exceed the budget for slow ops.
        n_timed = min(n_timed, max(1, int(max_seconds / max(first_forward_s, 1e-9))))
        # Extra warmup only pays off for cheap forwards; skip it for slow ones.
        extra_warmup = max(0, num_warmup - 1) if first_forward_s < 1.0 else 0

        # ── Remaining warmup ──────────────────────────────────────────────────
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for _ in range(extra_warmup):
                _ = module(x)
        torch.cuda.synchronize(device)

        # ── Timed iterations (CUDA events) ────────────────────────────────────
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            for _ in range(n_timed):
                _ = module(x)
        end.record()
        torch.cuda.synchronize(device)

        return {
            "status": "ok",
            "ms": start.elapsed_time(end) / n_timed,
            "mem_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "iters": n_timed,
        }
    except Exception as exc:
        if isinstance(exc, ImportError):  # e.g. mamba_ssm not installed — not a wall
            status = "unavailable"
        elif _is_oom(exc):
            status = "oom"
        else:
            status = "error"
        # Print the full traceback to the log for genuine errors (not clean
        # OOM/unavailable) so failures like Mamba's construction error are
        # diagnosable without a rerun. Only the repr goes into the JSONL.
        if status == "error":
            traceback.print_exc(file=sys.stdout)
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
    # subq_ops (1D/2D only) needs even resolutions; 3D uses torch_fft so it is exempt.
    if args.fft_backend.startswith("subq_ops") and args.data_dim in (1, 2) and "hyena" in args.mixers:
        odd = [r for r in args.resolutions if r % 2 != 0]
        if odd:
            raise SystemExit(f"subq_ops requires even resolutions; got odd {odd}.")


# The fused 2D kernel's largest FFT tile is 128 and it requires
# max(X, Y) <= fft_size // 2, so it tops out at a 64-per-axis grid. Kept as a
# literal (not an import of fused_fftconv2d_max_spatial) so the sweep can be
# planned on a login node without subquadratic_ops_torch installed.
_FUSED_MAX_SPATIAL = 64


def resolve_hyena_backend(requested: str, data_dim: int, resolution: int) -> str:
    """Effective Hyena FFT backend for one point of the sweep.

    The subq_ops backends do not cover the whole (data_dim, resolution) grid, so a
    single ``--fft-backend`` choice has to degrade per point rather than fail the
    run.  Each fallback goes to the fastest backend that *does* cover the point:

    * 3D — no CUDA FFT-conv kernel at all, so ``torch_fft``.
    * ``subq_ops_fused`` in 1D/3D — no fused kernel outside 2D, so ``subq_ops``
      (which in turn becomes ``torch_fft`` in 3D).
    * ``subq_ops_fused`` in 2D above ``_FUSED_MAX_SPATIAL`` — outside the fused
      kernel's tile range, so ``subq_ops``.

    Returns:
        The backend to construct this point's Hyena layer with.
    """
    if data_dim == 3 and requested.startswith("subq_ops"):
        return "torch_fft"
    if requested == "subq_ops_fused":
        if data_dim != 2:
            return "subq_ops"
        if resolution > _FUSED_MAX_SPATIAL:
            return "subq_ops"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024],
        help="Per-axis resolution R. Token count L=R^data_dim. Swept ascending.",
    )
    parser.add_argument(
        "--data-dim",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="Spatial dimensionality N. Input is [B, R,...(N)..., C], L=R^N. Hyena is causal in 1D "
        "(the fused genomics kernel) and non-causal in 2D/3D; 3D auto-falls back to torch_fft "
        "(subq_ops has no 3D kernel).",
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
        choices=["subq_ops", "subq_ops_fused", "torch_fft"],
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
        "--disable-cudnn",
        action="store_true",
        help="Turn off cuDNN (Conv2d native fallback; SDPA avoids the cuDNN backend). "
        "Workaround for images where cuDNN fails to initialize (CUDNN_STATUS_NOT_INITIALIZED).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/forward_time.jsonl",
        help="Output JSONL path (one row per mixer/resolution).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for the forward-time benchmark.")
    _validate(args)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        print("[cudnn] disabled (Conv2d native fallback; SDPA flash/mem-efficient).", flush=True)

    dtype = DTYPE_MAP[args.dtype]
    compile_mode = None if args.no_compile else args.compile_mode
    resolutions = sorted(args.resolutions)
    device_name = torch.cuda.get_device_name(device)
    data_dim = args.data_dim
    # The requested backend does not cover every point; resolve it per resolution
    # and report the plan up front so the coverage is visible before the sweep runs.
    backend_at = {R: resolve_hyena_backend(args.fft_backend, data_dim, R) for R in resolutions}
    if "hyena" in args.mixers:
        for eff in dict.fromkeys(backend_at.values()):
            if eff == args.fft_backend:
                continue
            at = [R for R in resolutions if backend_at[R] == eff]
            print(f"[note] Hyena fft_backend '{args.fft_backend}' -> '{eff}' at R={at} (unsupported there).")

    print(f"Device: {device_name}")
    backend_desc = "/".join(dict.fromkeys(backend_at.values()))
    print(
        f"Settings: data_dim={data_dim} batch_size={args.batch_size} hidden_dim={args.hidden_dim} dtype={args.dtype} "
        f"compile={compile_mode} fft_backend={backend_desc} grid_type={args.grid_type} rope={args.attn_rope} "
        f"warmup={args.num_warmup} timed={args.num_iters} max_s={args.max_seconds_per_point}"
    )
    print(f"Mixers: {args.mixers}    Resolutions: {resolutions}")

    # Attention-kernel eligibility. SDPA (auto cuDNN / flash) needs head_dim % 8 == 0
    # and is *optimized* for 64/128; below a multiple of 8 it silently falls back to a
    # weak math / mem-efficient path. FlexAttention and FA4 additionally *require*
    # head_dim >= 16 — below that they raise, so those points error out (rather than
    # degrade) at a too-small head_dim.
    attn_mixers = [m for m in args.mixers if m in _ATTN_IMPL]
    if attn_mixers:
        head_dim = args.hidden_dim // args.num_heads
        print(
            f"[attn] kernels={attn_mixers}  head_dim={head_dim} (hidden {args.hidden_dim} / heads {args.num_heads})",
            flush=True,
        )
        if head_dim % 8 != 0:
            print(
                f"[attn] head_dim={head_dim} is NOT a multiple of 8 → SDPA flash/cuDNN INELIGIBLE "
                f"(falls back to math/mem-efficient, a weak non-flash baseline).",
                flush=True,
            )
        elif head_dim < 64:
            print(
                f"[attn] head_dim={head_dim}: flash eligible but BELOW its optimized regime "
                f"(flash/FA4 tuned for head_dim 64/128).",
                flush=True,
            )
        else:
            print(
                f"[attn] head_dim={head_dim} → flash/cuDNN eligible; SDPA auto-selects the fused kernel.", flush=True
            )
        needs16 = sorted({"flex", "fa4"} & set(attn_mixers))
        if needs16 and head_dim < 16:
            print(
                f"[attn] head_dim={head_dim} < 16 → {needs16} will ERROR "
                f"(FlexAttention/FA4 require head_dim >= 16); those points are marked 'error'. "
                f"Raise hidden_dim or lower num_heads.",
                flush=True,
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # rows[mixer][seq_len] = result dict, for the summary table
    rows: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in args.mixers}

    with out_path.open("w") as fh:
        for mixer in args.mixers:
            last_ok: tuple[int, float] | None = None  # (seq_len, ms) of last ok/timeout point
            for R in resolutions:
                seq_len = R**data_dim
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
                            "data_dim": data_dim,
                            "backend": backend_at[R] if mixer == "hyena" else None,
                            "batch_size": args.batch_size,
                            "hidden_dim": args.hidden_dim,
                            "num_heads": args.num_heads,
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
                    fft_backend=backend_at[R],
                    grid_type=args.grid_type,
                    num_heads=args.num_heads,
                    attn_rope=args.attn_rope,
                    mamba_headdim=args.mamba_headdim,
                    mamba_expand=args.mamba_expand,
                    data_dim=data_dim,
                    max_seconds=args.max_seconds_per_point,
                    device=device,
                )
                wall = time.perf_counter() - t0

                ms, mem = result.get("ms"), result.get("mem_gb")
                if result["status"] == "ok":
                    n = result.get("iters", "?")
                    print(
                        f"   ms/fwd = {ms:9.3f} (n={n})  |  peak mem = {mem:6.2f} GB  |  wall = {wall:5.1f}s",
                        flush=True,
                    )
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
                    "data_dim": data_dim,
                    "backend": backend_at[R] if mixer == "hyena" else None,
                    "batch_size": args.batch_size,
                    "hidden_dim": args.hidden_dim,
                    "num_heads": args.num_heads,
                    "dtype": args.dtype,
                    "device": device_name,
                    "status": result["status"],
                    "ms": ms,
                    "mem_gb": mem,
                    "iters": result.get("iters"),
                }
                if "detail" in result:
                    record["detail"] = result["detail"]
                fh.write(json.dumps(record) + "\n")
                fh.flush()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("Forward-time summary (ms/fwd, lower is better; 'oom'/'timeout'/'error' otherwise)")
    print(f"{'=' * 78}")
    header = f"{'R':>6s} {f'L=R^{data_dim}':>10s}  " + "  ".join(f"{m:>14s}" for m in args.mixers)
    print(header)
    print("-" * len(header))
    for R in resolutions:
        seq_len = R**data_dim
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
