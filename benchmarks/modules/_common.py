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

"""Shared scaffolding for the TriHyena benchmark.

CLI arguments, the pair-tensor workload, CUDA-event timing of forward / backward /
training step with an L2 flush before every timed region, caching-allocator memory
bookkeeping (resident peak, per-call working set, warm-up / graph-capture footprint;
see ``README.md`` "Memory columns"), and the (seq_len x arch) sweep that stops an
arch at its first OOM and writes the CSV incrementally.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import BoxModulationND, GaussianModulationND
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.tri_hyena import TriangleHyena


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DEFAULT_SEQ_LENS = [128, 256, 384, 512, 1024, 2048]
METRIC_FIELDS = [
    "param_count",
    "setup_ms",
    "fwd_ms",
    "bwd_ms",
    "step_ms",
    "fwd_peak_mb",
    "fwd_peak_delta_mb",
    "step_peak_mb",
    "step_peak_delta_mb",
    "act_mb",
    "setup_peak_delta_mb",
    "setup_peak_reserved_mb",
    "steady_reserved_delta_mb",
]


def add_common_args(parser: argparse.ArgumentParser, *, archs: list[str], csv_default: str) -> None:
    """Register the benchmark command-line arguments."""
    parser.add_argument("--seq-lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128, help="Pair channel width C.")
    parser.add_argument("--max-offset", type=int, default=256, help="Hard window half-width in taps.")
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument(
        "--warmup-seconds", type=float, default=1.0, help="Keep warming up until the GPU clocks settle."
    )
    parser.add_argument("--n-bench", type=int, default=10)
    parser.add_argument("--no-flush-l2", dest="flush_l2", action="store_false")
    parser.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument(
        "--right-pad-frac", type=float, default=0.125, help="Fraction of each sample padded at the end."
    )
    parser.add_argument("--archs", nargs="+", default=archs, choices=archs)
    parser.add_argument("--csv", default=csv_default)


def tri_hyena_mixer_cfg(
    hidden_dim: int,
    *,
    window: str,
    max_offset: int,
    fft_backend: str,
    short_conv_backend: str,
    start_node: bool,
    L_cache: int,
    use_shortcut: bool = False,
    right_padding: bool = False,
) -> LazyConfig:
    """``QKVSequenceMixer(TriangleHyena)`` config with the paper's SIREN sizes and no shortcut by default."""
    if window == "box":
        mask_cfg = LazyConfig(BoxModulationND)(data_dim=1, max_offset=max_offset)
    elif window == "gaussian":
        mask_cfg = LazyConfig(GaussianModulationND)(data_dim=1, num_channels=hidden_dim)
    elif window == "none":
        mask_cfg = LazyConfig(torch.nn.Identity)()
    else:
        raise ValueError(f"unknown window {window!r}")
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=hidden_dim,
        out_proj_bias=True,
        mixer_cfg=LazyConfig(TriangleHyena)(
            hidden_dim=hidden_dim,
            start_node=start_node,
            short_conv_backend=short_conv_backend,
            right_padding=right_padding,
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=1,
                hidden_dim=hidden_dim,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    out_dim=hidden_dim,
                    data_dim=1,
                    mlp_hidden_dim=32,
                    num_layers=3,
                    embedding_dim=32,
                    omega_0=30.0,
                    L_cache=L_cache,
                    use_bias=True,
                ),
                mask_cfg=mask_cfg,
                grid_type="double",
                fft_padding="zero",
                is_causal=False,
                use_shortcut=use_shortcut,
                fft_backend=fft_backend,
            ),
        ),
    )


def build_tri_hyena(hidden_dim: int, seq_len: int, **kwargs) -> QKVSequenceMixer:
    """Instantiate the mixer; ``L_cache`` defaults to the full two-sided filter length ``2L - 1``."""
    kwargs.setdefault("L_cache", 2 * seq_len - 1)
    return instantiate(tri_hyena_mixer_cfg(hidden_dim, **kwargs))


def cast_params(module: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    """Cast parameters only; the SIREN ``grid_cache`` buffer must stay fp32."""
    for param in module.parameters():
        param.data = param.data.to(dtype)
    return module


class PairMixer(torch.nn.Module):
    """Pre-norm + mixer called as ``module(pair, is_padding)``.

    In ``nvsubquadratic`` the residual block owns the ``LayerNorm``, so it is applied here.
    """

    def __init__(self, mixer: torch.nn.Module, norm: torch.nn.Module):
        """Wrap ``mixer`` with the pre-norm ``norm``."""
        super().__init__()
        self.norm = norm
        self.mixer = mixer

    def forward(self, pair: torch.Tensor, is_padding: torch.Tensor) -> torch.Tensor:
        """Apply ``norm`` then the mixer with ``is_padding`` as the padding mask."""
        return self.mixer(self.norm(pair), padding_mask=is_padding)


_l2_flush_buf: torch.Tensor | None = None
_SEGMENT_ROUND = 2 * 1024 * 1024


def _l2_flush_bytes() -> int:
    """Size in bytes of the L2 flush buffer, or 0 when L2 flushing is off."""
    return 0 if _l2_flush_buf is None else _l2_flush_buf.numel()


def _l2_flush_reserved_bytes() -> int:
    """Flush-buffer bytes rounded up to the allocator's 2 MiB segment granularity."""
    n = _l2_flush_bytes()
    return (n + _SEGMENT_ROUND - 1) // _SEGMENT_ROUND * _SEGMENT_ROUND


@dataclass
class BenchConfig:
    """Resolved benchmark settings."""

    device: torch.device
    device_name: str
    dtype: torch.dtype
    hidden_dim: int
    max_offset: int
    n_warmup: int
    warmup_seconds: float
    n_bench: int
    flush_l2: bool
    mode: str
    right_pad_frac: float

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BenchConfig:
        """Build the config from parsed CLI arguments and allocate the L2 flush buffer if requested."""
        if not torch.cuda.is_available():
            print("CUDA required.", file=sys.stderr)
            sys.exit(1)
        device = torch.device("cuda")
        if args.flush_l2:
            global _l2_flush_buf
            l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
            _l2_flush_buf = torch.empty(2 * l2_bytes, dtype=torch.uint8, device=device)
        return cls(
            device=device,
            device_name=torch.cuda.get_device_name(device),
            dtype=DTYPES[args.dtype],
            hidden_dim=args.hidden_dim,
            max_offset=args.max_offset,
            n_warmup=args.n_warmup,
            warmup_seconds=args.warmup_seconds,
            n_bench=args.n_bench,
            flush_l2=args.flush_l2,
            mode=args.mode,
            right_pad_frac=args.right_pad_frac,
        )

    def flush_l2_cache(self) -> None:
        """Overwrite the L2 flush buffer when L2 flushing is enabled."""
        if self.flush_l2:
            _l2_flush_buf.zero_()

    def warmup(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` ``n_warmup`` times, then until ``warmup_seconds`` of wall time have passed."""
        start = time.perf_counter()
        for _ in range(self.n_warmup):
            fn()
        while True:
            torch.cuda.synchronize(self.device)
            if time.perf_counter() - start >= self.warmup_seconds:
                break
            fn()

    @property
    def dtype_name(self) -> str:
        """Benchmark dtype name without the ``torch.`` prefix."""
        return str(self.dtype).replace("torch.", "")


def make_inputs(cfg: BenchConfig, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random pair tensor and a padding mask marking the last ``round(right_pad_frac * L)`` positions."""
    pair = torch.randn(batch_size, seq_len, seq_len, cfg.hidden_dim, device=cfg.device, dtype=cfg.dtype)
    is_padding = torch.zeros(batch_size, seq_len, device=cfg.device, dtype=torch.bool)
    is_padding[:, seq_len - round(cfg.right_pad_frac * seq_len) :] = True
    return pair, is_padding


def peak_mb(device) -> float:
    """Peak allocated memory in MiB, excluding the L2 flush buffer."""
    return (torch.cuda.max_memory_allocated(device) - _l2_flush_bytes()) / (1024 * 1024)


def alloc_mb(device) -> float:
    """Currently allocated memory in MiB, excluding the L2 flush buffer."""
    return (torch.cuda.memory_allocated(device) - _l2_flush_bytes()) / (1024 * 1024)


def reserved_mb(device) -> float:
    """Currently reserved memory in MiB, excluding the L2 flush buffer's segments."""
    return (torch.cuda.memory_reserved(device) - _l2_flush_reserved_bytes()) / (1024 * 1024)


def peak_reserved_mb(device) -> float:
    """Peak reserved memory in MiB, excluding the L2 flush buffer's segments."""
    return (torch.cuda.max_memory_reserved(device) - _l2_flush_reserved_bytes()) / (1024 * 1024)


def reset_device_state(device) -> None:
    """Drop compiled graphs and their cudagraph private pools, then release cached segments."""
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def _timed_events(n: int):
    """Two lists of ``n`` timing-enabled CUDA events (starts, ends)."""
    return (
        [torch.cuda.Event(enable_timing=True) for _ in range(n)],
        [torch.cuda.Event(enable_timing=True) for _ in range(n)],
    )


def _mean_ms(starts, ends) -> float:
    """Mean elapsed milliseconds over paired start and end events."""
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / len(starts)


def _setup(cfg: BenchConfig, fn: Callable[[], object]) -> dict:
    """Warm up ``fn`` (compile and graph capture happen here) and account its time and memory."""
    device = cfg.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base_mb, base_reserved_mb = alloc_mb(device), reserved_mb(device)
    t0 = time.perf_counter()
    cfg.warmup(fn)
    torch.cuda.synchronize(device)
    setup_ms = (time.perf_counter() - t0) * 1e3
    return {
        "setup_ms": setup_ms,
        "setup_peak_delta_mb": peak_mb(device) - base_mb,
        "setup_peak_reserved_mb": peak_reserved_mb(device),
        "_base_reserved_mb": base_reserved_mb,
    }


def _steady_reserved(cfg: BenchConfig, metrics: dict) -> dict:
    """Reserved memory setup leaves behind, read after the timed calls so no cache re-growth lands in them."""
    gc.collect()
    torch.cuda.empty_cache()
    metrics["steady_reserved_delta_mb"] = reserved_mb(cfg.device) - metrics.pop("_base_reserved_mb")
    return metrics


def _time_region(
    cfg: BenchConfig, fn: Callable[[], object], prepare: Callable[[], None] | None = None
) -> tuple[float, float, float]:
    """Mean ms, max peak MB and max peak-minus-base MB over ``n_bench`` calls of ``fn``.

    ``prepare`` runs before the base read of every iteration (frees the previous step's gradients).
    """
    starts, ends = _timed_events(cfg.n_bench)
    peak = peak_delta = 0.0
    for i in range(cfg.n_bench):
        if prepare:
            prepare()
        torch.cuda.synchronize(cfg.device)
        torch.cuda.reset_peak_memory_stats(cfg.device)
        base = alloc_mb(cfg.device)
        cfg.flush_l2_cache()
        starts[i].record()
        out = fn()
        ends[i].record()
        torch.cuda.synchronize(cfg.device)
        p = peak_mb(cfg.device)
        peak, peak_delta = max(peak, p), max(peak_delta, p - base)
        del out
    return _mean_ms(starts, ends), peak, peak_delta


def run_one(cfg: BenchConfig, module: torch.nn.Module, batch_size: int, seq_len: int) -> dict:
    """One (arch, seq_len) measurement; raises ``torch.cuda.OutOfMemoryError`` on OOM."""
    pair, is_padding = make_inputs(cfg, batch_size, seq_len)
    metrics = dict.fromkeys(METRIC_FIELDS)
    metrics["param_count"] = sum(p.numel() for p in module.parameters())

    if cfg.mode == "infer":
        with torch.inference_mode():
            metrics.update(_setup(cfg, lambda: module(pair, is_padding)))
            metrics["fwd_ms"], metrics["fwd_peak_mb"], metrics["fwd_peak_delta_mb"] = _time_region(
                cfg, lambda: module(pair, is_padding)
            )
        return _steady_reserved(cfg, metrics)

    def loss_fn():
        return module(pair, is_padding).float().pow(2).mean()

    def zero_grad():
        module.zero_grad(set_to_none=True)

    def train_step():
        zero_grad()
        loss_fn().backward()

    metrics.update(_setup(cfg, train_step))

    starts, ends = _timed_events(cfg.n_bench)
    bwd_starts, bwd_ends = _timed_events(cfg.n_bench)
    fwd_peak = fwd_peak_delta = act = 0.0
    for i in range(cfg.n_bench):
        zero_grad()
        torch.cuda.synchronize(cfg.device)
        torch.cuda.reset_peak_memory_stats(cfg.device)
        base = alloc_mb(cfg.device)
        cfg.flush_l2_cache()
        starts[i].record()
        loss = loss_fn()
        ends[i].record()
        torch.cuda.synchronize(cfg.device)
        p = peak_mb(cfg.device)
        fwd_peak, fwd_peak_delta = max(fwd_peak, p), max(fwd_peak_delta, p - base)
        act = max(act, alloc_mb(cfg.device))
        cfg.flush_l2_cache()
        bwd_starts[i].record()
        loss.backward()
        bwd_ends[i].record()
        torch.cuda.synchronize(cfg.device)
        del loss
    metrics["fwd_ms"], metrics["bwd_ms"] = _mean_ms(starts, ends), _mean_ms(bwd_starts, bwd_ends)
    metrics["fwd_peak_mb"], metrics["fwd_peak_delta_mb"], metrics["act_mb"] = fwd_peak, fwd_peak_delta, act
    metrics["step_ms"], metrics["step_peak_mb"], metrics["step_peak_delta_mb"] = _time_region(
        cfg, train_step, prepare=zero_grad
    )
    return _steady_reserved(cfg, metrics)


def print_row(row: dict) -> None:
    """Print one result row as a fixed-width line."""

    def fmt(value, width, spec):
        return f"{value:{width}{spec}}" if value is not None else " " * width

    print(
        f"  {row['arch']:>28s} L={row['seq_len']:<5d}"
        f"  fwd={fmt(row['fwd_ms'], 8, '.2f')}ms  bwd={fmt(row['bwd_ms'], 8, '.2f')}ms"
        f"  step={fmt(row['step_ms'], 8, '.2f')}ms"
        f"  | fwd_peak={fmt(row['fwd_peak_mb'], 8, '.0f')}MB (+{fmt(row['fwd_peak_delta_mb'], 6, '.0f')})"
        f"  step_peak={fmt(row['step_peak_mb'], 8, '.0f')}MB (+{fmt(row['step_peak_delta_mb'], 6, '.0f')})"
        f"  act={fmt(row['act_mb'], 8, '.0f')}MB"
        f"  steady_res=+{fmt(row['steady_reserved_delta_mb'], 6, '.0f')}MB"
        f"  setup={fmt(None if row['setup_ms'] is None else row['setup_ms'] / 1e3, 6, '.1f')}s"
        f"  {row['status']}"
    )


def save_csv(rows: list[dict], path: str) -> None:
    """Write ``rows`` to ``path`` as CSV; no-op when ``rows`` is empty."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(
    cfg: BenchConfig,
    args: argparse.Namespace,
    *,
    archs: dict[str, Callable[[BenchConfig, int], torch.nn.Module]],
) -> list[dict]:
    """Sweep ``seq_lens x archs`` and write ``args.csv`` incrementally.

    ``archs`` maps a name to ``builder(cfg, seq_len) -> module`` already on the
    device in ``cfg.dtype``.  Each module is built after the compile state and cached segments are
    reset, measured, and released before the next arch is built, so its memory columns and OOM status
    describe that arch alone.
    """
    selected = [name for name in args.archs if name in archs]
    print(f"Device: {cfg.device_name}  mode: {cfg.mode}  dtype: {cfg.dtype_name}")
    print(
        f"Config: B={args.batch_size}  C={cfg.hidden_dim}  max_offset={cfg.max_offset}  "
        f"warmup={cfg.n_warmup}+{cfg.warmup_seconds:g}s  bench={cfg.n_bench}  l2_flush={'on' if cfg.flush_l2 else 'off'}"
    )
    print(f"Seq lens: {args.seq_lens}\n")

    rows: list[dict] = []
    oomed: set[str] = set()
    for seq_len in args.seq_lens:
        print(f"--- L={seq_len}  B={args.batch_size} ---")
        for name in selected:
            row = {
                "arch": name,
                "seq_len": seq_len,
                "batch_size": args.batch_size,
                "hidden_dim": cfg.hidden_dim,
                "dtype": cfg.dtype_name,
                "mode": cfg.mode,
                "device_name": cfg.device_name,
                **dict.fromkeys(METRIC_FIELDS),
                "status": "OOM" if name in oomed else "OK",
            }
            if name not in oomed:
                reset_device_state(cfg.device)
                module = None
                try:
                    module = archs[name](cfg, seq_len)
                    row.update(run_one(cfg, module, args.batch_size, seq_len))
                except torch.cuda.OutOfMemoryError:
                    oomed.add(name)
                    row["status"] = "OOM"
                except Exception:
                    traceback.print_exc()
                    row["status"] = "ERROR"
                finally:
                    module = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(cfg.device)
            rows.append(row)
            print_row(row)
            save_csv(rows, args.csv)
        if len(oomed) == len(selected):
            print("All archs OOMed; stopping sweep.")
            break
    print(f"\nWrote {len(rows)} rows to {args.csv}")
    return rows
