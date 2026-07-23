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

"""Plot the forward-time-vs-resolution sweep (1D / 2D / 3D) in paper (Figure 1) style.

Dimensionality, hardware, and per-operator gaps are read from the JSONL, so the
title (``{N}D Forward-Time Scaling``), x-axis (``L = R^N``), tick exponents, and
device subtitle adapt automatically.

Reads the JSONL written by
``benchmarks/benchmark_forward_time_nd_resolution.py`` and renders forward time
(log y) against 2D token count ``L = R^2`` (log x) for HyenaND / Attention /
Mamba2 — the 2D analogue of Figure 1 (right).  Points that ran out of memory or
exceeded the time budget are drawn as a green "OOM" ``x`` (as in Figure 1), and
the largest HyenaND-vs-Attention gap is annotated (the "339x"-style label).

The plotter is decoupled from the GPU run: it needs only the JSONL and
matplotlib, so it can run on a login node / laptop.

Usage::

    python scripts/visualization/visualize_forward_time_nd.py \\
        --input benchmarks/results/forward_time.jsonl \\
        --out   benchmarks/results/forward_time.png
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# mixer key -> (legend label, colour, marker) — colours match Figure 1. The three
# attention kernels (SDPA / FlexAttention / FA4) share a cool-toned family so the
# plot reads them as attention variants, distinguished by shade + marker.
SERIES = {
    "hyena": {"label": "HyenaND (nSubQ)", "color": "#C0392B", "marker": "s"},
    "attention": {"label": "Attention (SDPA)", "color": "#3B6FB6", "marker": "o"},
    "flex": {"label": "FlexAttention", "color": "#17A2B8", "marker": "^"},
    "fa4": {"label": "FlashAttention-4", "color": "#9467BD", "marker": "v"},
    "mamba": {"label": "Mamba2", "color": "#7A8B3C", "marker": "D"},
}
SERIES_ORDER = ["hyena", "attention", "flex", "fa4", "mamba"]

# Attention-family mixers (all avoid the O(L^2) score matrix via flash-class kernels).
_ATTN_MIXERS = ("attention", "flex", "fa4")

FAIL_STATUS = {"oom", "error", "timeout"}
FAIL_COLOR = "#2E7D32"  # green "OOM x", as in Figure 1

# Approx usable HBM per GPU (GB), for the memory-metric ceiling line.
_GPU_MEM_GB = {
    "GB300": 279.0,
    "B300": 279.0,
    "GB200": 186.0,
    "B200": 186.0,
    "H200": 141.0,
    "H100": 80.0,
    "A100": 80.0,
    "A6000": 48.0,
}


def _gpu_mem_gb(device: str | None) -> float | None:
    if not device:
        return None
    for key, gb in _GPU_MEM_GB.items():
        if key in device:
            return gb
    return None


def _format_seq(n: int) -> str:
    if n >= 1024 * 1024:
        v = n / (1024 * 1024)
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1024:
        v = n / 1024
        return f"{v:.0f}K" if v == int(v) else f"{v:.1f}K"
    return str(n)


def _tick_label(seq_len: int, data_dim: int) -> str:
    if data_dim == 1:
        return _format_seq(seq_len)  # 1D: L == R, single line
    r = round(seq_len ** (1.0 / data_dim))
    sup = {2: "²", 3: "³"}.get(data_dim, f"^{data_dim}")
    return f"{_format_seq(seq_len)}\n{r}{sup}"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_plot(
    rows: list[dict],
    out_path: Path,
    show_fail_markers: bool = True,
    metric: str = "time",
    attn_heads: int | None = None,
    gpu_mem_gb: float | None = None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    mkey = "ms" if metric == "time" else "mem_gb"

    # Drop resolutions where no operator produced a value for this metric, so the
    # x-axis ends at the last reachable point.
    ok_seq = {int(r["seq_len"]) for r in rows if r.get("status") == "ok" and r.get(mkey) is not None}
    rows = [r for r in rows if int(r["seq_len"]) in ok_seq]

    # Dimensionality (L = R^data_dim) and hardware, inferred from the data.
    def _dim_of(r: dict) -> int:
        if r.get("data_dim"):
            return int(r["data_dim"])
        seq_len, res = int(r["seq_len"]), int(r["resolution"])
        return max(1, round(math.log(seq_len) / math.log(res))) if res > 1 else 2

    data_dim = _dim_of(rows[0])
    devices = [r.get("device") for r in rows if r.get("device")]
    device = max(set(devices), key=devices.count) if devices else None

    # Group by mixer -> {seq_len: row}
    by_mixer: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_mixer[r["mixer"]][int(r["seq_len"])] = r

    all_seq = sorted({int(r["seq_len"]) for r in rows})
    ok_vals = [r[mkey] for r in rows if r.get("status") == "ok" and r.get(mkey) is not None]
    if not ok_vals:
        raise SystemExit(f"No successful ('ok') {mkey} values in the JSONL — nothing to plot.")
    ceiling = max(ok_vals)  # where OOM/timeout x-marks with no measured value are parked

    # Widen with the tick count so the two-line "L / R^2" labels don't crowd.
    fig_w = min(9.0, max(3.6, 0.6 * len(all_seq) + 1.2))
    fig, ax = plt.subplots(figsize=(fig_w, 3.2), constrained_layout=True)

    fail_x, fail_y = [], []
    for mixer in SERIES_ORDER:
        pts = by_mixer.get(mixer)
        if not pts:
            continue
        cfg = SERIES[mixer]
        xs = sorted(pts)

        ok_x = [x for x in xs if pts[x].get("status") == "ok" and pts[x].get(mkey) is not None]

        # A series that never produced a single timing never actually ran here
        # (missing dep, build/version error, or OOM even at the smallest grid) —
        # that is a setup failure, not a scaling wall, so omit it entirely rather
        # than paint a wall of x's across every resolution.
        if not ok_x:
            n_fail = sum(1 for x in xs if pts[x].get("status") in FAIL_STATUS)
            if n_fail:
                print(f"Omitting '{mixer}': 0/{len(xs)} points ran — {pts[xs[0]].get('detail', '')[:90]}")
            continue

        ax.plot(
            ok_x,
            [pts[x][mkey] for x in ok_x],
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle="-",
            linewidth=1.6,
            markersize=5.5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=cfg["label"],
        )

        # Mark only the FIRST failure per series — the wall where it stops. Later
        # failures at larger L are redundant (and clutter/overlap other series).
        # A wall-timed timeout carries a measured (over-budget) ms → place the x
        # there; a predictive skip has ms=None → park it at the ceiling.
        fails = [x for x in xs if pts[x].get("status") in FAIL_STATUS]
        if fails:
            x = fails[0]  # xs is sorted ascending → smallest-L failure
            fail_x.append(x)
            fail_y.append(pts[x][mkey] if pts[x].get(mkey) is not None else ceiling)

    if fail_x and show_fail_markers:
        ax.scatter(
            fail_x,
            fail_y,
            marker="x",
            s=70,
            linewidths=2.0,
            color=FAIL_COLOR,
            zorder=5,
            label="OOM / timeout",
        )

    # ── Metric-specific overlays ──────────────────────────────────────────────
    gaps: dict[str, tuple[int, float] | None] = {}
    mem_ceiling: float | None = None
    if metric == "time":
        # HyenaND vs each slower operator: one double-arrow "N×" at its max gap
        # (vs Attention at the largest shared L; vs Mamba at its last point).
        def _annotate_gap(slow_key: str, fast_key: str) -> tuple[int, float] | None:
            s_pts, f_pts = by_mixer.get(slow_key, {}), by_mixer.get(fast_key, {})
            best = None  # (x, ratio, slow, fast)
            for x in all_seq:
                s, f = s_pts.get(x), f_pts.get(x)
                if not s or not f or s.get("status") != "ok" or f.get("status") != "ok":
                    continue
                if s.get("ms") is None or f.get("ms") is None:
                    continue
                ratio = s["ms"] / f["ms"]
                if best is None or ratio > best[1]:
                    best = (x, ratio, s["ms"], f["ms"])
            if best is None or best[1] < 1.5:
                return None
            x, ratio, s_ms, f_ms = best
            ax.annotate(
                "",
                xy=(x, s_ms),
                xytext=(x, f_ms),
                arrowprops={"arrowstyle": "<->", "color": "0.4", "lw": 0.8, "mutation_scale": 6},
            )
            ax.text(
                x,
                math.sqrt(s_ms * f_ms),
                f" {ratio:.0f}×",
                color="0.25",
                fontsize=9,
                ha="left",
                va="center",
                fontweight="bold",
            )
            return (x, ratio)

        gaps = {slow: _annotate_gap(slow, "hyena") for slow in ("attention", "mamba")}
        title = f"{data_dim}D Forward-Time Scaling"
        ylabel = "Forward time (ms)"
    else:
        # O(L^2) memory a *materialized* (non-flash) attention needs for its
        # H×L×L score matrix in the model dtype — the term flash avoids by
        # recomputing (and pays for in time). The measured curves are all O(L);
        # this dashed reference shows what attention would cost without flash,
        # and where it exceeds GPU memory (~256K here).
        heads = (
            attn_heads
            or next(
                (r.get("num_heads") for r in rows if r.get("mixer") in _ATTN_MIXERS and r.get("num_heads")),
                None,
            )
            or 2
        )
        dbytes = {"bf16": 2, "fp16": 2, "fp32": 4}.get(rows[0].get("dtype", "bf16"), 2)
        batch = rows[0].get("batch_size", 1) or 1
        materialized = [batch * heads * (x**2) * dbytes / 1e9 for x in all_seq]
        ax.plot(
            all_seq,
            materialized,
            color=SERIES["attention"]["color"],
            linestyle="--",
            dashes=(4, 2),
            linewidth=1.4,
            label="Attention (materialized scores)",
        )
        mem = gpu_mem_gb if gpu_mem_gb is not None else _gpu_mem_gb(device)
        if mem:
            ax.axhline(mem, color="0.45", linestyle=":", linewidth=1.0, zorder=1)
            ax.text(
                all_seq[-1], mem, f"  {device or 'GPU'} memory", fontsize=7.5, color="0.4", va="bottom", ha="right"
            )
            mem_ceiling = mem  # applied as ylim after the log scale is set
        title = f"{data_dim}D Peak-Memory Scaling"
        ylabel = "Peak memory (GB)"

    fig.suptitle(title, fontsize=12, fontweight="bold")
    if device:
        ax.set_title(device, fontsize=9, fontweight="normal", color="0.4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(all_seq)
    ax.set_xticklabels([_tick_label(s, data_dim) for s in all_seq])
    ax.minorticks_off()
    if data_dim == 1:
        ax.set_xlabel("Sequence length  (tokens)")
    else:
        ax.set_xlabel(f"{data_dim}D context  (tokens $L = R^{data_dim}$)")
    ax.set_ylabel(ylabel)
    if mem_ceiling:  # after set_yscale, else switching to log re-autoscales
        ax.set_ylim(top=mem_ceiling * 4)
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.6, borderaxespad=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")
    for slow, g in gaps.items():
        if g:
            print(f"Max {slow}/HyenaND gap: {g[1]:.0f}x at L={g[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmarks/results/forward_time.jsonl"),
        help="JSONL produced by benchmark_forward_time_nd_resolution.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/forward_time.png"),
    )
    parser.add_argument(
        "--no-fail-markers",
        action="store_true",
        help="Hide the green OOM/timeout 'x' markers (each curve just ends where it walls).",
    )
    parser.add_argument(
        "--metric",
        choices=["time", "memory"],
        default="time",
        help="'time' = forward ms (Figure 1). 'memory' = peak GB, with an O(L^2) "
        "materialized-attention reference line and the GPU-memory ceiling.",
    )
    parser.add_argument(
        "--attn-heads",
        type=int,
        default=None,
        help="Head count for the materialized-scores memory line (default: read from the JSONL, else 2).",
    )
    parser.add_argument(
        "--gpu-mem-gb",
        type=float,
        default=None,
        help="GPU memory (GB) for the ceiling line (default: inferred from the device name).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input JSONL not found: {args.input}")
    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")
    make_plot(
        rows,
        args.out,
        show_fail_markers=not args.no_fail_markers,
        metric=args.metric,
        attn_heads=args.attn_heads,
        gpu_mem_gb=args.gpu_mem_gb,
    )


if __name__ == "__main__":
    main()
