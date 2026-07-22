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

"""Plot the 2D forward-time-vs-resolution sweep in paper (Figure 1) style.

Reads the JSONL written by
``benchmarks/benchmark_forward_time_2d_resolution.py`` and renders forward time
(log y) against 2D token count ``L = R^2`` (log x) for HyenaND / Attention /
Mamba2 — the 2D analogue of Figure 1 (right).  Points that ran out of memory or
exceeded the time budget are drawn as a green "OOM" ``x`` (as in Figure 1), and
the largest HyenaND-vs-Attention gap is annotated (the "339x"-style label).

The plotter is decoupled from the GPU run: it needs only the JSONL and
matplotlib, so it can run on a login node / laptop.

Usage::

    python scripts/visualization/visualize_forward_time_2d.py \\
        --input benchmarks/results/forward_time_2d_resolution.jsonl \\
        --out   benchmarks/results/forward_time_2d_resolution.png
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


# mixer key -> (legend label, colour, marker) — colours match Figure 1.
SERIES = {
    "hyena": {"label": "HyenaND (nSubQ)", "color": "#C0392B", "marker": "s"},
    "attention": {"label": "Attention", "color": "#3B6FB6", "marker": "o"},
    "mamba": {"label": "Mamba2", "color": "#7A8B3C", "marker": "D"},
}
SERIES_ORDER = ["hyena", "attention", "mamba"]

FAIL_STATUS = {"oom", "error", "timeout"}
FAIL_COLOR = "#2E7D32"  # green "OOM x", as in Figure 1


def _format_seq(n: int) -> str:
    if n >= 1024 * 1024:
        v = n / (1024 * 1024)
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1024:
        v = n / 1024
        return f"{v:.0f}K" if v == int(v) else f"{v:.1f}K"
    return str(n)


def _tick_label(seq_len: int) -> str:
    r = round(math.sqrt(seq_len))
    return f"{_format_seq(seq_len)}\n{r}²"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_plot(rows: list[dict], out_path: Path) -> None:
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

    # Group by mixer -> {seq_len: row}
    by_mixer: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_mixer[r["mixer"]][int(r["seq_len"])] = r

    all_seq = sorted({int(r["seq_len"]) for r in rows})
    ok_ms = [r["ms"] for r in rows if r.get("status") == "ok" and r.get("ms") is not None]
    if not ok_ms:
        raise SystemExit("No successful ('ok') timings in the JSONL — nothing to plot.")
    ceiling = max(ok_ms)  # where OOM/timeout x-marks with no measured ms are parked

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

        ok_x = [x for x in xs if pts[x].get("status") == "ok" and pts[x].get("ms") is not None]

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
            [pts[x]["ms"] for x in ok_x],
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle="-",
            linewidth=1.6,
            markersize=5.5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=cfg["label"],
        )

        # Walls (oom/timeout/error) only for a series that was working — i.e. that
        # has at least one ok point. A wall-timed timeout carries a measured
        # (over-budget) ms; place the x there. A predictive skip has ms=None → park
        # it at the ceiling.
        for x in xs:
            if pts[x].get("status") in FAIL_STATUS:
                fail_x.append(x)
                fail_y.append(pts[x]["ms"] if pts[x].get("ms") is not None else ceiling)

    if fail_x:
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

    # ── Speedup annotation (max Attention/HyenaND ratio) ──────────────────────
    hy = by_mixer.get("hyena", {})
    at = by_mixer.get("attention", {})
    best = None  # (seq_len, ratio, attn_ms, hyena_ms)
    for x in all_seq:
        h, a = hy.get(x), at.get(x)
        if not h or not a:
            continue
        if h.get("status") != "ok" or h.get("ms") is None or a.get("ms") is None:
            continue
        ratio = a["ms"] / h["ms"]  # attention measured (ok or over-budget timeout) vs hyena
        if best is None or ratio > best[1]:
            best = (x, ratio, a["ms"], h["ms"])
    if best is not None and best[1] > 1.5:
        x, ratio, a_ms, h_ms = best
        ymid = math.sqrt(a_ms * h_ms)  # geometric mean sits centred on a log axis
        ax.annotate(
            "",
            xy=(x, a_ms),
            xytext=(x, h_ms),
            arrowprops={"arrowstyle": "<->", "color": "0.35", "lw": 1.0},
        )
        ax.text(x, ymid, f" {ratio:.0f}×", color="0.25", fontsize=9, ha="left", va="center", fontweight="bold")

    ax.set_title("2D Forward-Time Scaling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(all_seq)
    ax.set_xticklabels([_tick_label(s) for s in all_seq])
    ax.minorticks_off()
    ax.set_xlabel("2D context  (tokens $L = R^2$)")
    ax.set_ylabel("Forward time (ms)")
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.6, borderaxespad=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")
    if best is not None:
        print(f"Max Attention/HyenaND gap: {best[1]:.0f}x at L={best[0]} ({round(math.sqrt(best[0]))}^2)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmarks/results/forward_time_2d_resolution.jsonl"),
        help="JSONL produced by benchmark_forward_time_2d_resolution.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/forward_time_2d_resolution.png"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input JSONL not found: {args.input}")
    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")
    make_plot(rows, args.out)


if __name__ == "__main__":
    main()
