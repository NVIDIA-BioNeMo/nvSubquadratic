"""Plot patch-size / sequence-length throughput results in paper style.

Renders the ms/batch numbers from the H100 patch-size benchmark
(attention vs. hyena vs. mamba_ssm, batch=4, bf16, max-autotune) as a
single small-multiple-style panel matching the figure layout used in
the manuscript (see paper/ figs: bold title, scientific-notation y-axis,
log-spaced sequence-length ticks, circle/square markers).

Usage::

    python scripts/visualize_patch_size_throughput.py \
        --out throughput_benchmark/results/patch_size_step_time.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# (patch, seq_len, attention, hyena, mamba) — ms / batch, lower is better
DATA = [
    (1, 4096, 1.986, 1.550, 4.478),
    (2, 1024, 0.692, 1.502, 4.333),
    (4,  256, 0.687, 1.552, 4.301),
    (8,   64, 0.649, 1.372, 4.283),
]

SERIES = {
    "Attention": {"idx": 2, "color": "#3B6FB6", "marker": "o"},
    "Hyena":     {"idx": 3, "color": "#C0392B", "marker": "s"},
    "Mamba2":    {"idx": 4, "color": "#7A8B3C", "marker": "D", "linestyle": "--"},
}


def _format_seq(n: int) -> str:
    if n >= 1000:
        return f"{round(n / 1024)}K" if n >= 1024 else f"{round(n / 1000)}K"
    return str(n)


def make_plot(out_path: Path) -> None:
    plt.rcParams.update({
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
    })

    fig, ax = plt.subplots(figsize=(2.6, 2.4), constrained_layout=True)

    seq_lens = np.array([row[1] for row in DATA])
    sort_idx = np.argsort(seq_lens)
    x = seq_lens[sort_idx]

    for label, cfg in SERIES.items():
        y = np.array([row[cfg["idx"]] for row in DATA])[sort_idx]
        ax.plot(
            x, y,
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=cfg.get("linestyle", "-"),
            linewidth=1.6,
            markersize=5.5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=label,
        )

    ax.set_title("Patch-Size Throughput")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_seq(s) for s in x])
    ax.minorticks_off()
    ax.set_xlabel("Seq. length")
    ax.set_ylabel("ms / batch")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_ylim(bottom=0)

    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
        handlelength=1.6,
        borderaxespad=0.4,
        ncol=3,
        columnspacing=1.0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("throughput_benchmark/results/patch_size_step_time.png"),
    )
    args = parser.parse_args()
    make_plot(args.out)


if __name__ == "__main__":
    main()
