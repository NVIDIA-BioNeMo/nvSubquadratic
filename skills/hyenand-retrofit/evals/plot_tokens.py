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

"""Plot LLM token usage / cost for the hyenand-retrofit skill.

Reads bench_results.jsonl (produced by bench_tokens.py) and renders a two-panel
figure to bench_tokens.png:

  Panel A — stacked mean token breakdown per eval (cache_read, cache_creation,
            input, output). The stack height is the billed total; the labels
            make clear how much is fixed skill/system overhead (cache) vs.
            task-driven generation (output).
  Panel B — mean agent turns per eval with std-dev error bars across runs.
            Turns track how much back-and-forth the retrofit took and are the
            cleanest proxy for work (the CLI's dollar figure is noisy for
            multi-turn runs, so it is deliberately not plotted).

Usage:
    python plot_tokens.py [--results bench_results.jsonl] [--out bench_tokens.png]
"""

import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = pathlib.Path(__file__).parent
COMPONENTS = [
    ("cache_read_tokens", "cache read", "#bdd7e7"),
    ("cache_creation_tokens", "cache creation", "#6baed6"),
    ("input_tokens", "input", "#2171b5"),
    ("output_tokens", "output (generated)", "#e6550d"),
]


def mean(xs):
    """Arithmetic mean; returns 0.0 for an empty sequence."""
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    """Sample standard deviation; returns 0.0 for fewer than two values."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def load(results_path: pathlib.Path):
    """Load successful benchmark rows from a JSONL results file."""
    rows = []
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" not in r:
            rows.append(r)
    if not rows:
        raise SystemExit(f"No successful runs found in {results_path}")
    return rows


def aggregate(rows):
    """Group rows by (eval_id, eval_name), preserving eval_id order."""
    by = defaultdict(list)
    # Preserve eval_id ordering for a stable x-axis.
    for r in rows:
        by[(r["eval_id"], r["eval_name"])].append(r)
    keys = sorted(by, key=lambda k: k[0])
    return keys, by


def main():
    """CLI entry point for plotting benchmark token usage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "bench_results.jsonl"))
    ap.add_argument("--out", default=str(HERE / "bench_tokens.png"))
    args = ap.parse_args()

    rows = load(pathlib.Path(args.results))
    keys, by = aggregate(rows)
    names = [k[1] for k in keys]
    model = rows[0].get("model", "?")
    n_runs = max(len(by[k]) for k in keys)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- Panel A: stacked token breakdown ----
    x = range(len(keys))
    bottoms = [0.0] * len(keys)
    for field, label, color in COMPONENTS:
        vals = [mean([r[field] for r in by[k]]) for k in keys]
        axA.bar(x, vals, bottom=bottoms, label=label, color=color, width=0.62)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    # Annotate billed total on top of each stack.
    for xi, total in zip(x, bottoms):
        axA.text(xi, total, f"{total / 1000:.0f}k", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axA.set_xticks(list(x))
    axA.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    axA.set_ylabel("tokens (mean of runs)")
    axA.set_title("Token breakdown per retrofit eval")
    axA.set_ylim(top=max(bottoms) * 1.15)  # headroom so top labels clear the legend
    axA.legend(fontsize=8, loc="best")  # auto-place away from the tallest bar
    axA.grid(axis="y", alpha=0.3)

    # ---- Panel B: agent turns per eval with error bars ----
    turns = [mean([r["num_turns"] for r in by[k]]) for k in keys]
    turns_err = [std([r["num_turns"] for r in by[k]]) for k in keys]
    bars = axB.bar(x, turns, yerr=turns_err, capsize=4, color="#31a354", width=0.62)
    for bar, t in zip(bars, turns):
        axB.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axB.set_xticks(list(x))
    axB.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    axB.set_ylabel("agent turns (mean ± std)")
    axB.set_title("Agent turns per retrofit eval")
    axB.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"hyenand-retrofit skill — token efficiency  (model: {model}, {n_runs} runs/eval, {len(rows)} runs)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
