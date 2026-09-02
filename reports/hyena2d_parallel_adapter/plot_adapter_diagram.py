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

"""Render the architecture diagram for the zero-init parallel adapter.

What: a two-panel figure.  Panel A shows the adapter's data flow inside one
ViT5ResidualBlock -- frozen attention and the trainable zero-init bottleneck
branch running in parallel into the residual add.  Panel B shows the four
``inner_mixer`` variants (the C0/C1/C2s/C2f ablation ladder) with measured
accuracy.

Hardware: CPU only.

Invoke::

    PYTHONPATH=. python reports/hyena2d_parallel_adapter/plot_adapter_diagram.py

Output: ``adapter_diagram.png`` (+ ``.svg``) next to this script.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


# Semantic palette.  Each colour is paired with a non-colour cue (icon, border
# weight, border style) so the figure survives greyscale and CVD.
FROZEN_FILL, FROZEN_EDGE = "#E2E8F0", "#64748B"
TRAIN_FILL, TRAIN_EDGE = "#CCFBF1", "#0D9488"
ZERO_FILL, ZERO_EDGE = "#FEF3C7", "#D97706"
INK, MUTED, FLOW = "#1F2937", "#6B7280", "#9CA3AF"

MONO = {"family": "monospace"}


def box(ax, xy, w, h, label, sub=None, fill="#fff", edge=INK, lw=1.4, ls="-", fs=11):
    """Draw a rounded box with a bold label and optional sub-label."""
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            linestyle=ls,
            zorder=3,
        )
    )
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    dy = 0.11 if sub else 0
    ax.text(cx, cy + dy, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4, **MONO)
    if sub:
        ax.text(cx, cy - 0.13, sub, ha="center", va="center", fontsize=8.5, color=MUTED, zorder=4)
    return cx, cy


def arrow(ax, a, b, label=None, color=FLOW, lw=1.7, rad=0.0, fs=8.5, off=(0.06, 0)):
    """Draw a flow arrow, optionally labelled with a tensor shape."""
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            zorder=2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        mx, my = (a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1]
        ax.text(mx, my, label, fontsize=fs, color=MUTED, ha="left", va="center", **MONO)


def trapezoid(ax, cx, ytop, ybot, wtop, wbot, label, sub, fill, edge, lw=1.6):
    """Draw a trapezoid, used to make the 128->32->128 bottleneck legible."""
    pts = [(cx - wtop / 2, ytop), (cx + wtop / 2, ytop), (cx + wbot / 2, ybot), (cx - wbot / 2, ybot)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3))
    cy = (ytop + ybot) / 2
    ax.text(cx, cy + 0.06, label, ha="center", va="center", fontsize=10.5, color=INK, zorder=4, **MONO)
    ax.text(cx, cy - 0.13, sub, ha="center", va="center", fontsize=8.5, color=MUTED, zorder=4)


def panel_a(ax):
    """Panel A -- the adapter mechanism inside one block.

    Layout uses an explicit vertical budget so no label shares space with a
    border or an arrow:

        6.30-6.85  input box
        4.25-5.65  attention box (all three of its labels live INSIDE it, so the
                   attention->join arrow cannot cross them)
        5.65-5.05  W_down          |  right branch, centred on CX_R
        4.15-4.90  inner_mixer     |  0.15 gap above, 0.20 below
        3.35-3.95  W_up            |
        2.05       join
        0.70-1.25  output box
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    CX_R, JX, JY = 7.2, 5.0, 2.05

    ax.text(
        0.05,
        7.20,
        "A.  What the adapter does  (inside one ViT5ResidualBlock)",
        fontsize=13,
        color=INK,
        va="top",
        weight="bold",
    )

    # Input
    box(ax, (3.6, 6.30), 2.8, 0.55, "x  (pre-normed)", fill="#fff", edge=MUTED, lw=1.2)
    ax.text(5.0, 6.12, "[B, 784, 128]", ha="center", fontsize=9, color=MUTED, **MONO)

    # Fan-out -- endpoints must match the shapes they point at.
    arrow(ax, (5.0, 6.30), (2.5, 5.70), rad=0.15)
    arrow(ax, (5.0, 6.30), (CX_R, 5.70), rad=-0.15)

    # LEFT -- frozen attention.  Dashed border + inline label are the non-colour
    # cues; keeping "FROZEN" inside the box avoids the join arrow crossing it.
    ax.add_patch(
        FancyBboxPatch(
            (1.05, 4.25),
            2.9,
            1.40,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=FROZEN_FILL,
            edgecolor=FROZEN_EDGE,
            linewidth=1.6,
            linestyle=(0, (5, 2)),
            zorder=3,
        )
    )
    ax.text(2.5, 5.24, "ViT5Attention", ha="center", va="center", fontsize=11, color=INK, zorder=4, **MONO)
    ax.text(2.5, 4.97, "RoPE + QK-norm", ha="center", va="center", fontsize=8.5, color=MUTED, zorder=4)
    ax.text(
        2.5,
        4.60,
        "FROZEN - not trained",
        ha="center",
        va="center",
        fontsize=9.5,
        color=FROZEN_EDGE,
        weight="bold",
        zorder=4,
    )

    # RIGHT -- trainable bottleneck.
    trapezoid(ax, CX_R, 5.65, 5.05, 2.9, 1.5, "W_down", "128 -> 32", TRAIN_FILL, TRAIN_EDGE)
    box(
        ax,
        (CX_R - 0.80, 4.15),
        1.6,
        0.75,
        "inner_mixer",
        sub="see panel B",
        fill=TRAIN_FILL,
        edge=TRAIN_EDGE,
        lw=1.6,
        fs=9.5,
    )
    trapezoid(ax, CX_R, 3.95, 3.35, 1.5, 2.9, "W_up", "32 -> 128,  init = 0", ZERO_FILL, ZERO_EDGE, lw=2.6)

    arrow(ax, (CX_R, 5.03), (CX_R, 4.92), lw=1.5)
    arrow(ax, (CX_R, 4.13), (CX_R, 3.99), lw=1.5)

    # Rejoin
    ax.add_patch(Circle((JX, JY), 0.26, facecolor="#fff", edgecolor=INK, linewidth=1.8, zorder=4))
    ax.text(JX, JY, "+", ha="center", va="center", fontsize=17, color=INK, zorder=5)
    arrow(ax, (2.5, 4.25), (JX - 0.24, JY + 0.18), rad=-0.12)
    arrow(ax, (CX_R, 3.35), (JX + 0.24, JY + 0.18), rad=-0.12)

    # Callout sits low-right, clear of W_up's flared base and of the join arrow.
    ax.text(8.30, 2.62, "at step 0 this branch", ha="left", fontsize=9, color=ZERO_EDGE)
    ax.text(8.30, 2.36, "outputs exactly 0", ha="left", fontsize=9.5, color=ZERO_EDGE, weight="bold")
    ax.text(8.30, 2.06, "-> model is bit-identical", ha="left", fontsize=8.4, color=MUTED)
    ax.text(8.30, 1.84, "   to the pretrained one", ha="left", fontsize=8.4, color=MUTED)

    arrow(ax, (JX, JY - 0.26), (JX, 1.30), lw=2.0)
    box(ax, (3.6, 0.70), 2.8, 0.55, "out", fill="#fff", edge=MUTED, lw=1.2)
    ax.text(5.0, 0.52, "[B, 784, 128]", ha="center", fontsize=9, color=MUTED, **MONO)

    ax.text(5.32, 1.60, "additive - attention is never replaced", ha="left", fontsize=9, color=MUTED, style="italic")

    ax.text(0.05, 0.14, "out = Attn(x) + W_up( inner_mixer( W_down(x) ) )", fontsize=11.5, color=INK, **MONO)


def panel_b(ax):
    """Panel B -- the four inner_mixer variants with measured accuracy."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    ax.text(
        0.05,
        3.38,
        "B.  The ablation ladder  -  identical scaffold, only inner_mixer differs",
        fontsize=13,
        color=INK,
        va="top",
        weight="bold",
    )

    # Descriptions kept to one short line each so nothing wraps into the box below.
    variants = [
        ("C0", "nn.Identity", None, "no token mixing", "48.0%", "50.6k"),
        ("C1", "depthwise", "Conv2d 3x3", "local prior", "72.1%", "52.3k"),
        ("C2s", "Hyena2D", "static kernel", "global, no input-dep.", "85.5%", "144k"),
        ("C2f", "Hyena2D", "+ FiLM(z(x))", "global, input-dependent", "87.3%", "344k"),
    ]
    x0, w, gap = 0.35, 2.15, 0.29
    for i, (tag, head, sub, desc, acc, prm) in enumerate(variants):
        x = x0 + i * (w + gap)
        hl = tag == "C2f"
        box(ax, (x, 1.50), w, 1.10, head, sub=sub, fill=TRAIN_FILL, edge=TRAIN_EDGE, lw=2.4 if hl else 1.5, fs=10)
        ax.text(x + w / 2, 2.78, tag, ha="center", fontsize=12.5, color=INK, weight="bold" if hl else "normal")
        ax.text(x + w / 2, 1.30, desc, ha="center", fontsize=8.4, color=MUTED, va="top")
        ax.text(x + w / 2, 0.70, acc, ha="center", fontsize=14, color=INK, weight="bold" if hl else "normal")
        ax.text(x + w / 2, 0.44, f"{prm} trainable", ha="center", fontsize=8.4, color=MUTED, **MONO)

    ax.text(
        0.05,
        0.06,
        "frozen backbone, 784 tokens, 3 seeds  |  linear-probe floor 25.2%  |  pretrained model on this split: 12.7%",
        fontsize=9,
        color=MUTED,
    )


def main(output_dir: Path) -> None:
    """Render and save the two-panel diagram."""
    fig = plt.figure(figsize=(12.2, 10.0), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[7.4, 3.5], hspace=0.10, left=0.02, right=0.98, top=0.97, bottom=0.03)
    panel_a(fig.add_subplot(gs[0]))
    panel_b(fig.add_subplot(gs[1]))

    for ext in ("png", "svg"):
        out = output_dir / f"adapter_diagram.{ext}"
        fig.savefig(out, dpi=170, facecolor="white")
        print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    main(a.output_dir)
