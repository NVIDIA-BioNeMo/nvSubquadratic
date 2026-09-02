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

"""Render sample canvases from the spatial-recall classification task.

What: a 2x5 grid of task samples — top row ``placement="fixed"`` (the pretrain
distribution), bottom row ``placement="random"`` (the fine-tuning distribution) —
annotated with the readout region, the single read-out patch token, and the
integer class label.

Hardware: CPU only, no GPU required.

Invoke::

    PYTHONPATH=. python reports/hyena2d_parallel_adapter/plot_task_samples.py

Output: ``task_samples.png`` written next to this script (override with
``--output-dir``).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from examples.hyena2d_parallel_adapter import _base_lg as B
from experiments.datamodules.mnist import MNISTDataModule
from experiments.datamodules.spatial_recall_classification import (
    SpatialRecallClassificationDataset,
)


# Recessive annotation ink (skill: annotations must not compete with the data).
READOUT_INK = "#C2410C"  # readout region outline
TOKEN_INK = "#0369A1"  # the single corner token actually read
TEXT_INK = "#1F2937"
MUTED_INK = "#6B7280"


def build_dataset(placement: str, seed: int) -> SpatialRecallClassificationDataset:
    """Build the classification spatial-recall dataset used by the experiment.

    Args:
        placement: ``"fixed"`` (pretrain) or ``"random"`` (fine-tune).
        seed: Seed for the placement generator.

    Returns:
        Dataset yielding ``(canvas [C, H, W], int class label)``.
    """
    base = MNISTDataModule(
        data_dir=".data/mnist",
        batch_size=1,
        data_type="image",
        num_workers=0,
        pin_memory=False,
        use_deterministic_worker_init=False,
        seed=seed,
        task="classification",
    )
    base.setup("fit")
    return SpatialRecallClassificationDataset(
        base_dataset=base.train_dataset,
        target_size=B.TARGET_SIZE,
        canvas_size=B.CANVAS_SIZE,
        generator=torch.Generator().manual_seed(seed),
        placement=placement,
        with_mask=False,
        readout_value=B.READOUT_VALUE,
    )


def main(output_dir: Path, n_cols: int = 5, seed: int = 7) -> None:
    """Render and save the sample grid.

    Args:
        output_dir: Directory to write ``task_samples.png`` into.
        n_cols: Number of samples per row.
        seed: Base seed controlling digit placement.
    """
    rows = [
        ("fixed", "PRETRAIN\nplacement='fixed'"),
        ("random", "FINE-TUNE\nplacement='random'"),
    ]

    fig, axes = plt.subplots(2, n_cols, figsize=(2.05 * n_cols, 5.8), gridspec_kw={"hspace": 0.28})
    C, T, P = B.CANVAS_SIZE, B.TARGET_SIZE, B.PATCH_SIZE

    for r, (placement, row_title) in enumerate(rows):
        ds = build_dataset(placement, seed)
        for c in range(n_cols):
            ax = axes[r, c]
            canvas, label = ds[c]
            img = canvas[0].numpy()

            ax.imshow(img, cmap="gray", vmin=B.READOUT_VALUE, vmax=float(img.max()), interpolation="nearest")

            # Readout region: the 16x16 corner filled with readout_value.
            ax.add_patch(
                Rectangle(
                    (C - T - 0.5, C - T - 0.5),
                    T,
                    T,
                    fill=False,
                    edgecolor=READOUT_INK,
                    lw=1.6,
                    linestyle="--",
                )
            )
            # The single patch token the classifier actually reads.
            ax.add_patch(
                Rectangle(
                    (C - P - 0.5, C - P - 0.5),
                    P,
                    P,
                    fill=False,
                    edgecolor=TOKEN_INK,
                    lw=2.0,
                )
            )

            ax.set_title(f"label = {label}", fontsize=11, color=TEXT_INK, pad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#D1D5DB")

            if c == 0:
                ax.text(
                    -0.16,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    color=TEXT_INK,
                    linespacing=1.5,
                )

    fig.suptitle(
        "Spatial-recall classification task  ·  "
        f"{C}x{C} canvas, patch {P} -> {C // P}x{C // P} = {(C // P) ** 2} tokens",
        fontsize=12.5,
        color=TEXT_INK,
        y=0.975,
    )
    fig.text(
        0.5,
        0.072,
        f"dashed {chr(9633)} readout region ({T}x{T} px, filled with {B.READOUT_VALUE})     "
        f"solid {chr(9633)} the ONE patch token the classifier reads (index {(C // P) ** 2 - 1})",
        ha="center",
        fontsize=9.5,
        color=MUTED_INK,
    )
    fig.text(
        0.5,
        0.028,
        "Same task in both rows — only digit placement changes. The model must route "
        "digit identity to the bottom-right token.",
        ha="center",
        fontsize=9.5,
        color=MUTED_INK,
    )

    # tight_layout would collapse the inter-row gap that keeps row-2 titles
    # clear of row-1 images; place axes explicitly instead.
    fig.subplots_adjust(left=0.045, right=0.99, top=0.90, bottom=0.135, wspace=0.06, hspace=0.28)
    out = output_dir / "task_samples.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")

    # Value ranges matter for reading the figure — report them explicitly.
    ds = build_dataset("random", seed)
    canvas, label = ds[0]
    a = canvas[0].numpy()
    bg = a[T + 4, T + 4]
    print(f"\nsample: canvas {tuple(canvas.shape)}, label {label} (int, 0-9)")
    print(f"  readout region value : {a[-1, -1]:+.3f}")
    print(f"  background value     : {bg:+.3f}   (MNIST norm: (0-0.1307)/0.3081)")
    print(f"  digit stroke max     : {a.max():+.3f}")
    print(f"  digit occupies       : {T}x{T} px = {T // P}x{T // P} patches")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args.output_dir)
