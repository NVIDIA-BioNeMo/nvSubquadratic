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

"""Per-channel median histogram for {linear, log} ω₀ schedule × {per-row dense, block-diag off=0}.

A 2×2 grid of histograms, no overlapping variants per subplot:

    rows = schedule type   (linear  ω₀ ∈ [1, ω₀_max] vs log ω₀ ∈ [1, ω₀_max])
    cols = variant         (per-row dense MLP   vs   block-diag off=0)

Both variants use the SAME ω₀_min, ω₀_max, embedding_dim, and (for block-diag)
the SAME number of blocks K.  Histograms pool per-channel medians across
``--n-seeds`` random seeds and use the same x-axis range so the four panels
are directly comparable.

Run::

    PYTHONPATH=. python _tmp/spectrum_analysis/schedule_vs_variant_hist.py \\
        --omega0-max 12 --n-seeds 10 --output-dir _tmp/spectrum_analysis
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).parent))
from analyze_kernel_spectrum import (
    PR_BASE_L,
    PR_DATA_DIM,
    PR_HIDDEN_DIM,
    PR_KERNEL_EMBEDDING_DIM,
    PR_KERNEL_HIDDEN_OMEGA_0,
    PR_KERNEL_MLP_HIDDEN_DIM,
    PR_KERNEL_NUM_LAYERS,
    compute_spectrum_stats,
)
from multiband_omega0 import build_aligned_mask
from multiomega_classes import (
    BlockDiagonalMultiOmegaSIRENKernelND,
    build_multiomega_siren_clean,
)


NUM_BLOCKS_DEFAULT = 8

COLOR_PR = "#ff7f0e"  # orange — per-row dense
COLOR_BD = "#d62728"  # red    — block-diag off=0


def _kernel_2d(siren, mask, L: int) -> torch.Tensor:
    with torch.no_grad():
        k, grid = siren(seq_lens=(L, L))
        if mask is not None:
            k = mask(grid=grid, x=k)
    return k.squeeze(0)


def _schedule(kind: str, omega_min: float, omega_max: float, n: int) -> np.ndarray:
    if kind == "linear":
        return np.linspace(omega_min, omega_max, n)
    if kind == "log":
        return np.logspace(math.log10(omega_min), math.log10(omega_max), n)
    raise ValueError(kind)


def _build_per_row(schedule_kind: str, omega_min: float, omega_max: float, *, L: int, seed: int):
    omega_per_row = _schedule(schedule_kind, omega_min, omega_max, PR_KERNEL_EMBEDDING_DIM)
    return build_multiomega_siren_clean(omega_per_row, L_cache=L, seed=seed)


def _build_block_diag(schedule_kind: str, omega_min: float, omega_max: float, *, L: int, num_blocks: int, seed: int):
    omega_per_block = _schedule(schedule_kind, omega_min, omega_max, num_blocks)
    torch.manual_seed(seed)
    return BlockDiagonalMultiOmegaSIRENKernelND(
        out_dim=PR_HIDDEN_DIM,
        data_dim=PR_DATA_DIM,
        mlp_hidden_dim=PR_KERNEL_MLP_HIDDEN_DIM,
        num_layers=PR_KERNEL_NUM_LAYERS,
        embedding_dim=PR_KERNEL_EMBEDDING_DIM,
        L_cache=L,
        use_bias=True,
        omega_0_per_block=omega_per_block,
        num_blocks=num_blocks,
        off_block_scale=0.0,
        hidden_omega_0=PR_KERNEL_HIDDEN_OMEGA_0,
    )


def _pool_medians(builder, mask_factory, *, L: int, n_seeds: int) -> np.ndarray:
    medians = []
    for s in range(n_seeds):
        siren = builder(seed=s, L=L)
        mask = mask_factory() if mask_factory is not None else None
        k = _kernel_2d(siren, mask, L)
        st = compute_spectrum_stats(k)
        medians.append(st.median_radius_per_channel)
    return np.concatenate(medians, axis=0)


def _make_figure(omega_max: float, num_blocks: int, n_seeds: int, with_mask: bool, L: int, out_path: Path) -> None:
    omega_min = 1.0
    K = num_blocks
    channels_per_block = PR_HIDDEN_DIM // K
    tag = "withmask_aligned" if with_mask else "nomask"

    mask_factory = (
        (lambda: build_aligned_mask(L_cache=L, K=K, channels_per_band=[channels_per_block] * K)) if with_mask else None
    )

    # Compute pooled medians for all 4 (schedule × variant) cells.
    cells = {
        ("linear", "per-row dense"): _pool_medians(
            lambda seed, L: _build_per_row("linear", omega_min, omega_max, L=L, seed=seed),
            mask_factory,
            L=L,
            n_seeds=n_seeds,
        ),
        ("linear", "block-diag off=0"): _pool_medians(
            lambda seed, L: _build_block_diag("linear", omega_min, omega_max, L=L, num_blocks=K, seed=seed),
            mask_factory,
            L=L,
            n_seeds=n_seeds,
        ),
        ("log", "per-row dense"): _pool_medians(
            lambda seed, L: _build_per_row("log", omega_min, omega_max, L=L, seed=seed),
            mask_factory,
            L=L,
            n_seeds=n_seeds,
        ),
        ("log", "block-diag off=0"): _pool_medians(
            lambda seed, L: _build_block_diag("log", omega_min, omega_max, L=L, num_blocks=K, seed=seed),
            mask_factory,
            L=L,
            n_seeds=n_seeds,
        ),
    }

    # Print summary
    print(f"\n[{tag}] ω₀ range = [{omega_min}, {omega_max}], K = {K}, n_seeds = {n_seeds}")
    print(f"  {'schedule':<10} {'variant':<22} {'μ':>8} {'σ':>8} {'%ch <0.1':>10}")
    for (sched, variant), arr in cells.items():
        mu = arr.mean()
        sg = arr.std()
        pct_low = 100.0 * float(np.mean(arr < 0.1))
        print(f"  {sched:<10} {variant:<22} {mu:>8.4f} {sg:>8.4f} {pct_low:>9.1f}%")

    # Plot 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 41)
    schedules = ["linear", "log"]
    variants = ["per-row dense", "block-diag off=0"]
    colors = {"per-row dense": COLOR_PR, "block-diag off=0": COLOR_BD}

    for i, sched in enumerate(schedules):
        for j, variant in enumerate(variants):
            ax = axes[i, j]
            arr = cells[(sched, variant)]
            ax.hist(
                arr,
                bins=bins,
                alpha=0.85,
                color=colors[variant],
                density=True,
                label=f"μ={arr.mean():.3f}, σ={arr.std():.3f}",
            )
            ax.set_title(f"{sched} schedule  —  {variant}", fontsize=11)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9, loc="upper right")
            if i == 1:
                ax.set_xlabel("Per-channel median radial frequency / Nyquist")
            if j == 0:
                ax.set_ylabel("Density")

    fig.suptitle(
        f"Per-channel median histogram — schedule × variant  "
        f"(ω₀ ∈ [{omega_min}, {omega_max}], K={K}, N={2 * L - 1}, "
        f"mean over {n_seeds} seeds, {tag})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--omega0-max", type=float, default=12.0)
    parser.add_argument("--num-blocks", type=int, default=NUM_BLOCKS_DEFAULT)
    parser.add_argument("--n-seeds", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    L = PR_BASE_L
    print(f"L = {L}, N = {2 * L - 1}")

    for with_mask in (False, True):
        tag = "withmask_aligned" if with_mask else "nomask"
        out_path = args.output_dir / f"schedule_vs_variant_omega{int(args.omega0_max)}_{tag}.png"
        _make_figure(
            omega_max=args.omega0_max,
            num_blocks=args.num_blocks,
            n_seeds=args.n_seeds,
            with_mask=with_mask,
            L=L,
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
