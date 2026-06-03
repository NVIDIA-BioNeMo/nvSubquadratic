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

"""Sweep ω₀_max for the linear-schedule block-diag (off=0) variant.

Goal: pick ω₀_max so the per-channel median histogram covers the entire
[0, 1] range up to Nyquist on the PR's CKConv grid (L_cache=15 → N=29).

For each ω₀_max in --omega0-max-list (default {12, 16, 20, 24, 32, 48}):
  - Build block-diag off=0.0 with linear ω₀ schedule [1, ω₀_max], K=8
  - Apply aligned Gaussian mask (widest σ on lowest-ω₀ block)
  - Pool per-channel medians across --n-seeds seeds
  - Print per-block stats (which blocks alias)
  - Plot a row of histograms with the same x/y range so the user can pick

Output: a single figure with one panel per ω₀_max value.

Run::

    PYTHONPATH=. python _tmp/spectrum_analysis/linear_blockdiag_omega_sweep.py \\
        --omega0-max-list 12 16 20 24 32 48 --n-seeds 10 \\
        --output-dir _tmp/spectrum_analysis
"""

from __future__ import annotations

import argparse
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
from multiomega_classes import BlockDiagonalMultiOmegaSIRENKernelND


NUM_BLOCKS = 8
NYQUIST_OMEGA0 = 12.0  # Single-scalar Nyquist on N=29.
COLOR_BD = "#d62728"


def _kernel_2d(siren, mask, L: int) -> torch.Tensor:
    with torch.no_grad():
        k, grid = siren(seq_lens=(L, L))
        if mask is not None:
            k = mask(grid=grid, x=k)
    return k.squeeze(0)


def _build_block_diag_linear(omega_max: float, *, L: int, num_blocks: int, seed: int):
    omega_per_block = np.linspace(1.0, omega_max, num_blocks)
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


def _stats_for_omega_max(
    omega_max: float, *, L: int, K: int, channels_per_block: int, n_seeds: int, mask_factory
) -> dict:
    medians_pooled = []
    band_meds = [[] for _ in range(K)]
    for s in range(n_seeds):
        siren = _build_block_diag_linear(omega_max, L=L, num_blocks=K, seed=s)
        mask = mask_factory()
        k = _kernel_2d(siren, mask, L)
        st = compute_spectrum_stats(k)
        medians_pooled.append(st.median_radius_per_channel)
        for b in range(K):
            c0, c1 = b * channels_per_block, (b + 1) * channels_per_block
            band_meds[b].append(st.median_radius_per_channel[c0:c1])
    return dict(
        omega_max=omega_max,
        medians_pooled=np.concatenate(medians_pooled, axis=0),
        band_med_means=[np.concatenate(b, axis=0).mean() for b in band_meds],
        omega_per_block=np.linspace(1.0, omega_max, K),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--omega0-max-list", type=float, nargs="+", default=[12.0, 16.0, 20.0, 24.0, 32.0, 48.0])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--num-blocks", type=int, default=NUM_BLOCKS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    L = PR_BASE_L
    K = args.num_blocks
    channels_per_block = PR_HIDDEN_DIM // K
    print(f"L = {L}, N = {2 * L - 1}, K = {K}, channels/block = {channels_per_block}")
    print(f"Sweep ω₀_max ∈ {args.omega0_max_list},  n_seeds = {args.n_seeds}")
    print("Schedule: LINEAR ω₀ ∈ [1, ω₀_max]   Mask: aligned   Variant: block-diag off=0.0")

    mask_factory = lambda: build_aligned_mask(L_cache=L, K=K, channels_per_band=[channels_per_block] * K)

    results = [
        _stats_for_omega_max(
            om, L=L, K=K, channels_per_block=channels_per_block, n_seeds=args.n_seeds, mask_factory=mask_factory
        )
        for om in args.omega0_max_list
    ]

    # ── Print a table ────────────────────────────────────────────────────────
    print(
        f"\n{'ω₀_max':>8} {'#alias':>8} {'μ':>8} {'σ':>8} {'p10':>7} {'p50':>7} {'p90':>7} "
        f"{'p99':>7}  {'block-medians':<48}"
    )
    for r in results:
        m = r["medians_pooled"]
        n_alias = int(np.sum(r["omega_per_block"] > NYQUIST_OMEGA0))
        bms = "  ".join(f"{x:.2f}" for x in r["band_med_means"])
        print(
            f"{r['omega_max']:>8.1f} {n_alias:>8d} {m.mean():>8.4f} {m.std():>8.4f} "
            f"{np.percentile(m, 10):>7.3f} {np.percentile(m, 50):>7.3f} "
            f"{np.percentile(m, 90):>7.3f} {np.percentile(m, 99):>7.3f}  [{bms}]"
        )

    # ── Plot row of histograms ───────────────────────────────────────────────
    n = len(results)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    bins = np.linspace(0.0, 1.0, 41)

    for idx, r in enumerate(results):
        i, j = idx // ncols, idx % ncols
        ax = axes[i, j]
        m = r["medians_pooled"]
        n_alias = int(np.sum(r["omega_per_block"] > NYQUIST_OMEGA0))
        ax.hist(m, bins=bins, alpha=0.85, color=COLOR_BD, density=True, label=f"μ={m.mean():.3f}, σ={m.std():.3f}")
        # Mark per-block mean median locations as vertical ticks
        for b, bm in enumerate(r["band_med_means"]):
            ax.axvline(bm, color="black", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_title(f"ω₀_max = {r['omega_max']:.0f}   ({n_alias}/{NUM_BLOCKS} blocks alias)")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
        if i == nrows - 1:
            ax.set_xlabel("Per-channel median radial frequency / Nyquist")
        if j == 0:
            ax.set_ylabel("Density")

    fig.suptitle(
        f"Linear schedule + block-diag off=0  —  ω₀_max sweep  "
        f"(K={K}, N={2 * L - 1}, aligned mask, mean over {args.n_seeds} seeds)\n"
        f"black dotted lines = per-block mean median; "
        f"single-scalar Nyquist threshold ≈ ω₀ = {NYQUIST_OMEGA0:.0f}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = args.output_dir / "linear_blockdiag_omega_sweep.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ wrote {out_path}")


if __name__ == "__main__":
    main()
