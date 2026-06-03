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

"""Compare per-row ω₀ + block-diagonal MLP init against single-SIREN per-row
and against K parallel SIRENs.

Hypothesis (yours):
  The K-parallel-SIRENs win comes from having ZERO cross-correlation between
  band frequency-content at init.  We can replicate this in a SINGLE dense
  SIREN by initializing its hidden + output linears as block-diagonal (or
  block-rectangular) so each output channel only "sees" one band's worth of
  first-layer modes at init time.  Training can later fill in the cross-block
  weights via gradient flow.

We sweep ``off_block_scale ∈ {0.0, 0.1, 0.5, 1.0}`` (0.0 = strictly block-
diagonal at init = mathematically equivalent to K parallel SIRENs; 1.0 =
full dense init = parent ``MultiOmegaSIRENKernelND``).

Variants compared:
  (A) baseline single ω₀ = 8.355
  (B) per-row first-layer log[1, 12]   (full mixing, off_block_scale=1)
  (C) per-row + block-diag,  off_block_scale = 0.5
  (D) per-row + block-diag,  off_block_scale = 0.1
  (E) per-row + block-diag,  off_block_scale = 0.0    (≡ K parallel SIRENs at init)
  (F) K=8 parallel SIRENs reference (from ``multiband_omega0.MultiBandSIREN``)

K=8 blocks (so rows per block = embedding_dim/K = 4, hidden block size = 4×4,
output block = 48 channels per block).

Run::

    PYTHONPATH=. python _tmp/spectrum_analysis/multiomega_block_diag.py \\
        --output-dir _tmp/spectrum_analysis
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
    build_mask,
    build_siren,
    compute_spectrum_stats,
)
from multiband_omega0 import MultiBandSIREN, build_aligned_mask
from multiomega_classes import (
    BlockDiagonalMultiOmegaSIRENKernelND,
    build_multiomega_siren_clean,
)


NUM_BLOCKS = 8
OMEGA0_MIN = 1.0
OMEGA0_MAX = 12.0  # Stays within single-scalar Nyquist on N=29.

# Fixed palette — same blue/orange as previous figures for shared variants.
PALETTE = {
    "single": "#1f77b4",  # blue
    "per_row": "#ff7f0e",  # orange
    "blk_0p5": "#2ca02c",  # green
    "blk_0p1": "#9467bd",  # purple
    "blk_0": "#d62728",  # red  (strict block-diag at init)
    "k_parallel": "#8c564b",  # brown (K parallel SIRENs reference)
}


def _kernel_2d(siren, mask, L: int) -> torch.Tensor:
    with torch.no_grad():
        k, grid = siren(seq_lens=(L, L))
        if mask is not None:
            k = mask(grid=grid, x=k)
    return k.squeeze(0)


def _build_block_diag(off_block_scale: float, L: int, *, seed: int) -> BlockDiagonalMultiOmegaSIRENKernelND:
    omega_per_block = np.logspace(math.log10(OMEGA0_MIN), math.log10(OMEGA0_MAX), NUM_BLOCKS)
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
        num_blocks=NUM_BLOCKS,
        off_block_scale=off_block_scale,
        hidden_omega_0=PR_KERNEL_HIDDEN_OMEGA_0,
    )


def _build_per_row(L: int, *, seed: int):
    omega_per_row = np.logspace(0.0, math.log10(OMEGA0_MAX), PR_KERNEL_EMBEDDING_DIM)
    return build_multiomega_siren_clean(omega_per_row, L_cache=L, seed=seed)


def _build_k_parallel(L: int, *, seed: int) -> MultiBandSIREN:
    torch.manual_seed(seed)
    return MultiBandSIREN(
        K=NUM_BLOCKS,
        omega0_min=OMEGA0_MIN,
        omega0_max=OMEGA0_MAX,
        L_cache=L,
        out_dim=PR_HIDDEN_DIM,
    )


def _avg_stats(builder, mask_fn, L: int, n_seeds: int) -> dict:
    """Average per-channel μ, σ, r_05_mean, %ch_low across n_seeds."""
    mu, sg, r05m, plow = [], [], [], []
    cum_curves = []  # for plotting the channel-mean cumulative
    medians_all = []  # to pool per-channel medians for histogram
    r05_all = []
    radial_freq = None
    for s in range(n_seeds):
        siren = builder(L=L, seed=s)
        mask = mask_fn() if mask_fn is not None else None
        k = _kernel_2d(siren, mask, L)
        st = compute_spectrum_stats(k)
        mu.append(st.median_radius_per_channel.mean())
        sg.append(st.median_radius_per_channel.std())
        r05m.append(st.r05_per_channel.mean())
        plow.append(float(np.mean(st.r05_per_channel < 0.05)))
        cum_curves.append(st.cum_fraction_mean)
        medians_all.append(st.median_radius_per_channel)
        r05_all.append(st.r05_per_channel)
        radial_freq = st.radial_freq_norm
    return dict(
        mu=float(np.mean(mu)),
        sigma=float(np.mean(sg)),
        sigma_se=float(np.std(sg) / math.sqrt(n_seeds)),
        r05_mean=float(np.mean(r05m)),
        pct_low=float(np.mean(plow)),
        cum_curve_mean=np.mean(cum_curves, axis=0),
        radial_freq=radial_freq,
        medians_pooled=np.concatenate(medians_all, axis=0),
        r05_pooled=np.concatenate(r05_all, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--baseline-omega0", type=float, default=8.355)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    L = PR_BASE_L
    print(f"L_base = {L}   N = {2 * L - 1}")
    print(f"K (num_blocks) = {NUM_BLOCKS}, ω₀ range = [{OMEGA0_MIN}, {OMEGA0_MAX}]")
    print(f"n_seeds per variant = {args.n_seeds}")

    # The default GaussianModulationND init has std logspaced LOW->HIGH per channel, i.e.
    # ch 0 = narrowest spatial Gaussian (preserves only high-freq content),
    # ch N-1 = widest (near-identity).  For variants whose output channels carry an
    # implicit low->high frequency ordering (block-diag and K=8 parallel),
    # this default mask is REVERSED relative to the SIREN's per-channel ω₀:
    # low-ω₀ channels get killed by narrow masks.  ``build_aligned_mask`` flips
    # the std order so widest masks land on the lowest-ω₀ channels.
    #
    # For variants without channel ordering (single ω₀, per-row first-layer with
    # full dense MLP), the choice is irrelevant: any permutation yields the same
    # per-channel statistics in expectation.  We therefore always use the aligned
    # mask in the with-mask comparison so the ordered variants get a fair shake.
    mask_specs = [
        ("nomask", None),
        ("withmask_default", lambda: build_mask(L_cache=L)),
        (
            "withmask_aligned",
            lambda: build_aligned_mask(
                L_cache=L, K=NUM_BLOCKS, channels_per_band=[PR_HIDDEN_DIM // NUM_BLOCKS] * NUM_BLOCKS
            ),
        ),
    ]
    for tag, mask_fn in mask_specs:
        print(f"\n{'=' * 80}\n{tag}\n{'=' * 80}")

        # Build all variants
        results: dict[str, dict] = {}
        results["single ω₀ = 8.355"] = _avg_stats(
            lambda L, seed: build_siren(omega_0=args.baseline_omega0, L_cache=L, seed=seed),
            mask_fn,
            L,
            args.n_seeds,
        )
        results["per-row log[1, 12]  (no block)"] = _avg_stats(
            _build_per_row,
            mask_fn,
            L,
            args.n_seeds,
        )
        results["block-diag, off=0.5"] = _avg_stats(
            lambda L, seed: _build_block_diag(0.5, L, seed=seed),
            mask_fn,
            L,
            args.n_seeds,
        )
        results["block-diag, off=0.1"] = _avg_stats(
            lambda L, seed: _build_block_diag(0.1, L, seed=seed),
            mask_fn,
            L,
            args.n_seeds,
        )
        results["block-diag, off=0.0"] = _avg_stats(
            lambda L, seed: _build_block_diag(0.0, L, seed=seed),
            mask_fn,
            L,
            args.n_seeds,
        )
        results["K=8 parallel SIRENs"] = _avg_stats(
            _build_k_parallel,
            mask_fn,
            L,
            args.n_seeds,
        )

        # Print table
        print(f"\n{'variant':<36} {'μ':>7} {'σ':>14} {'r_05':>7} {'%ch low':>8}")
        for name, r in results.items():
            print(
                f"{name:<36} {r['mu']:>7.4f} {r['sigma']:>6.4f}±{r['sigma_se']:.3f} "
                f"{r['r05_mean']:>7.4f} {100 * r['pct_low']:>7.1f}%"
            )

        # Plot
        keys_in_order = list(results.keys())
        color_for = {
            "single ω₀ = 8.355": PALETTE["single"],
            "per-row log[1, 12]  (no block)": PALETTE["per_row"],
            "block-diag, off=0.5": PALETTE["blk_0p5"],
            "block-diag, off=0.1": PALETTE["blk_0p1"],
            "block-diag, off=0.0": PALETTE["blk_0"],
            "K=8 parallel SIRENs": PALETTE["k_parallel"],
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # 1) Cumulative energy (channel-mean, averaged across seeds)
        ax = axes[0, 0]
        for name, r in results.items():
            ax.plot(
                r["radial_freq"],
                r["cum_curve_mean"],
                color=color_for[name],
                linewidth=1.7,
                label=f"{name}  (μ={r['mu']:.3f})",
            )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Radial frequency / Nyquist")
        ax.set_ylabel("Cumulative fraction of energy")
        ax.set_title("Cumulative spectral energy (channel-mean)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # 2) Per-channel median histogram (pooled across all seeds)
        ax = axes[0, 1]
        bins = np.linspace(0.0, 1.0, 41)
        for name, r in results.items():
            ax.hist(
                r["medians_pooled"],
                bins=bins,
                alpha=0.4,
                density=True,
                color=color_for[name],
                label=f"{name}  (σ={r['sigma']:.3f})",
            )
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Per-channel median radial frequency / Nyquist")
        ax.set_ylabel("Density")
        ax.set_title(f"Per-channel median histogram  ({args.n_seeds} seeds pooled)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        # 3) σ across variants (bar chart with error bars)
        ax = axes[1, 0]
        sigmas = [results[k]["sigma"] for k in keys_in_order]
        sigmas_se = [results[k]["sigma_se"] for k in keys_in_order]
        colors = [color_for[k] for k in keys_in_order]
        x = np.arange(len(keys_in_order))
        ax.bar(x, sigmas, yerr=sigmas_se, capsize=4, color=colors, alpha=0.85)
        for xi, s, se in zip(x, sigmas, sigmas_se):
            ax.text(xi, s + se + 0.005, f"{s:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("  ", "\n") for k in keys_in_order], rotation=15, ha="right", fontsize=7)
        ax.set_ylabel("Per-channel median σ  (mean ± SE across seeds)")
        ax.set_title("Per-channel spread σ — does block-diag init close the gap?")
        ax.grid(alpha=0.3, axis="y")

        # 4) %ch r_05<0.05 across variants (bar chart)
        ax = axes[1, 1]
        plows = [100 * results[k]["pct_low"] for k in keys_in_order]
        ax.bar(x, plows, color=colors, alpha=0.85)
        for xi, p in zip(x, plows):
            ax.text(xi, p + 1, f"{p:.1f}%", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("  ", "\n") for k in keys_in_order], rotation=15, ha="right", fontsize=7)
        ax.set_ylabel("% channels with r_05 < 0.05")
        ax.set_title("Low-end coverage — same question")
        ax.grid(alpha=0.3, axis="y")

        fig.suptitle(
            f"Block-diagonal MLP init  ({tag}, K={NUM_BLOCKS}, ω₀∈[{OMEGA0_MIN}, {OMEGA0_MAX}], "
            f"N={2 * L - 1}, mean over {args.n_seeds} seeds)"
        )
        fig.tight_layout()
        out_path = args.output_dir / f"multiomega_block_diag_{tag}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\n→ wrote {out_path}")


if __name__ == "__main__":
    main()
