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

"""Verify the resolution scaling rule  ω₀(m·N) = m · ω₀(N)  on the block-diag
variant with the user's chosen defaults (off=0.1, linear[1, 12] at base).

We build the block-diag SIREN + aligned mask at three resolution multipliers
``m ∈ {1, 2, 4}`` under two regimes:

  - SCALED (rule applied): at multiplier m, ω₀_per_block = m · [1, 12]   schedule.
    Mask is rebuilt natively at the new grid_size (so init_std_low tracks the
    denser grid).  If the rule holds, per-channel median histograms in
    Nyquist-normalized coordinates should collapse onto each other.
  - UNSCALED (control): ω₀_per_block stays at [1, 12] at every m.  This is the
    "wrong" recipe; histograms should shift LEFT as m grows because the same
    continuous content occupies a smaller fraction of the new Nyquist.

Linear schedule, K=8, off_block_scale=0.1, aligned mask.

Run::

    PYTHONPATH=. python _tmp/spectrum_analysis/block_diag_scaling_verify.py
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
from multiomega_classes import BlockDiagonalMultiOmegaSIRENKernelND


NUM_BLOCKS = 8
OMEGA0_MIN_BASE = 1.0
OMEGA0_MAX_BASE = 12.0
OFF_BLOCK_SCALE = 0.1


def _build(omega_per_block: np.ndarray, *, L: int, seed: int) -> BlockDiagonalMultiOmegaSIRENKernelND:
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
        off_block_scale=OFF_BLOCK_SCALE,
        hidden_omega_0=PR_KERNEL_HIDDEN_OMEGA_0,
    )


def _make_aligned_mask(L_cache: int):
    return build_aligned_mask(
        L_cache=L_cache,
        K=NUM_BLOCKS,
        channels_per_band=[PR_HIDDEN_DIM // NUM_BLOCKS] * NUM_BLOCKS,
    )


def _kernel_2d(siren, mask, L):
    with torch.no_grad():
        k, grid = siren(seq_lens=(L, L))
        if mask is not None:
            k = mask(grid=grid, x=k)
    return k.squeeze(0)


def _avg_stats(omega_per_block: np.ndarray, L: int, n_seeds: int) -> dict:
    mu, sg, r05m, plow = [], [], [], []
    medians_all = []
    for s in range(n_seeds):
        siren = _build(omega_per_block, L=L, seed=s)
        mask = _make_aligned_mask(L_cache=L)
        k = _kernel_2d(siren, mask, L)
        st = compute_spectrum_stats(k)
        mu.append(st.median_radius_per_channel.mean())
        sg.append(st.median_radius_per_channel.std())
        r05m.append(st.r05_per_channel.mean())
        plow.append(float(np.mean(st.r05_per_channel < 0.05)))
        medians_all.append(st.median_radius_per_channel)
    return dict(
        mu=float(np.mean(mu)),
        sigma=float(np.mean(sg)),
        sigma_se=float(np.std(sg) / math.sqrt(n_seeds)),
        r05_mean=float(np.mean(r05m)),
        pct_low=float(np.mean(plow)),
        medians_pooled=np.concatenate(medians_all, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--multipliers", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"K = {NUM_BLOCKS}  off_block_scale = {OFF_BLOCK_SCALE}  "
        f"base schedule = linear[{OMEGA0_MIN_BASE}, {OMEGA0_MAX_BASE}]  "
        f"n_seeds = {args.n_seeds}"
    )
    print(f"L_base = {PR_BASE_L}  (N_base = {2 * PR_BASE_L - 1})")

    results: dict[tuple[str, int], dict] = {}
    for m in args.multipliers:
        L_m = PR_BASE_L * m
        N_m = 2 * L_m - 1
        # SCALED: apply the rule — multiply the whole schedule by m.
        om_scaled = np.linspace(m * OMEGA0_MIN_BASE, m * OMEGA0_MAX_BASE, NUM_BLOCKS)
        r_scaled = _avg_stats(om_scaled, L_m, args.n_seeds)
        r_scaled["omega_per_block"] = om_scaled
        r_scaled["L"] = L_m
        r_scaled["N"] = N_m
        results[("scaled", m)] = r_scaled

        # UNSCALED (control): keep base schedule at every resolution.
        om_fixed = np.linspace(OMEGA0_MIN_BASE, OMEGA0_MAX_BASE, NUM_BLOCKS)
        r_fixed = _avg_stats(om_fixed, L_m, args.n_seeds)
        r_fixed["omega_per_block"] = om_fixed
        r_fixed["L"] = L_m
        r_fixed["N"] = N_m
        results[("unscaled", m)] = r_fixed

        print(f"\nm = {m}  (L = {L_m}, N = {N_m})")
        print(
            f"  SCALED   ω₀∈[{m * OMEGA0_MIN_BASE:.1f}, {m * OMEGA0_MAX_BASE:.1f}]  "
            f"μ={r_scaled['mu']:.3f}  σ={r_scaled['sigma']:.3f}  "
            f"r₀₅={r_scaled['r05_mean']:.3f}  %low={100 * r_scaled['pct_low']:.1f}%"
        )
        print(
            f"  UNSCALED ω₀∈[{OMEGA0_MIN_BASE:.1f}, {OMEGA0_MAX_BASE:.1f}]  "
            f"μ={r_fixed['mu']:.3f}  σ={r_fixed['sigma']:.3f}  "
            f"r₀₅={r_fixed['r05_mean']:.3f}  %low={100 * r_fixed['pct_low']:.1f}%"
        )

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print(f"{'regime':>10} {'m':>3} {'L':>4} {'N':>4} {'ω₀ range':>14}   {'μ':>6} {'σ':>6} {'r₀₅':>6} {'%low':>6}")
    print("-" * 96)
    for regime in ("scaled", "unscaled"):
        for m in args.multipliers:
            r = results[(regime, m)]
            om = r["omega_per_block"]
            print(
                f"{regime:>10} {m:>3} {r['L']:>4} {r['N']:>4} "
                f"[{om[0]:>5.1f}, {om[-1]:>5.1f}]   "
                f"{r['mu']:>6.3f} {r['sigma']:>6.3f} {r['r05_mean']:>6.3f} "
                f"{100 * r['pct_low']:>5.1f}%"
            )
    print("=" * 96)
    # Scaling rule strength: max absolute deviation of (μ, σ) across m in the scaled regime.
    mus_scaled = np.array([results[("scaled", m)]["mu"] for m in args.multipliers])
    sgs_scaled = np.array([results[("scaled", m)]["sigma"] for m in args.multipliers])
    print("\n  SCALED regime collapse (what the rule predicts):")
    print(
        f"    μ    across m: mean={mus_scaled.mean():.4f}  "
        f"max|Δ|={float(np.max(np.abs(mus_scaled - mus_scaled.mean()))):.4f}  "
        f"(relative {100 * float(np.max(np.abs(mus_scaled - mus_scaled.mean()))) / mus_scaled.mean():.2f}%)"
    )
    print(
        f"    σ    across m: mean={sgs_scaled.mean():.4f}  "
        f"max|Δ|={float(np.max(np.abs(sgs_scaled - sgs_scaled.mean()))):.4f}  "
        f"(relative {100 * float(np.max(np.abs(sgs_scaled - sgs_scaled.mean()))) / sgs_scaled.mean():.2f}%)"
    )
    mus_unscaled = np.array([results[("unscaled", m)]["mu"] for m in args.multipliers])
    print("\n  UNSCALED control (what the rule predicts should SHIFT with m):")
    print(f"    μ    across m: {mus_unscaled}  (expected to drop as ≈ 1/m)")
    # If the rule is exact, μ_unscaled(m) ≈ μ_unscaled(1) / m
    ideal_drop = mus_unscaled[0] / np.array(args.multipliers, dtype=float)
    print(f"    predicted μ/m: {ideal_drop}  (theory if all content is below Nyquist at m=1)")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 41)
    colors = {1: "#1f77b4", 2: "#2ca02c", 4: "#d62728"}

    # Common y limit
    ymax = 0.0
    for r in results.values():
        d, _ = np.histogram(r["medians_pooled"], bins=bins, density=True)
        ymax = max(ymax, float(d.max()))

    # LEFT: scaled regime (rule applied)
    ax = axes[0]
    for m in args.multipliers:
        r = results[("scaled", m)]
        om_low, om_high = r["omega_per_block"][0], r["omega_per_block"][-1]
        ax.hist(
            r["medians_pooled"],
            bins=bins,
            density=True,
            color=colors[m],
            alpha=0.45,
            edgecolor=colors[m],
            linewidth=1.0,
            histtype="stepfilled",
            label=(f"m = {m}   ω₀ ∈ [{om_low:.0f}, {om_high:.0f}]   μ={r['mu']:.3f}  σ={r['sigma']:.3f}"),
        )
        ax.axvline(r["mu"], color=colors[m], linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax * 1.1)
    ax.set_xlabel("Per-channel median radial freq / Nyquist")
    ax.set_ylabel("Density")
    ax.set_title("SCALED: ω₀ scaled by m  (rule applied — expect collapse)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # RIGHT: unscaled control
    ax = axes[1]
    for m in args.multipliers:
        r = results[("unscaled", m)]
        ax.hist(
            r["medians_pooled"],
            bins=bins,
            density=True,
            color=colors[m],
            alpha=0.45,
            edgecolor=colors[m],
            linewidth=1.0,
            histtype="stepfilled",
            label=(f"m = {m}   ω₀ ∈ [1, 12] (fixed)   μ={r['mu']:.3f}  σ={r['sigma']:.3f}"),
        )
        ax.axvline(r["mu"], color=colors[m], linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax * 1.1)
    ax.set_xlabel("Per-channel median radial freq / Nyquist")
    ax.set_title("UNSCALED: ω₀ fixed at [1, 12]  (control — expect left-shift)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Scaling rule verification  ω₀_max(m·N) = m · ω₀_max(N)\n"
        f"block-diag off={OFF_BLOCK_SCALE}  •  linear schedule  •  K={NUM_BLOCKS}  •  "
        f"aligned mask (re-init'd at each resolution)  •  {args.n_seeds} seeds pooled",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    out_path = args.output_dir / "block_diag_scaling_verify.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ wrote {out_path}")


if __name__ == "__main__":
    main()
