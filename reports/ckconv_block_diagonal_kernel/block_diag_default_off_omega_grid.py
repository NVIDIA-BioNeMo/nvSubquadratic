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

"""4x4 grid of per-channel median histograms over
``off_block_scale ∈ {0.0, 0.1} × ω₀_max ∈ {12, 14, 16, 18}``.

Linear ω₀ schedule, K=8 blocks, aligned mask.  Top two rows are the production
grid (L=15 → N=29).  Bottom two rows evaluate the *same continuous SIREN +
mask* on a 2× denser grid (L=29 → N=57) and rescale the x-axis to the new
(dense) Nyquist, so that the vertical line at 0.5 marks the *old* Nyquist.

Aliasing read-off (on the dense rows): any mass to the right of x=0.5 is
continuous content above the old Nyquist that gets folded back on N=29.  The
panel annotation ``alias%`` is the fraction of channels whose median sits past
0.5·new_Nyquist.

Run::

    PYTHONPATH=. python _tmp/spectrum_analysis/block_diag_default_off_omega_grid.py
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
OMEGA0_MIN = 1.0


def _build_block_diag(omega_per_block, *, off_block_scale, L, seed):
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


def _kernel_2d(siren, mask, L):
    with torch.no_grad():
        k, grid = siren(seq_lens=(L, L))
        if mask is not None:
            k = mask(grid=grid, x=k)
    return k.squeeze(0)


def _make_aligned_mask(L_cache: int):
    return build_aligned_mask(
        L_cache=L_cache,
        K=NUM_BLOCKS,
        channels_per_band=[PR_HIDDEN_DIM // NUM_BLOCKS] * NUM_BLOCKS,
    )


def _make_aligned_mask_continuous(L_target: int, ref_mask: torch.nn.Module) -> torch.nn.Module:
    """Build a Gaussian mask at L_target whose ``std_param`` is copied byte-for-byte
    from ``ref_mask``.  This evaluates the *same continuous Gaussians* on a denser
    grid (rather than re-initializing min_std/init_std_low for the new grid).
    """
    mask = _make_aligned_mask(L_cache=L_target)
    with torch.no_grad():
        mask.std_param.copy_(ref_mask.std_param)
    return mask


def _avg_stats(builder, mask_fn_base, L_base, L_dense, n_seeds):
    """Compute per-channel median histograms on both base and 2x-dense grids.

    Returns a dict with keys ``base`` and ``dense``, each containing the same
    summary fields (``mu``, ``sigma``, ``sigma_se``, ``r05_mean``, ``pct_low``,
    ``medians_pooled``).  The dense run additionally has ``alias_frac`` = fraction
    of per-channel medians that sit past 0.5·new_Nyquist (= old_Nyquist).
    """
    base_mu, base_sg, base_r05, base_plow = [], [], [], []
    dense_mu, dense_sg, dense_r05, dense_plow, dense_alias = [], [], [], [], []
    base_medians_all, dense_medians_all = [], []
    for s in range(n_seeds):
        # Base resolution.
        siren_base = builder(L=L_base, seed=s)
        mask_base = mask_fn_base()
        k_base = _kernel_2d(siren_base, mask_base, L_base)
        st_base = compute_spectrum_stats(k_base)
        base_mu.append(st_base.median_radius_per_channel.mean())
        base_sg.append(st_base.median_radius_per_channel.std())
        base_r05.append(st_base.r05_per_channel.mean())
        base_plow.append(float(np.mean(st_base.r05_per_channel < 0.05)))
        base_medians_all.append(st_base.median_radius_per_channel)

        # Dense resolution: same SIREN (params are L_cache-independent → same seed
        # rebuilds identical params on a denser grid), same continuous mask
        # (std_param copied from base mask).
        siren_dense = builder(L=L_dense, seed=s)
        mask_dense = _make_aligned_mask_continuous(L_dense, mask_base)
        k_dense = _kernel_2d(siren_dense, mask_dense, L_dense)
        st_dense = compute_spectrum_stats(k_dense)
        dense_mu.append(st_dense.median_radius_per_channel.mean())
        dense_sg.append(st_dense.median_radius_per_channel.std())
        dense_r05.append(st_dense.r05_per_channel.mean())
        dense_plow.append(float(np.mean(st_dense.r05_per_channel < 0.05)))
        dense_alias.append(float(np.mean(st_dense.median_radius_per_channel > 0.5)))
        dense_medians_all.append(st_dense.median_radius_per_channel)

    def _pack(mu, sg, r05, plow, medians_all, alias=None):
        out = dict(
            mu=float(np.mean(mu)),
            sigma=float(np.mean(sg)),
            sigma_se=float(np.std(sg) / math.sqrt(n_seeds)),
            r05_mean=float(np.mean(r05)),
            pct_low=float(np.mean(plow)),
            medians_pooled=np.concatenate(medians_all, axis=0),
        )
        if alias is not None:
            out["alias_frac"] = float(np.mean(alias))
        return out

    return dict(
        base=_pack(base_mu, base_sg, base_r05, base_plow, base_medians_all),
        dense=_pack(dense_mu, dense_sg, dense_r05, dense_plow, dense_medians_all, alias=dense_alias),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--omega0-max-list", type=float, nargs="+", default=[12.0, 14.0, 16.0, 18.0])
    parser.add_argument("--off-list", type=float, nargs="+", default=[0.0, 0.1])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    L_base = PR_BASE_L
    L_dense = 2 * L_base - 1  # → N_dense = 2*L_dense - 1 = 4*L_base - 3 = 57 (≈ 2× of N_base=29)
    N_base = 2 * L_base - 1
    N_dense = 2 * L_dense - 1
    print(
        f"L_base = {L_base} (N_base = {N_base})   "
        f"L_dense = {L_dense} (N_dense = {N_dense}, ≈{N_dense / N_base:.2f}× of base)"
    )
    print(f"K = {NUM_BLOCKS}  schedule = linear[{OMEGA0_MIN}, ω₀_max]  n_seeds = {args.n_seeds}")

    # Color per off_block_scale (consistent across columns).
    OFF_COLORS = {0.0: "#d62728", 0.1: "#9467bd"}

    print("\nComputing variants (base + 2× dense)...")
    grid: dict[tuple[float, float], dict] = {}
    for off in args.off_list:
        for om in args.omega0_max_list:
            omega_per_block = np.linspace(OMEGA0_MIN, om, NUM_BLOCKS)
            stats = _avg_stats(
                lambda L, seed, _omega=omega_per_block, _off=off: _build_block_diag(
                    _omega, off_block_scale=_off, L=L, seed=seed
                ),
                lambda: _make_aligned_mask(L_base),
                L_base,
                L_dense,
                args.n_seeds,
            )
            stats["omega_per_block"] = omega_per_block
            grid[(off, om)] = stats
            r_b, r_d = stats["base"], stats["dense"]
            print(
                f"  off={off:>3}  ω₀_max={om:>5}  "
                f"BASE  μ={r_b['mu']:.3f} σ={r_b['sigma']:.3f} r05={r_b['r05_mean']:.3f} "
                f"%low={100 * r_b['pct_low']:.1f}%   "
                f"DENSE μ={r_d['mu']:.3f} σ={r_d['sigma']:.3f} "
                f"alias%={100 * r_d['alias_frac']:.1f}%"
            )

    # ── Table ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 130)
    print(
        f"{'off':>5} {'ω₀_max':>7} | "
        f"{'BASE μ':>7} {'BASE σ':>7} {'r_05':>7} {'%low':>6}  | "
        f"{'DENSE μ':>8} {'DENSE σ':>8} {'r_05':>7} {'%low':>6} {'alias%':>7}  | "
        f"per-block ω₀"
    )
    print("-" * 130)
    for off in args.off_list:
        for om in args.omega0_max_list:
            stats = grid[(off, om)]
            r_b, r_d = stats["base"], stats["dense"]
            ob = ", ".join(f"{x:.1f}" for x in stats["omega_per_block"])
            print(
                f"{off:>5.2f} {om:>7.1f} | "
                f"{r_b['mu']:>7.3f} {r_b['sigma']:>7.3f} {r_b['r05_mean']:>7.3f} "
                f"{100 * r_b['pct_low']:>5.1f}% | "
                f"{r_d['mu']:>8.3f} {r_d['sigma']:>8.3f} {r_d['r05_mean']:>7.3f} "
                f"{100 * r_d['pct_low']:>5.1f}% {100 * r_d['alias_frac']:>6.1f}% | "
                f"[{ob}]"
            )
    print("=" * 130)

    # ── 4x4 histogram grid (rows: base@off=0, base@off=0.1, dense@off=0, dense@off=0.1) ─
    rows: list[tuple[str, str, float, str]] = []  # (kind ∈ {base, dense}, off, label, axis_label)
    for off in args.off_list:
        rows.append(("base", off, f"L = {L_base}  •  N = {N_base}\noff = {off}"))
    for off in args.off_list:
        rows.append(("dense", off, f"L = {L_dense}  •  N = {N_dense}  (2×)\noff = {off}"))

    nrows = len(rows)
    ncols = len(args.omega0_max_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    bins = np.linspace(0.0, 1.0, 41)

    # Common y-limit so heights are comparable across panels.
    max_density = 0.0
    for stats in grid.values():
        for kind in ("base", "dense"):
            d, _ = np.histogram(stats[kind]["medians_pooled"], bins=bins, density=True)
            max_density = max(max_density, float(d.max()))

    for i, (kind, off, ylabel) in enumerate(rows):
        for j, om in enumerate(args.omega0_max_list):
            ax = axes[i, j]
            r = grid[(off, om)][kind]
            color = OFF_COLORS[off]
            # Slightly fade the dense rows so the stark "alias" mass past x=0.5 reads as commentary.
            alpha = 0.85 if kind == "base" else 0.65
            ax.hist(
                r["medians_pooled"],
                bins=bins,
                color=color,
                alpha=alpha,
                density=True,
                edgecolor="black",
                linewidth=0.3,
            )
            ax.axvline(r["mu"], color="black", linestyle="--", linewidth=0.9, label=f"μ = {r['mu']:.3f}")
            if kind == "base":
                # Vertical line at 0.5·Nyquist (just a reference for symmetry with dense rows).
                ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.6, alpha=0.7)
                stats_txt = f"σ = {r['sigma']:.3f}\nr₀₅ = {r['r05_mean']:.3f}\n%low = {100 * r['pct_low']:.1f}%"
            else:
                # On the dense axis, x=0.5 = old (base-grid) Nyquist.  Mass to the
                # right of that line is content that was aliased on the base grid.
                ax.axvline(0.5, color="darkred", linestyle="-", linewidth=1.2, alpha=0.9, label="old Nyquist")
                ax.axvspan(0.5, 1.0, color="red", alpha=0.06, zorder=-1)
                stats_txt = f"σ = {r['sigma']:.3f}\nr₀₅ = {r['r05_mean']:.3f}\nalias% = {100 * r['alias_frac']:.1f}%"
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, max_density * 1.10)
            ax.text(
                0.97,
                0.97,
                stats_txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="lightgray", alpha=0.85),
            )
            ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
            if i == 0:
                ax.set_title(f"ω₀_max = {int(om)}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{ylabel}\n\nDensity", fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel("Per-channel median radial freq / Nyquist", fontsize=9)
            ax.grid(alpha=0.25)

    fig.suptitle(
        f"Block-diag SIREN init  •  off ∈ {args.off_list} × ω₀_max ∈ "
        f"{[int(o) for o in args.omega0_max_list]}\n"
        f"top 2 rows: base grid (N={N_base})    bottom 2 rows: same continuous "
        f"SIREN+mask on 2× dense grid (N={N_dense})\n"
        f"on dense rows, dark-red line = old Nyquist; mass in the shaded region was "
        f"aliased on the base grid  •  K={NUM_BLOCKS}  •  {args.n_seeds} seeds pooled",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_path = args.output_dir / "block_diag_default_off_omega_grid.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ wrote {out_path}")


if __name__ == "__main__":
    main()
