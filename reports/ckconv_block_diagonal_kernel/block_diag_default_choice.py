"""Side-by-side per-channel median histograms to lock down the production
defaults for the block-diagonal SIREN init.

This is the same set of variants as ``multiomega_block_diag_withmask_aligned.png``
but with histograms placed in their own panels (no overlay) so the shape of
each distribution can be read directly.  It also adds the *chosen* candidate
default (linear ω₀ ∈ [1, 16], block-diag off=0, K=8) so all other variants
are compared against the actual proposed default.

Aligned mask is used everywhere — that was the right choice for any
ω₀-ordered init.

Variants:
  (1) baseline: single ω₀ = 8.355                                  [reference]
  (2) per-row dense, log[1, 12]                                    [no block-diag]
  (3) block-diag, log[1, 12], off=0.5                              [partial block]
  (4) block-diag, log[1, 12], off=0.1                              [near-strict]
  (5) block-diag, log[1, 12], off=0.0                              [strict block]
  (6) K=8 parallel SIRENs, log[1, 12]                              [reference]
  (7) block-diag, linear[1, 16], off=0.0                           [PROPOSED DEFAULT]

For each variant we report:
  μ              mean of per-channel median radial freq (Nyquist-normalized)
  σ              spread of per-channel medians (higher = more diversified)
  r_05_mean      mean of per-channel 5th-percentile radial freq
  %ch r_05<0.05  fraction of channels with substantial DC content

Run::

    PYTHONPATH=. python _tmp/spectrum_analysis/block_diag_default_choice.py
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
    build_siren,
    compute_spectrum_stats,
)
from multiband_omega0 import MultiBandSIREN, build_aligned_mask
from multiomega_classes import (
    BlockDiagonalMultiOmegaSIRENKernelND,
    build_multiomega_siren_clean,
)


NUM_BLOCKS = 8
LOG_OMEGA_MIN = 1.0
LOG_OMEGA_MAX = 12.0
LIN_OMEGA_MIN = 1.0
LIN_OMEGA_MAX = 16.0  # the candidate default


# ─── Builders ───────────────────────────────────────────────────────────────


def _build_block_diag(
    omega_per_block: np.ndarray,
    *,
    off_block_scale: float,
    L: int,
    seed: int,
) -> BlockDiagonalMultiOmegaSIRENKernelND:
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


def _build_per_row_log(L: int, *, seed: int):
    omega_per_row = np.logspace(math.log10(LOG_OMEGA_MIN), math.log10(LOG_OMEGA_MAX), PR_KERNEL_EMBEDDING_DIM)
    return build_multiomega_siren_clean(omega_per_row, L_cache=L, seed=seed)


def _build_k_parallel(L: int, *, seed: int) -> MultiBandSIREN:
    torch.manual_seed(seed)
    return MultiBandSIREN(
        K=NUM_BLOCKS,
        omega0_min=LOG_OMEGA_MIN,
        omega0_max=LOG_OMEGA_MAX,
        L_cache=L,
        out_dim=PR_HIDDEN_DIM,
    )


# ─── Stats accumulator ─────────────────────────────────────────────────────


def _kernel_2d(siren, mask, L: int) -> torch.Tensor:
    with torch.no_grad():
        k, grid = siren(seq_lens=(L, L))
        if mask is not None:
            k = mask(grid=grid, x=k)
    return k.squeeze(0)


def _avg_stats(builder, mask_fn, L: int, n_seeds: int) -> dict:
    mu, sg, r05m, plow = [], [], [], []
    medians_all, r05_all = [], []
    cum_curves = []
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


# ─── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--baseline-omega0", type=float, default=8.355)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    L = PR_BASE_L
    N = 2 * L - 1
    print(f"L_base = {L}   N = {N}   K = {NUM_BLOCKS}")
    print(f"n_seeds per variant = {args.n_seeds}")

    aligned_mask = lambda: build_aligned_mask(
        L_cache=L,
        K=NUM_BLOCKS,
        channels_per_band=[PR_HIDDEN_DIM // NUM_BLOCKS] * NUM_BLOCKS,
    )

    log_per_block = np.logspace(math.log10(LOG_OMEGA_MIN), math.log10(LOG_OMEGA_MAX), NUM_BLOCKS)
    lin_per_block = np.linspace(LIN_OMEGA_MIN, LIN_OMEGA_MAX, NUM_BLOCKS)

    # Each entry: (display_name, color, builder, kind_for_subtitle)
    PROPOSED_COLOR = "#17becf"  # cyan — visually distinct from the reds/greens
    variants = [
        (
            "(1) single ω₀ = 8.355",
            "#1f77b4",
            lambda L, seed: build_siren(omega_0=args.baseline_omega0, L_cache=L, seed=seed),
            "baseline (no per-row, no block)",
        ),
        ("(2) per-row dense  log[1, 12]", "#ff7f0e", _build_per_row_log, "off-block = 1.0  (full mixing)"),
        (
            "(3) block-diag  log[1, 12]  off=0.5",
            "#2ca02c",
            lambda L, seed: _build_block_diag(log_per_block, off_block_scale=0.5, L=L, seed=seed),
            "partial mixing",
        ),
        (
            "(4) block-diag  log[1, 12]  off=0.1",
            "#9467bd",
            lambda L, seed: _build_block_diag(log_per_block, off_block_scale=0.1, L=L, seed=seed),
            "near-strict mixing",
        ),
        (
            "(5) block-diag  log[1, 12]  off=0.0",
            "#d62728",
            lambda L, seed: _build_block_diag(log_per_block, off_block_scale=0.0, L=L, seed=seed),
            "strict block-diag at init",
        ),
        ("(6) K=8 parallel SIRENs  log[1, 12]", "#8c564b", _build_k_parallel, "reference — K independent SIRENs"),
        (
            "(7) block-diag  linear[1, 16]  off=0.0  ★ proposed default",
            PROPOSED_COLOR,
            lambda L, seed: _build_block_diag(lin_per_block, off_block_scale=0.0, L=L, seed=seed),
            "linear schedule, full Nyquist coverage on N=29",
        ),
    ]

    print("\nComputing variants...")
    results = []
    for name, color, builder, subtitle in variants:
        r = _avg_stats(builder, aligned_mask, L, args.n_seeds)
        results.append((name, color, subtitle, r))
        print(
            f"  {name:<60} μ={r['mu']:.3f}  σ={r['sigma']:.3f}  "
            f"r05={r['r05_mean']:.3f}  %low={100 * r['pct_low']:.1f}%"
        )

    # ── Table ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'variant':<62} {'μ':>6} {'σ':>14} {'r_05':>6} {'% r_05<0.05':>12}")
    print("-" * 100)
    for name, _color, _sub, r in results:
        print(
            f"{name:<62} {r['mu']:>6.3f} {r['sigma']:>6.3f}±{r['sigma_se']:.3f} "
            f"{r['r05_mean']:>6.3f} {100 * r['pct_low']:>11.1f}%"
        )
    print("=" * 100)

    # ── Side-by-side histograms (one panel per variant) ─────────────────────
    n = len(results)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.3 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    bins = np.linspace(0.0, 1.0, 41)

    # Common y-limit so heights are comparable across panels.
    max_density = 0.0
    densities = []
    for _, _, _, r in results:
        d, _ = np.histogram(r["medians_pooled"], bins=bins, density=True)
        densities.append(d)
        max_density = max(max_density, float(d.max()))

    for ax_idx, (name, color, subtitle, r) in enumerate(results):
        i, j = divmod(ax_idx, ncols)
        ax = axes[i, j]
        ax.hist(
            r["medians_pooled"], bins=bins, color=color, alpha=0.85, density=True, edgecolor="black", linewidth=0.3
        )
        ax.axvline(r["mu"], color="black", linestyle="--", linewidth=0.9, label=f"μ = {r['mu']:.3f}")
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, max_density * 1.10)
        ax.set_title(f"{name}\n{subtitle}", fontsize=8.5)
        # Stats annotation in upper-right
        stats_txt = f"σ = {r['sigma']:.3f}\nr₀₅ = {r['r05_mean']:.3f}\n%low = {100 * r['pct_low']:.1f}%"
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
        if i == nrows - 1:
            ax.set_xlabel("Per-channel median radial frequency / Nyquist", fontsize=9)
        if j == 0:
            ax.set_ylabel("Density", fontsize=9)
        ax.grid(alpha=0.25)

    # Hide unused panels.
    for ax_idx in range(n, nrows * ncols):
        i, j = divmod(ax_idx, ncols)
        axes[i, j].axis("off")

    fig.suptitle(
        f"Side-by-side per-channel median histograms — choosing the production default\n"
        f"K={NUM_BLOCKS}  •  N={N}  •  aligned mask  •  {args.n_seeds} seeds pooled  •  "
        f"vertical dashed = μ  •  vertical dotted = 0.5·Nyquist",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    out_path = args.output_dir / "block_diag_default_choice_hist.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ wrote {out_path}")


if __name__ == "__main__":
    main()
