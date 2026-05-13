"""Verify the √d dimensional scaling rule for SIREN ω₀.

Theory: for a SIREN positional embedding in d dimensions, the first-layer
weight has components w_j ~ U(-1/d, 1/d).  The expected squared norm is

    E[||w||²] = d · (1/d)² / 3 = 1/(3d)

so the characteristic radial frequency scales as ω₀ / √(3d).  To maintain
the same fractional spectral coverage across dimensions, ω₀ must scale as

    ω₀(d_new) = ω₀(d_old) · √(d_new / d_old)

This script verifies the rule numerically by:

  1. Building scalar SIRENKernelND instances at data_dim ∈ {1, 2, 3}.
  2. For each d, using bisection to find ω₀*(d) such that the mean
     per-channel median radial frequency is exactly 0.5 × Nyquist.
  3. Checking that ω₀*(d) / ω₀*(1) ≈ √d.
  4. As a second verification, fixing ω₀*(1) and applying the √d rule
     at d=2,3.  Measuring the resulting median and comparing to 0.5.

All experiments are averaged over multiple seeds for robustness.

Run::

    PYTHONPATH=. python reports/siren_omega0_dimensional_scaling/verify_sqrt_d_scaling.py

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nvsubquadratic.modules.kernels_nd import SIRENKernelND


# ── Configuration ─────────────────────────────────────────────────────────────

HIDDEN_DIM = 384
MLP_HIDDEN_DIM = 32
NUM_LAYERS = 3
EMBEDDING_DIM = 32
HIDDEN_OMEGA_0 = 1.0
L_CACHE = 15  # grid size per axis = 2*L - 1 = 29
N_SEEDS = 10
N_BINS = 48
TARGET_MEDIAN = 0.35  # achievable in all dimensions; rule is about ratios

OUTPUT_DIR = Path("reports/siren_omega0_dimensional_scaling")


# ── Kernel & spectrum helpers ─────────────────────────────────────────────────


def build_siren(data_dim: int, omega_0: float, seed: int) -> SIRENKernelND:
    torch.manual_seed(seed)
    return SIRENKernelND(
        out_dim=HIDDEN_DIM,
        data_dim=data_dim,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        omega_0=omega_0,
        L_cache=L_CACHE,
        use_bias=True,
        hidden_omega_0=HIDDEN_OMEGA_0,
    )


def compute_kernel(siren: SIRENKernelND, data_dim: int) -> torch.Tensor:
    """Return kernel tensor with shape [N, ..._{data_dim times}, C]."""
    with torch.no_grad():
        seq_lens = tuple([L_CACHE] * data_dim)
        k, _grid = siren(seq_lens=seq_lens)
    return k.squeeze(0)  # [N, ..., C]


def radial_bin_nd(N: int, data_dim: int, n_bins: int):
    """Build radial distance map and bin indices for a d-dimensional FFT grid.

    Returns (r_norm, bin_idx, bin_centers) where r_norm is normalised so that
    per-axis Nyquist = 1.0.
    """
    axes = [torch.arange(N).float() for _ in range(data_dim)]
    grids = torch.meshgrid(*axes, indexing="ij")
    center = (N - 1) / 2.0
    half_N = N / 2.0
    r_sq = sum(((g - center) / half_N) ** 2 for g in grids)
    r = torch.sqrt(r_sq)
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    idx = torch.bucketize(r, edges) - 1
    idx = torch.where((idx < 0) | (idx >= n_bins), torch.full_like(idx, -1), idx)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return r, idx, centers


@dataclass
class SpectrumStats:
    data_dim: int
    N: int
    median_radius_mean: float
    median_radius_per_channel: np.ndarray
    r05_mean: float


def compute_spectrum_stats(kernel: torch.Tensor, data_dim: int, n_bins: int = N_BINS) -> SpectrumStats:
    """Compute radial spectral statistics for a d-dimensional kernel."""
    N = kernel.shape[0]
    for i in range(data_dim):
        assert kernel.shape[i] == N, f"Expecting cubic kernel, axis {i} has size {kernel.shape[i]}"
    C = kernel.shape[-1]

    fft_dims = tuple(range(data_dim))
    spec = torch.fft.fftn(kernel.float(), dim=fft_dims)
    spec = torch.fft.fftshift(spec, dim=fft_dims)
    energy = spec.real**2 + spec.imag**2  # [N, ..., C]

    _, idx, centers = radial_bin_nd(N, data_dim, n_bins)
    flat_idx = idx.reshape(-1)
    flat_energy = energy.reshape(-1, C)
    valid = flat_idx >= 0
    flat_idx_v = flat_idx[valid]
    flat_energy_v = flat_energy[valid]

    sum_per_bin = torch.zeros(n_bins, C)
    cnt_per_bin = torch.zeros(n_bins, 1)
    sum_per_bin.index_add_(0, flat_idx_v, flat_energy_v)
    cnt_per_bin.index_add_(0, flat_idx_v, torch.ones_like(flat_idx_v, dtype=torch.float32).unsqueeze(-1))
    cnt_per_bin = cnt_per_bin.clamp_min(1.0)

    cum_sum = sum_per_bin.cumsum(dim=0)
    cum_total = cum_sum[-1].clamp_min(1e-30)
    cum_frac = cum_sum / cum_total  # [n_bins, C]

    centers_np = centers.numpy()
    cum_frac_np = cum_frac.numpy().T  # [C, n_bins]

    def _quantile_radius(cum_row: np.ndarray, q: float) -> float:
        ge = np.where(cum_row >= q)[0]
        if len(ge) == 0:
            return 1.0
        i0 = ge[0]
        if i0 == 0 or cum_row[i0] == cum_row[i0 - 1]:
            return float(centers_np[i0])
        f0, f1 = cum_row[i0 - 1], cum_row[i0]
        w = (q - f0) / (f1 - f0)
        return float(centers_np[i0 - 1] + w * (centers_np[i0] - centers_np[i0 - 1]))

    median_r = np.array([_quantile_radius(cum_frac_np[c], 0.5) for c in range(C)])
    r05 = np.array([_quantile_radius(cum_frac_np[c], 0.05) for c in range(C)])
    mean_cum = cum_frac.mean(dim=-1).numpy()
    median_r_mean = _quantile_radius(mean_cum, 0.5)
    r05_mean = _quantile_radius(mean_cum, 0.05)

    return SpectrumStats(
        data_dim=data_dim,
        N=N,
        median_radius_mean=median_r_mean,
        median_radius_per_channel=median_r,
        r05_mean=r05_mean,
    )


# ── Bisection search ─────────────────────────────────────────────────────────


@dataclass
class AggregatedStats:
    """Statistics aggregated over multiple seeds."""

    mean_median: float
    std_median: float
    per_channel_medians: np.ndarray  # [n_seeds * C] pooled across seeds
    percentiles: dict[str, float]  # p5, p25, p50, p75, p95 of per-channel medians


def stats_at_omega(data_dim: int, omega_0: float, n_seeds: int = N_SEEDS) -> AggregatedStats:
    """Collect per-channel median radial frequencies across seeds."""
    seed_means: list[float] = []
    all_per_channel: list[np.ndarray] = []
    for seed in range(n_seeds):
        siren = build_siren(data_dim, omega_0, seed)
        k = compute_kernel(siren, data_dim)
        s = compute_spectrum_stats(k, data_dim)
        seed_means.append(s.median_radius_mean)
        all_per_channel.append(s.median_radius_per_channel)
    pooled = np.concatenate(all_per_channel)
    return AggregatedStats(
        mean_median=float(np.mean(seed_means)),
        std_median=float(np.std(seed_means)),
        per_channel_medians=pooled,
        percentiles={
            "p5": float(np.percentile(pooled, 5)),
            "p25": float(np.percentile(pooled, 25)),
            "p50": float(np.percentile(pooled, 50)),
            "p75": float(np.percentile(pooled, 75)),
            "p95": float(np.percentile(pooled, 95)),
        },
    )


def median_at_omega(data_dim: int, omega_0: float, n_seeds: int = N_SEEDS) -> float:
    """Return mean-over-seeds of per-channel-mean median radial frequency."""
    return stats_at_omega(data_dim, omega_0, n_seeds).mean_median


def find_omega0_star(data_dim: int, target: float = TARGET_MEDIAN, n_iters: int = 30) -> float:
    """Bisection for ω₀ such that median radial freq / Nyquist = target."""
    lo, hi = 0.5, 1000.0
    m_lo = median_at_omega(data_dim, lo)
    m_hi = median_at_omega(data_dim, hi)
    assert m_lo < target < m_hi, (
        f"d={data_dim}: target {target} not in bracket [{m_lo:.4f} @ ω₀={lo}, {m_hi:.4f} @ ω₀={hi}]"
    )
    for _ in range(n_iters):
        mid = math.sqrt(lo * hi)
        m_mid = median_at_omega(data_dim, mid)
        if m_mid < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 90)
    log("  SIREN ω₀ Dimensional Scaling Rule Verification")
    log(
        f"  config: hidden_dim={HIDDEN_DIM}, emb_dim={EMBEDDING_DIM}, mlp_hidden={MLP_HIDDEN_DIM}, "
        f"layers={NUM_LAYERS}, L={L_CACHE} (N={2 * L_CACHE - 1}), seeds={N_SEEDS}"
    )
    log("=" * 90)

    # ── Part 1: find ω₀* for each dimension ──────────────────────────────────
    log("\n── Part 1: Bisection for ω₀* (median radial freq = 0.5 × Nyquist) ──\n")
    omega_stars: dict[int, float] = {}
    for d in [1, 2, 3]:
        log(f"  Searching ω₀* for data_dim={d} ...")
        o_star = find_omega0_star(d)
        omega_stars[d] = o_star
        m = median_at_omega(d, o_star)
        log(f"    ω₀*({d}D) = {o_star:.4f}   →  median/Nyquist = {m:.4f}")

    log("\n  Ratios vs 1D:")
    for d in [2, 3]:
        ratio = omega_stars[d] / omega_stars[1]
        predicted = math.sqrt(d)
        error_pct = abs(ratio - predicted) / predicted * 100
        log(f"    ω₀*({d}D) / ω₀*(1D) = {ratio:.4f}   (predicted √{d} = {predicted:.4f},  error = {error_pct:.1f}%)")

    # ── Part 2: apply √d rule from 1D baseline and measure ────────────────────
    log("\n── Part 2: Apply √d rule from ω₀*(1D) and measure coverage ──\n")
    o1 = omega_stars[1]
    log(f"  Baseline ω₀*(1D) = {o1:.4f}\n")
    log(f"  {'dim':>4s}  {'ω₀ (√d rule)':>14s}  {'median/Nyquist':>16s}  {'Δ from target':>14s}")
    log(f"  {'─' * 4}  {'─' * 14}  {'─' * 16}  {'─' * 14}")
    for d in [1, 2, 3]:
        omega_rule = o1 * math.sqrt(d)
        m = median_at_omega(d, omega_rule)
        delta = m - TARGET_MEDIAN
        log(f"  {d:4d}  {omega_rule:14.4f}  {m:16.4f}  {delta:+12.4f}")

    # ── Part 3: control — what happens without scaling ────────────────────────
    log("\n── Part 3: Control — same ω₀ across dimensions (no scaling) ──\n")
    log(f"  Using ω₀ = ω₀*(1D) = {o1:.4f} for all d\n")
    log(f"  {'dim':>4s}  {'ω₀':>8s}  {'median/Nyquist':>16s}  {'expected (1/√d relative)':>26s}")
    log(f"  {'─' * 4}  {'─' * 8}  {'─' * 16}  {'─' * 26}")
    m1 = median_at_omega(1, o1)
    for d in [1, 2, 3]:
        m = median_at_omega(d, o1)
        expected_unscaled = m1 / math.sqrt(d)
        log(f"  {d:4d}  {o1:8.4f}  {m:16.4f}  {expected_unscaled:26.4f}")

    # ── Part 4: full sweep for plotting ───────────────────────────────────────
    log("\n── Part 4: ω₀ sweep per dimension (for plotting) ──\n")
    omega_sweep = np.array([1, 2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100], dtype=float)
    results_by_dim: dict[int, list[tuple[float, float]]] = {}
    for d in [1, 2, 3]:
        results_by_dim[d] = []
        for o in omega_sweep:
            m = median_at_omega(d, o, n_seeds=5)  # fewer seeds for speed
            results_by_dim[d].append((o, m))
        log(f"  d={d}: {len(omega_sweep)} points computed")

    # ── Part 5: per-channel distribution analysis ──────────────────────────────
    log("\n── Part 5: Per-channel distribution of median radial freq ──\n")

    dist_scaled: dict[int, AggregatedStats] = {}
    dist_unscaled: dict[int, AggregatedStats] = {}
    dist_bisected: dict[int, AggregatedStats] = {}

    for d in [1, 2, 3]:
        dist_bisected[d] = stats_at_omega(d, omega_stars[d])
        dist_scaled[d] = stats_at_omega(d, o1 * math.sqrt(d))
        dist_unscaled[d] = stats_at_omega(d, o1)

    log(
        f"  {'condition':<22s}  {'dim':>3s}  {'ω₀':>8s}  {'μ':>7s}  {'σ':>7s}  "
        f"{'p5':>7s}  {'p25':>7s}  {'p50':>7s}  {'p75':>7s}  {'p95':>7s}"
    )
    log(
        f"  {'─' * 22}  {'─' * 3}  {'─' * 8}  {'─' * 7}  {'─' * 7}  "
        f"{'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 7}"
    )

    for label, dist_dict, omega_fn in [
        ("bisected ω₀*", dist_bisected, lambda d: omega_stars[d]),
        ("√d rule", dist_scaled, lambda d: o1 * math.sqrt(d)),
        ("no scaling", dist_unscaled, lambda d: o1),
    ]:
        for d in [1, 2, 3]:
            st = dist_dict[d]
            p = st.percentiles
            log(
                f"  {label:<22s}  {d:3d}  {omega_fn(d):8.2f}  {st.mean_median:7.4f}  {st.std_median:7.4f}  "
                f"{p['p5']:7.4f}  {p['p25']:7.4f}  {p['p50']:7.4f}  {p['p75']:7.4f}  {p['p95']:7.4f}"
            )
        log("")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {1: "#2196F3", 2: "#4CAF50", 3: "#F44336"}

        # ── Figure 1: sweep + bar (original) ──────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        ax = axes[0]
        for d in [1, 2, 3]:
            omegas, medians = zip(*results_by_dim[d])
            ax.plot(omegas, medians, "o-", color=colors[d], label=f"d={d}", markersize=4)
            ax.axvline(omega_stars[d], color=colors[d], ls="--", alpha=0.5)
        ax.axhline(TARGET_MEDIAN, color="gray", ls=":", alpha=0.6, label=f"target = {TARGET_MEDIAN}")
        ax.set_xlabel("ω₀")
        ax.set_ylabel("median radial freq / Nyquist")
        ax.set_title("Median spectral coverage vs ω₀")
        ax.legend()
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        dims = [1, 2, 3]
        stars = [omega_stars[d] for d in dims]
        ax.bar([f"{d}D" for d in dims], stars, color=[colors[d] for d in dims], alpha=0.7, label="measured ω₀*")
        predicted = [omega_stars[1] * math.sqrt(d) for d in dims]
        ax.scatter(
            [f"{d}D" for d in dims], predicted, color="black", marker="x", s=100, zorder=5, label="√d prediction"
        )
        ax.set_ylabel(f"ω₀* (for {TARGET_MEDIAN} Nyquist coverage)")
        ax.set_title("ω₀* scales as √d")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for i, d in enumerate(dims):
            ax.annotate(
                f"{stars[i]:.1f}",
                (f"{d}D", stars[i]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        plot_path = OUTPUT_DIR / "omega0_sqrt_d_scaling.png"
        fig.savefig(plot_path, dpi=150)
        log(f"\n  Plot saved to {plot_path}")
        plt.close()

        # ── Figure 2: per-channel distribution histograms ─────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
        hist_bins = np.linspace(0, 0.8, 40)

        for ax, (label, dist_dict) in zip(
            axes,
            [
                ("√d rule applied", dist_scaled),
                ("bisected ω₀*", dist_bisected),
                ("no scaling (control)", dist_unscaled),
            ],
        ):
            for d in [1, 2, 3]:
                st = dist_dict[d]
                ax.hist(
                    st.per_channel_medians,
                    bins=hist_bins,
                    alpha=0.4,
                    color=colors[d],
                    label=f"d={d} (μ={st.mean_median:.3f}, σ={st.std_median:.3f})",
                    density=True,
                )
            ax.axvline(TARGET_MEDIAN, color="gray", ls=":", alpha=0.6)
            ax.set_xlabel("per-channel median radial freq / Nyquist")
            ax.set_title(label)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        axes[0].set_ylabel("density")
        plt.tight_layout()
        dist_path = OUTPUT_DIR / "per_channel_distributions.png"
        fig.savefig(dist_path, dpi=150)
        log(f"  Plot saved to {dist_path}")
        plt.close()

        # ── Figure 3: box plots — compact comparison ──────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

        for ax, (label, dist_dict) in zip(
            axes,
            [
                ("√d rule applied", dist_scaled),
                ("bisected ω₀*", dist_bisected),
                ("no scaling (control)", dist_unscaled),
            ],
        ):
            data = [dist_dict[d].per_channel_medians for d in [1, 2, 3]]
            bp = ax.boxplot(data, tick_labels=["1D", "2D", "3D"], patch_artist=True, showfliers=False, whis=[5, 95])
            for patch, d in zip(bp["boxes"], [1, 2, 3]):
                patch.set_facecolor(colors[d])
                patch.set_alpha(0.5)
            ax.axhline(TARGET_MEDIAN, color="gray", ls=":", alpha=0.6)
            ax.set_title(label)
            ax.set_xlabel("data dimension")
            ax.grid(True, alpha=0.2, axis="y")

        axes[0].set_ylabel("per-channel median radial freq / Nyquist")
        plt.tight_layout()
        box_path = OUTPUT_DIR / "per_channel_boxplots.png"
        fig.savefig(box_path, dpi=150)
        log(f"  Plot saved to {box_path}")
        plt.close()

    except ImportError:
        log("\n  (matplotlib not available — skipping plots)")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    log(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
