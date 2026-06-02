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

"""Standalone analysis of CKConv kernel spectrum vs resolution and omega_0.

This script reproduces the *exact* SIREN kernel + GaussianModulationND mask
configuration used in `examples/vit5_imagenet/vit5_hybrid/_base_config.py`
(see ``_make_hyena_block_cfg``) and asks two questions:

  1. **Resolution invariance.**  When we keep all SIREN/mask weights fixed but
     resample the underlying continuous kernel on a denser grid over the same
     normalized domain ``[-1, 1]^2`` (i.e. "2x" or "4x" resolution), what
     fraction of the *digital* spectrum does the kernel still cover?

  2. **omega_0 compensation.**  The SIREN's first-layer weights scale linearly
     with ``omega_0`` (see ``_init_siren_weights``), so the kernel's continuous
     spatial frequency content is governed by ``omega_0``.  Increasing the
     sample count without changing ``omega_0`` keeps the continuous content
     identical but pushes it into a smaller fraction of the new digital
     spectrum.  We expect the rule

         omega_0(N)  =  omega_0(N_base) * (N / N_base)

     to preserve spectral coverage as a fraction of Nyquist.  We verify this
     numerically by:

       (a) finding an ``omega_0_*`` such that, at the base resolution
           ``N_base = 2 * L_base - 1`` (≈ 29 in the PR config), the *median*
           radial frequency (the radius at which 50% of the kernel's spectral
           energy is enclosed) equals exactly half of Nyquist, then
       (b) verifying that, at 2x and 4x resolution with the *same* SIREN
           weights, the median radial frequency drops to ~0.25 / ~0.125 of the
           new Nyquist, and that scaling ``omega_0`` by 2 / 4 (with everything
           else fresh-initialized) restores the 50% target.

Run::

    PYTHONPATH=. python _tmp/spectrum_analysis/analyze_kernel_spectrum.py \\
        --output-dir _tmp/spectrum_analysis

Output: PNG plots and a text summary in ``--output-dir``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND


# ─── PR config defaults (copied from examples/vit5_imagenet/vit5_hybrid/_base_config.py) ──
PR_KERNEL_MLP_HIDDEN_DIM = 32
PR_KERNEL_NUM_LAYERS = 3
PR_KERNEL_EMBEDDING_DIM = 32
PR_KERNEL_OMEGA_0 = 10.0
PR_KERNEL_HIDDEN_OMEGA_0 = 1.0
PR_HIDDEN_DIM = 384
PR_DATA_DIM = 2

# At image_size=224, patch_size=16, num_registers=4:
#   num_patches = 14, num_non_pad = 14*14 + 1 + 4 = 201,
#   pad_size = (-201) % 14 = 9, total = 210, grid_h = 210 // 14 = 15
PR_BASE_L = 15  # → kernel grid is (2*L - 1) = 29 per axis (data_dim=2 → 29x29)

# Gaussian mask defaults
PR_MASK_MIN_ATTENUATION_AT_STEP = 0.1
PR_MASK_MAX_ATTENUATION_AT_LIMIT = 0.95
PR_MASK_INIT_EXTENT = 1.0
PR_MASK_PARAMETRIZATION = "direct"


# ─── Builders that match the PR exactly ──────────────────────────────────────


def build_siren(omega_0: float, L_cache: int, *, seed: int = 0) -> SIRENKernelND:
    """Build a SIRENKernelND with the PR's hyperparameters."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        # Use a non-default generator path: we re-seed PyTorch's RNG so all
        # internal `nn.Linear` initializations are deterministic and identical
        # for the same seed regardless of L_cache (only the grid_cache buffer
        # changes with L_cache, the parameter shapes do not).
        siren = SIRENKernelND(
            out_dim=PR_HIDDEN_DIM,
            data_dim=PR_DATA_DIM,
            mlp_hidden_dim=PR_KERNEL_MLP_HIDDEN_DIM,
            num_layers=PR_KERNEL_NUM_LAYERS,
            embedding_dim=PR_KERNEL_EMBEDDING_DIM,
            omega_0=omega_0,
            L_cache=L_cache,
            use_bias=True,
            hidden_omega_0=PR_KERNEL_HIDDEN_OMEGA_0,
        )
    _ = g  # unused (kept for API clarity; we control determinism via manual_seed above)
    return siren


def build_mask(L_cache: int) -> GaussianModulationND:
    """Build a GaussianModulationND with the PR's hyperparameters.

    grid_size matches what CKConvND auto-injects: ``2 * L - 1``.
    """
    return GaussianModulationND(
        data_dim=PR_DATA_DIM,
        num_channels=PR_HIDDEN_DIM,
        grid_size=2 * L_cache - 1,
        min_attenuation_at_step=PR_MASK_MIN_ATTENUATION_AT_STEP,
        max_attenuation_at_limit=PR_MASK_MAX_ATTENUATION_AT_LIMIT,
        init_extent=PR_MASK_INIT_EXTENT,
        parametrization=PR_MASK_PARAMETRIZATION,
    )


def upsample_siren(reference: SIRENKernelND, L_target: int) -> SIRENKernelND:
    """Build a SIREN at ``L_target`` that samples the *same continuous function*
    as ``reference`` on a denser grid over the same ``[-1, 1]^2`` domain.

    All learnable parameters (positional embedding linear + bias, hidden
    linears, output linear) are copied verbatim from ``reference``.  Only the
    grid_cache buffer (which controls the sampling positions) is rebuilt for
    the new ``L_target``.
    """
    new = SIRENKernelND(
        out_dim=reference.out_dim,
        data_dim=reference.data_dim,
        mlp_hidden_dim=reference.mlp_hidden_dim,
        num_layers=reference.num_layers,
        embedding_dim=reference.embedding_dim,
        omega_0=reference.omega_0,
        L_cache=L_target,
        use_bias=True,
        hidden_omega_0=reference.hidden_omega_0,
    )
    # Copy parameters; skip buffers (grid_cache is shape-dependent and is
    # already correct for L_target after construction).
    with torch.no_grad():
        new.positional_embedding.linear.weight.copy_(reference.positional_embedding.linear.weight)
        if new.positional_embedding.linear.bias is not None:
            new.positional_embedding.linear.bias.copy_(reference.positional_embedding.linear.bias)
        for ln_new, ln_ref in zip(new.hidden_linears, reference.hidden_linears):
            ln_new.weight.copy_(ln_ref.weight)
            if ln_new.bias is not None:
                ln_new.bias.copy_(ln_ref.bias)
        new.out_linear.weight.copy_(reference.out_linear.weight)
        if new.out_linear.bias is not None:
            new.out_linear.bias.copy_(reference.out_linear.bias)
    return new


# ─── Spectrum analysis utilities ─────────────────────────────────────────────


@dataclass
class SpectrumStats:
    """Per-resolution spectrum analysis result."""

    L: int  # L_cache used to generate the kernel
    N: int  # kernel grid size per axis (= 2L - 1)
    radial_freq_norm: np.ndarray  # bin centers, normalized so Nyquist=1
    radial_energy_mean: np.ndarray  # mean across channels (E[|F|^2] in bin)
    radial_energy_per_channel: np.ndarray  # [C, n_bins] cumulative-friendly mean per channel in each bin
    cum_fraction_mean: np.ndarray  # cumulative energy fraction (mean over channels)
    cum_fraction_per_channel: np.ndarray  # [C, n_bins]
    median_radius_mean: float  # radius at which mean cum frac crosses 0.5
    median_radius_per_channel: np.ndarray  # [C]
    r05_per_channel: np.ndarray  # [C] — smallest radius enclosing 5% of energy
    r05_mean: float  # 5th-percentile radius for the channel-mean curve


def _compute_kernel_2d(siren: SIRENKernelND, mask: GaussianModulationND | None, L: int) -> torch.Tensor:
    """Generate the 2D kernel ``K[H, W, C]`` at grid ``(L, L)`` (so spatial size = 2L-1)."""
    with torch.no_grad():
        k, grid = siren(seq_lens=(L, L))  # k: [1, 2L-1, 2L-1, C], grid: [1, 2L-1, 2L-1, 2]
        if mask is not None:
            k = mask(grid=grid, x=k)
    return k.squeeze(0)  # [N, N, C]


def _radial_bin_2d(N: int, n_bins: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(r_norm, bin_idx, bin_centers)`` for an NxN fft-shifted 2D grid.

    ``r_norm`` is each pixel's radial distance from the centre (in [0, sqrt(2)])
    where 1.0 corresponds to the Nyquist frequency along an axis.
    ``bin_idx[i, j]`` is the bin index for pixel ``(i, j)``.
    ``bin_centers`` is shape ``(n_bins,)`` in normalized units.
    """
    cy, cx = (N - 1) / 2.0, (N - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(N).float(), torch.arange(N).float(), indexing="ij")
    # half-extent in each axis is N/2 samples → frequency Nyquist is at radius N/2 (per axis).
    r = torch.sqrt(((yy - cy) / (N / 2.0)) ** 2 + ((xx - cx) / (N / 2.0)) ** 2)
    # Build bin edges in [0, 1] (we discard the corner area > 1 to keep "fraction of Nyquist")
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    # bucketize returns 1..n_bins for in-range; map to 0..n_bins-1 and tag out-of-range as -1
    idx = torch.bucketize(r, edges) - 1
    idx = torch.where((idx < 0) | (idx >= n_bins), torch.full_like(idx, -1), idx)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return r, idx, centers


def compute_spectrum_stats(kernel_hwc: torch.Tensor, n_bins: int = 24) -> SpectrumStats:
    """Compute per-channel and mean radial energy / cumulative fractions."""
    N = kernel_hwc.shape[0]
    assert kernel_hwc.shape[0] == kernel_hwc.shape[1], "Expecting square 2D kernel"
    C = kernel_hwc.shape[-1]

    # 2D FFT per channel; energy = |F|^2
    spec = torch.fft.fft2(kernel_hwc.float(), dim=(0, 1))
    spec = torch.fft.fftshift(spec, dim=(0, 1))
    energy = spec.real**2 + spec.imag**2  # [N, N, C]

    _, idx, centers = _radial_bin_2d(N, n_bins)
    # Per-channel radial mean
    flat_idx = idx.reshape(-1)
    flat_energy = energy.reshape(-1, C)  # [N*N, C]
    valid = flat_idx >= 0
    flat_idx_v = flat_idx[valid]
    flat_energy_v = flat_energy[valid]
    # scatter-mean per bin per channel
    sum_per_bin = torch.zeros(n_bins, C)
    cnt_per_bin = torch.zeros(n_bins, 1)
    sum_per_bin.index_add_(0, flat_idx_v, flat_energy_v)
    cnt_per_bin.index_add_(0, flat_idx_v, torch.ones_like(flat_idx_v, dtype=torch.float32).unsqueeze(-1))
    cnt_per_bin = cnt_per_bin.clamp_min(1.0)
    mean_per_bin = sum_per_bin / cnt_per_bin  # [n_bins, C] = E[|F|^2] in each radial annulus
    # Normalize so each channel sums to 1 (fraction-of-energy distribution).
    # Using mean (not sum) gives equal weight to each radial annulus regardless
    # of its area; we instead want the *total* energy contribution per annulus
    # for cumulative fraction, so use the area-weighted SUM:
    # cumulative = cumsum( sum_per_bin )
    cum_sum = sum_per_bin.cumsum(dim=0)  # [n_bins, C]
    cum_total = cum_sum[-1].clamp_min(1e-30)  # [C]
    cum_frac = cum_sum / cum_total  # [n_bins, C]

    centers_np = centers.numpy()
    cum_frac_np = cum_frac.numpy().T  # [C, n_bins]

    def _quantile_radius(cum_row: np.ndarray, q: float) -> float:
        """Smallest radius (in fraction of Nyquist) at which cum_row crosses q."""
        ge = np.where(cum_row >= q)[0]
        if len(ge) == 0:
            return 1.0
        i0 = ge[0]
        if i0 == 0 or cum_row[i0] == cum_row[i0 - 1]:
            return float(centers_np[i0])
        f0, f1 = cum_row[i0 - 1], cum_row[i0]
        w = (q - f0) / (f1 - f0)
        return float(centers_np[i0 - 1] + w * (centers_np[i0] - centers_np[i0 - 1]))

    # Per-channel quantile radii (5%, 50%)
    median_r = np.zeros(C, dtype=np.float64)
    r05 = np.zeros(C, dtype=np.float64)
    for c in range(C):
        median_r[c] = _quantile_radius(cum_frac_np[c], 0.5)
        r05[c] = _quantile_radius(cum_frac_np[c], 0.05)

    # Mean across channels
    mean_radial_energy = mean_per_bin.mean(dim=-1).numpy()
    mean_cum = cum_frac.mean(dim=-1).numpy()
    median_r_mean = _quantile_radius(mean_cum, 0.5)
    r05_mean = _quantile_radius(mean_cum, 0.05)

    return SpectrumStats(
        L=(N + 1) // 2,
        N=N,
        radial_freq_norm=centers_np,
        radial_energy_mean=mean_radial_energy,
        radial_energy_per_channel=mean_per_bin.numpy().T,  # [C, n_bins]
        cum_fraction_mean=mean_cum,
        cum_fraction_per_channel=cum_frac_np,
        median_radius_mean=median_r_mean,
        median_radius_per_channel=median_r,
        r05_per_channel=r05,
        r05_mean=r05_mean,
    )


# ─── Plotting ────────────────────────────────────────────────────────────────


def plot_radial_spectrum(stats_by_label: dict[str, SpectrumStats], title: str, out_path: Path) -> None:
    """Plot mean radial energy + cumulative fraction for each label."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax0 = axes[0]
    for lbl, s in stats_by_label.items():
        ax0.semilogy(s.radial_freq_norm, s.radial_energy_mean + 1e-30, label=lbl, linewidth=1.8)
    ax0.set_xlabel("Radial frequency (fraction of Nyquist)")
    ax0.set_ylabel("Mean energy per pixel  (log)")
    ax0.set_title("Per-frequency energy density (channel-mean)")
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=9)
    ax0.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax1 = axes[1]
    for lbl, s in stats_by_label.items():
        ax1.plot(
            s.radial_freq_norm,
            s.cum_fraction_mean,
            label=f"{lbl}  (median r={s.median_radius_mean:.3f})",
            linewidth=1.8,
        )
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel("Radial frequency (fraction of Nyquist)")
    ax1.set_ylabel("Cumulative fraction of energy")
    ax1.set_title("Cumulative spectral energy")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_median_radius_histogram(stats_by_label: dict[str, SpectrumStats], title: str, out_path: Path) -> None:
    """Histogram of per-channel median radial frequency (digital, fraction of Nyquist)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    bins = np.linspace(0.0, 1.0, 31)
    for lbl, s in stats_by_label.items():
        ax.hist(
            s.median_radius_per_channel,
            bins=bins,
            alpha=0.45,
            label=f"{lbl}  (mean={s.median_radius_per_channel.mean():.3f})",
            density=True,
        )
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="half-Nyquist")
    ax.set_xlabel("Per-channel median radial frequency (fraction of Nyquist)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─── Studies ─────────────────────────────────────────────────────────────────


def study_resolution_invariance(
    *,
    omega_0: float,
    L_base: int,
    multipliers: tuple[int, ...],
    use_mask: bool,
    seed: int,
    out_dir: Path,
    tag: str,
) -> dict[str, SpectrumStats]:
    """Generate a single SIREN at ``L_base``, then upsample (same weights) to
    higher resolutions; compute and plot spectrum at each.

    Returns a dict of label → SpectrumStats.
    """
    torch.manual_seed(seed)
    ref = build_siren(omega_0=omega_0, L_cache=L_base, seed=seed)
    mask_ref = build_mask(L_base) if use_mask else None
    stats_by_label: dict[str, SpectrumStats] = {}

    for m in multipliers:
        L = L_base if m == 1 else (m * L_base - (m - 1))  # so 2L-1 grows by exactly factor m
        # 2L-1 should be m * (2 L_base - 1).  Solve: 2L - 1 = m(2 L_base - 1) → L = (m(2 L_base - 1) + 1) / 2.
        # When m is odd this is integer; for m=2 we get L = (2*(2 L_base - 1) + 1) / 2 = 2 L_base - 0.5 (fractional).
        # Use a slightly different rule: target N = m * (2 L_base - 1), then L = (N + 1) // 2 (rounded up).
        N_target = m * (2 * L_base - 1)
        L = (N_target + 1) // 2
        # NOTE: when N_target is even, the actual N = 2L-1 = N_target - 1 (off by 1 vs 2× exactly).
        # That's fine for an analysis; spectral comparison normalizes by N.

        upsampled = upsample_siren(ref, L_target=L)
        if use_mask:
            # Build a mask at the new L so its grid_size is auto-injected to 2L-1; we'll *copy*
            # the std_param from the base mask so the same continuous Gaussian is sampled.
            mask = build_mask(L_cache=L)
            with torch.no_grad():
                mask.std_param.copy_(mask_ref.std_param)
        else:
            mask = None

        kernel = _compute_kernel_2d(upsampled, mask, L)
        s = compute_spectrum_stats(kernel)
        stats_by_label[f"N={s.N}  ({m}×)"] = s

    title_mask = "with Gaussian mask" if use_mask else "no mask"
    plot_radial_spectrum(
        stats_by_label,
        f"Resolution sweep ({title_mask}, omega_0={omega_0})",
        out_dir / f"resolution_sweep_{tag}_radial.png",
    )
    plot_median_radius_histogram(
        stats_by_label,
        f"Per-channel median freq ({title_mask}, omega_0={omega_0})",
        out_dir / f"resolution_sweep_{tag}_hist.png",
    )
    return stats_by_label


def study_with_vs_without_mask(
    *,
    omega_0: float,
    L_base: int,
    seed: int,
    out_dir: Path,
) -> None:
    """Compare spectrum with vs without Gaussian mask at the base resolution only."""
    torch.manual_seed(seed)
    ref = build_siren(omega_0=omega_0, L_cache=L_base, seed=seed)
    k_no = _compute_kernel_2d(ref, None, L_base)
    mask = build_mask(L_base)
    k_yes = _compute_kernel_2d(ref, mask, L_base)
    stats = {
        "no mask": compute_spectrum_stats(k_no),
        "Gaussian mask": compute_spectrum_stats(k_yes),
    }
    plot_radial_spectrum(
        stats,
        f"Effect of Gaussian mask  (N={stats['no mask'].N}, omega_0={omega_0})",
        out_dir / "mask_effect_radial.png",
    )
    plot_median_radius_histogram(
        stats, f"Per-channel median freq (N={stats['no mask'].N}, omega_0={omega_0})", out_dir / "mask_effect_hist.png"
    )


def find_omega0_for_target_median(
    *,
    target_median: float,
    L_base: int,
    use_mask: bool,
    seed: int,
    omega0_lo: float = 0.5,
    omega0_hi: float = 80.0,
    n_iters: int = 25,
) -> tuple[float, SpectrumStats]:
    """Bisection search for omega_0 such that the *channel-mean* median radial
    frequency at ``L_base`` equals ``target_median`` (fraction of Nyquist).

    Median radial frequency is monotonically increasing in omega_0, so bisection
    converges quickly.
    """

    def median_at(o0: float) -> tuple[float, SpectrumStats]:
        torch.manual_seed(seed)
        siren = build_siren(omega_0=o0, L_cache=L_base, seed=seed)
        mask = build_mask(L_base) if use_mask else None
        k = _compute_kernel_2d(siren, mask, L_base)
        s = compute_spectrum_stats(k)
        return s.median_radius_mean, s

    m_lo, _ = median_at(omega0_lo)
    m_hi, _ = median_at(omega0_hi)
    assert m_lo < target_median < m_hi, (
        f"target_median={target_median} out of bracket: median(omega0={omega0_lo})={m_lo:.3f}, "
        f"median(omega0={omega0_hi})={m_hi:.3f}.  Widen the search range."
    )

    for _ in range(n_iters):
        mid = math.sqrt(omega0_lo * omega0_hi)  # geometric bisection (omega_0 spans orders of magnitude)
        m_mid, s_mid = median_at(mid)
        if m_mid < target_median:
            omega0_lo = mid
        else:
            omega0_hi = mid
    o0_star = math.sqrt(omega0_lo * omega0_hi)
    m_star, s_star = median_at(o0_star)
    return o0_star, s_star


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--omega-0",
        type=float,
        default=PR_KERNEL_OMEGA_0,
        help="omega_0 used for the initial 'PR-config' study (default: 10.0).",
    )
    parser.add_argument(
        "--L-base",
        type=int,
        default=PR_BASE_L,
        help="Base L_cache (default 15 → 29x29 kernel grid; matches PR config).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--multipliers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Resolution multipliers to study (relative to base N=2L-1).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        summary_lines.append(msg)

    def _log_table(stats_by_label: dict[str, SpectrumStats], log_fn) -> None:
        """Print a summary table including median r, r_05, and low-end coverage."""
        log_fn(
            f"  {'label':<20} {'N':>5} {'median r/Nyq':>14}  {'r_05 (mean)':>12}  "
            f"{'%ch r_05<0.05':>14}  {'energy<0.5':>11}"
        )
        for lbl, s in stats_by_label.items():
            idx_05 = int(np.searchsorted(s.radial_freq_norm, 0.5))
            e05 = float(s.cum_fraction_mean[max(idx_05 - 1, 0)])
            pct_low = 100.0 * float(np.mean(s.r05_per_channel < 0.05))
            log_fn(
                f"  {lbl:<20} {s.N:>5d} {s.median_radius_mean:>14.4f}  "
                f"{s.r05_per_channel.mean():>12.4f}  {pct_low:>13.1f}%  {e05:>11.3f}"
            )

    log("=" * 78)
    log("CKConv kernel spectrum study  (vit5_hybrid PR config)")
    log("=" * 78)
    log(f"L_base                = {args.L_base}   (kernel grid N = {2 * args.L_base - 1})")
    log(f"omega_0               = {args.omega_0}")
    log(f"hidden_omega_0        = {PR_KERNEL_HIDDEN_OMEGA_0}")
    log(f"hidden_dim (channels) = {PR_HIDDEN_DIM}")
    log(f"data_dim              = {PR_DATA_DIM}")
    log(f"SIREN MLP             = {PR_KERNEL_NUM_LAYERS} layers × {PR_KERNEL_MLP_HIDDEN_DIM}")
    log(
        f"Mask                  = GaussianModulationND(min_att@step={PR_MASK_MIN_ATTENUATION_AT_STEP}, "
        f"max_att@limit={PR_MASK_MAX_ATTENUATION_AT_LIMIT}, init_extent={PR_MASK_INIT_EXTENT})"
    )
    log("")

    # ── Study 1: effect of the Gaussian mask at base resolution ───────────────
    log("[Study 1] Effect of the Gaussian mask at base resolution")
    log("-" * 78)
    study_with_vs_without_mask(omega_0=args.omega_0, L_base=args.L_base, seed=args.seed, out_dir=args.output_dir)
    log(f"  → wrote mask_effect_radial.png, mask_effect_hist.png in {args.output_dir}")
    log("")

    # ── Study 2: resolution invariance with PR's omega_0 (no mask) ────────────
    log("[Study 2a] Resolution sweep (no mask, omega_0 fixed)")
    log("-" * 78)
    s_no = study_resolution_invariance(
        omega_0=args.omega_0,
        L_base=args.L_base,
        multipliers=tuple(args.multipliers),
        use_mask=False,
        seed=args.seed,
        out_dir=args.output_dir,
        tag="nomask",
    )
    _log_table(s_no, log)

    # ── Study 2b: resolution invariance with mask ─────────────────────────────
    log("[Study 2b] Resolution sweep (with Gaussian mask, omega_0 fixed)")
    log("-" * 78)
    s_with = study_resolution_invariance(
        omega_0=args.omega_0,
        L_base=args.L_base,
        multipliers=tuple(args.multipliers),
        use_mask=True,
        seed=args.seed,
        out_dir=args.output_dir,
        tag="withmask",
    )
    _log_table(s_with, log)

    # ── Study 3: find omega_0* for "half-spectrum coverage" at base ───────────
    log("[Study 3] Find omega_0* such that median spectral radius = 0.5 of Nyquist at base resolution")
    log("-" * 78)
    log("  (using NO mask so the search isolates SIREN frequency content)")
    o0_star, s_star = find_omega0_for_target_median(
        target_median=0.5,
        L_base=args.L_base,
        use_mask=False,
        seed=args.seed,
    )
    log(f"  omega_0*  = {o0_star:.4f}   (vs PR default {PR_KERNEL_OMEGA_0})")
    log(f"  median r  = {s_star.median_radius_mean:.4f}    (target 0.5)")
    log(
        f"  r_05 mean = {s_star.r05_per_channel.mean():.4f}    "
        f"(% channels with r_05 < 0.05: {100.0 * float(np.mean(s_star.r05_per_channel < 0.05)):.1f}%)"
    )
    log("")

    # ── Study 4: the same omega_0* at higher resolution ───────────────────────
    log("[Study 4] Resolution sweep at omega_0* (no mask) — same SIREN, denser grid")
    log("-" * 78)
    s_star_sweep = study_resolution_invariance(
        omega_0=o0_star,
        L_base=args.L_base,
        multipliers=tuple(args.multipliers),
        use_mask=False,
        seed=args.seed,
        out_dir=args.output_dir,
        tag=f"omega_star_{o0_star:.2f}",
    )
    log(f"  {'label':<20} {'N':>5} {'median r/Nyq':>14}  {'predicted':>10}  {'r_05 mean':>10}  {'%ch r_05<0.05':>14}")
    for lbl, s in s_star_sweep.items():
        ratio = s.N / s_star.N
        pred = 0.5 / ratio
        pct_low = 100.0 * float(np.mean(s.r05_per_channel < 0.05))
        log(
            f"  {lbl:<20} {s.N:>5d} {s.median_radius_mean:>14.4f}  {pred:>10.4f}  "
            f"{s.r05_per_channel.mean():>10.4f}  {pct_low:>13.1f}%"
        )
    log("")
    log("  → median r decreases ~1/m as expected: same continuous content, but the new")
    log("    Nyquist is m× higher, so the relative coverage shrinks by m.")
    log("")

    # ── Study 5: rescale omega_0 by m to restore coverage ─────────────────────
    log("[Study 5] omega_0 ← m * omega_0*  →  re-tests coverage at each new resolution")
    log("-" * 78)
    log(
        f"  {'multiplier':<12} {'omega_0':>10}  {'N':>5}  {'median r/Nyq':>14}  {'r_05 mean':>10}  {'%ch r_05<0.05':>14}"
    )
    summary_per_m = {}
    for m in args.multipliers:
        N_target = m * (2 * args.L_base - 1)
        L = (N_target + 1) // 2
        o0 = m * o0_star
        torch.manual_seed(args.seed)
        siren = build_siren(omega_0=o0, L_cache=L, seed=args.seed)
        kernel = _compute_kernel_2d(siren, None, L)
        s = compute_spectrum_stats(kernel)
        summary_per_m[m] = s
        pct_low = 100.0 * float(np.mean(s.r05_per_channel < 0.05))
        log(
            f"  {m:<12} {o0:>10.4f}  {s.N:>5d}  {s.median_radius_mean:>14.4f}  "
            f"{s.r05_per_channel.mean():>10.4f}  {pct_low:>13.1f}%"
        )
    log("")
    log("  → median r stays ≈ 0.5 across resolutions when omega_0 scales linearly with N.")
    log("    This is the rule:    omega_0(N) = omega_0(N_base) * N / N_base.")
    log("")

    # Comparative plot for Study 5
    plot_radial_spectrum(
        {f"m={m}, ω₀={m * o0_star:.2f}, N={summary_per_m[m].N}": summary_per_m[m] for m in args.multipliers},
        "omega_0 ∝ N preserves spectral coverage  (target median r = 0.5)",
        args.output_dir / "study5_omega0_scaling.png",
    )
    plot_median_radius_histogram(
        {f"m={m}, ω₀={m * o0_star:.2f}, N={summary_per_m[m].N}": summary_per_m[m] for m in args.multipliers},
        "Per-channel median freq under omega_0 ∝ N rule",
        args.output_dir / "study5_omega0_scaling_hist.png",
    )

    # ── Summary file ─────────────────────────────────────────────────────────
    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    log(f"[done] Wrote summary → {summary_path}")


if __name__ == "__main__":
    main()
