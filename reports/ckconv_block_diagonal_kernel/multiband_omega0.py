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

"""Multi-band SIREN init: stratify omega_0 per channel group, align with mask.

Motivation: in the current PR, all 384 SIREN output channels are linear
projections of the same 32-dim hidden representation, which itself is built
from a *single* omega_0.  Empirically that gives a per-channel median-radius
spread of std ≈ 0.07 — i.e. all channels look roughly spectrally identical.
The Gaussian mask is then the only thing producing channel-level spectral
diversity.

This script tries an alternative initialization idea:
  - Partition the 384 output channels into K bands (default K=8).
  - For each band k = 0..K-1, build a *separate* mini-SIREN (same per-band
    architecture: embedding_dim=32, mlp_hidden_dim=32, num_layers=3) but with
    its own band-level omega_0_k, logspaced from omega0_min..omega0_max.
  - Each mini-SIREN produces out_dim/K channels; concatenate.
  - Re-order the Gaussian mask's per-channel std so that the *widest* spatial
    std (the most low-pass channel) sits in the lowest-omega_0 band, and the
    *narrowest* sits in the highest-omega_0 band.  Within a band, ordering is
    arbitrary (we keep the mask's logspace order).

Diagnostics are the same per-channel quantities as `analyze_kernel_spectrum.py`:
  - per-channel median radial frequency
  - per-channel r_05  (low-end coverage)
  - per-band mean spectrum  (does each band actually live at its own freq?)

Run::

    PYTHONPATH=. python _tmp/spectrum_analysis/multiband_omega0.py \\
        --K 8 --omega0-min 1.0 --omega0-max 32.0 --output-dir _tmp/spectrum_analysis
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).parent))
from analyze_kernel_spectrum import (
    PR_BASE_L,
    PR_HIDDEN_DIM,
    PR_KERNEL_EMBEDDING_DIM,
    PR_KERNEL_HIDDEN_OMEGA_0,
    PR_KERNEL_MLP_HIDDEN_DIM,
    PR_KERNEL_NUM_LAYERS,
    SpectrumStats,
    build_mask,
    build_siren,
    compute_spectrum_stats,
)

from nvsubquadratic.modules.kernels_nd import SIRENKernelND


class MultiBandSIREN(nn.Module):
    """K parallel mini-SIRENs with logspaced omega_0, concatenated channelwise.

    Total output channels = ``out_dim``.  Each band produces
    ``ceil(out_dim / K)`` channels (last band may be slightly smaller to hit
    the exact ``out_dim`` total).
    """

    def __init__(
        self,
        K: int,
        omega0_min: float,
        omega0_max: float,
        L_cache: int,
        out_dim: int = PR_HIDDEN_DIM,
        data_dim: int = 2,
        mlp_hidden_dim: int = PR_KERNEL_MLP_HIDDEN_DIM,
        num_layers: int = PR_KERNEL_NUM_LAYERS,
        embedding_dim: int = PR_KERNEL_EMBEDDING_DIM,
        hidden_omega_0: float = PR_KERNEL_HIDDEN_OMEGA_0,
    ):
        super().__init__()
        self.K = K
        # Logspaced per-band omega_0 (low → high).
        self.omega0_per_band = np.logspace(math.log10(omega0_min), math.log10(omega0_max), K).tolist()
        self.out_dim = out_dim
        # Compute per-band channel allocation that sums to out_dim.
        base = out_dim // K
        rem = out_dim - base * K
        # Give the first `rem` bands one extra channel each.
        self.channels_per_band = [base + (1 if k < rem else 0) for k in range(K)]
        assert sum(self.channels_per_band) == out_dim

        self.bands = nn.ModuleList()
        for k in range(K):
            band = SIRENKernelND(
                out_dim=self.channels_per_band[k],
                data_dim=data_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                num_layers=num_layers,
                embedding_dim=embedding_dim,
                omega_0=self.omega0_per_band[k],
                L_cache=L_cache,
                use_bias=True,
                hidden_omega_0=hidden_omega_0,
            )
            self.bands.append(band)

    def forward(self, seq_lens, conditioning=None):
        outs = []
        grid = None
        for band in self.bands:
            k_band, g = band(seq_lens, conditioning=conditioning)
            outs.append(k_band)
            grid = g
        # Concatenate along channel (last) dim.
        return torch.cat(outs, dim=-1), grid


def build_aligned_mask(L_cache: int, K: int, channels_per_band: list[int]) -> torch.nn.Module:
    """Build a Gaussian mask whose per-channel std is reordered so that
    *widest-σ* channels are in the lowest-omega_0 band.

    The default ``GaussianModulationND`` initializes std logspaced from
    ``init_std_low`` (narrowest) to ``init_std_high`` (widest).  We just
    *flip* that order so widest-σ comes first — and then within the K bands
    the std is monotonic-decreasing (widest→narrowest) across the channel axis.
    """
    mask = build_mask(L_cache=L_cache)  # std_param shape: [data_dim, num_channels], increasing
    with torch.no_grad():
        # Flip the channel axis so std goes widest -> narrowest.
        mask.std_param.copy_(mask.std_param.flip(dims=[-1]))
    return mask


def _compute_kernel_2d(siren_or_mb, mask, L: int) -> torch.Tensor:
    with torch.no_grad():
        k, grid = siren_or_mb(seq_lens=(L, L))
        if mask is not None:
            k = mask(grid=grid, x=k)
    return k.squeeze(0)  # [N, N, C]


def _per_band_summary(stats: SpectrumStats, channels_per_band: list[int]) -> list[dict]:
    """Return a list of dicts, one per band, with mean median r and mean r_05."""
    out = []
    start = 0
    for k, n in enumerate(channels_per_band):
        med = float(stats.median_radius_per_channel[start : start + n].mean())
        r05 = float(stats.r05_per_channel[start : start + n].mean())
        out.append({"band": k, "channels": (start, start + n), "median_mean": med, "r05_mean": r05})
        start += n
    return out


def _plot_channel_band_diagnostics(
    stats_baseline: SpectrumStats,
    stats_multiband: SpectrumStats,
    channels_per_band: list[int],
    omega0_per_band: list[float],
    out_path: Path,
    title_suffix: str,
) -> None:
    # Fixed colors — single-ω₀ baseline uses the same blue as the per-row script.
    COLOR_BASE = "#1f77b4"  # blue   (matches multiomega_first_layer.PALETTE["baseline"])
    COLOR_MB = "#8c564b"  # brown  (multi-band marker)
    COLOR_R05 = "#2ca02c"  # green  (r_05 line)
    COLOR_OM = "#d62728"  # red    (ω₀ axis)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1) Per-channel median histogram, baseline vs multiband
    ax = axes[0, 0]
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(
        stats_baseline.median_radius_per_channel,
        bins=bins,
        alpha=0.45,
        color=COLOR_BASE,
        label=f"single-ω₀  (mean={stats_baseline.median_radius_per_channel.mean():.3f}, "
        f"std={stats_baseline.median_radius_per_channel.std():.3f})",
        density=True,
    )
    ax.hist(
        stats_multiband.median_radius_per_channel,
        bins=bins,
        alpha=0.45,
        color=COLOR_MB,
        label=f"multi-band  (mean={stats_multiband.median_radius_per_channel.mean():.3f}, "
        f"std={stats_multiband.median_radius_per_channel.std():.3f})",
        density=True,
    )
    ax.set_xlabel("Per-channel median radial frequency / Nyquist")
    ax.set_ylabel("Density")
    ax.set_title("Per-channel median histogram")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 2) Per-channel r_05 histogram, baseline vs multiband
    ax = axes[0, 1]
    ax.hist(
        stats_baseline.r05_per_channel,
        bins=bins,
        alpha=0.45,
        color=COLOR_BASE,
        label=f"single-ω₀  (mean={stats_baseline.r05_per_channel.mean():.3f})",
        density=True,
    )
    ax.hist(
        stats_multiband.r05_per_channel,
        bins=bins,
        alpha=0.45,
        color=COLOR_MB,
        label=f"multi-band  (mean={stats_multiband.r05_per_channel.mean():.3f})",
        density=True,
    )
    ax.set_xlabel("Per-channel r_05  (5%-energy radius / Nyquist)")
    ax.set_ylabel("Density")
    ax.set_title("Per-channel low-end coverage histogram")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3) Per-band median + r_05 (multiband only) — does each band sit at its own freq?
    ax = axes[1, 0]
    band_idx = np.arange(len(channels_per_band))
    band_meds = []
    band_r05s = []
    band_lows = []
    band_highs = []
    start = 0
    for n in channels_per_band:
        ch_med = stats_multiband.median_radius_per_channel[start : start + n]
        ch_r05 = stats_multiband.r05_per_channel[start : start + n]
        band_meds.append(ch_med.mean())
        band_r05s.append(ch_r05.mean())
        band_lows.append(np.percentile(ch_med, 10))
        band_highs.append(np.percentile(ch_med, 90))
        start += n
    ax.errorbar(
        band_idx,
        band_meds,
        yerr=[np.array(band_meds) - np.array(band_lows), np.array(band_highs) - np.array(band_meds)],
        marker="o",
        capsize=4,
        label="multi-band median r (band mean ± p10–p90)",
        color=COLOR_MB,
    )
    ax.plot(band_idx, band_r05s, "s--", label="multi-band r_05 (band mean)", color=COLOR_R05)
    ax2 = ax.twinx()
    ax2.plot(band_idx, omega0_per_band, "x:", color=COLOR_OM, label="ω₀ per band (right axis)")
    ax2.set_yscale("log")
    ax2.set_ylabel("ω₀ per band  (log)", color=COLOR_OM)
    ax.set_xlabel("Band index (low ω₀ → high ω₀)")
    ax.set_ylabel("Radial frequency / Nyquist")
    ax.set_title("Per-band spectrum location (multi-band)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    # 4) Cumulative energy curves: baseline vs multiband (channel-mean)
    ax = axes[1, 1]
    ax.plot(
        stats_baseline.radial_freq_norm,
        stats_baseline.cum_fraction_mean,
        color=COLOR_BASE,
        label=f"single-ω₀  (median={stats_baseline.median_radius_mean:.3f})",
        linewidth=1.8,
    )
    ax.plot(
        stats_multiband.radial_freq_norm,
        stats_multiband.cum_fraction_mean,
        color=COLOR_MB,
        label=f"multi-band  (median={stats_multiband.median_radius_mean:.3f})",
        linewidth=1.8,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Radial frequency / Nyquist")
    ax.set_ylabel("Cumulative fraction of energy")
    ax.set_title("Cumulative energy (channel-mean)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Multi-band SIREN init  ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--L-base", type=int, default=PR_BASE_L)
    parser.add_argument("--K", type=int, default=8, help="Number of omega_0 bands")
    parser.add_argument("--omega0-min", type=float, default=1.0)
    parser.add_argument("--omega0-max", type=float, default=32.0)
    parser.add_argument(
        "--baseline-omega0",
        type=float,
        default=8.355,
        help="omega_0 used for the single-ω₀ baseline (default: omega_0_star).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-mask", action="store_true", help="Apply the (sorted) Gaussian mask in addition.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"L_base = {args.L_base}   N = {2 * args.L_base - 1}")
    print(f"K bands = {args.K},  omega_0 range = [{args.omega0_min}, {args.omega0_max}] (logspace)")
    print(f"Baseline single-ω₀ = {args.baseline_omega0}")

    # ── Build baseline single-ω₀ SIREN ────────────────────────────────────────
    torch.manual_seed(args.seed)
    siren_baseline = build_siren(omega_0=args.baseline_omega0, L_cache=args.L_base, seed=args.seed)
    mask_baseline = build_mask(L_cache=args.L_base) if args.with_mask else None
    k_baseline = _compute_kernel_2d(siren_baseline, mask_baseline, args.L_base)
    stats_baseline = compute_spectrum_stats(k_baseline)

    # ── Build multi-band SIREN ────────────────────────────────────────────────
    torch.manual_seed(args.seed + 1)
    multiband = MultiBandSIREN(
        K=args.K,
        omega0_min=args.omega0_min,
        omega0_max=args.omega0_max,
        L_cache=args.L_base,
    )
    mask_aligned = (
        build_aligned_mask(L_cache=args.L_base, K=args.K, channels_per_band=multiband.channels_per_band)
        if args.with_mask
        else None
    )
    k_multi = _compute_kernel_2d(multiband, mask_aligned, args.L_base)
    stats_multi = compute_spectrum_stats(k_multi)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n[Channel-level summary]")
    print(f"  {'variant':<22} {'median (mean)':>14} {'median (std)':>13} {'r_05 (mean)':>13} {'%ch r_05<0.05':>14}")
    for name, s in [("single-ω₀ baseline", stats_baseline), ("multi-band (K bands)", stats_multi)]:
        med_mean = s.median_radius_per_channel.mean()
        med_std = s.median_radius_per_channel.std()
        r05_mean = s.r05_per_channel.mean()
        pct_low = 100.0 * float(np.mean(s.r05_per_channel < 0.05))
        print(f"  {name:<22} {med_mean:>14.4f} {med_std:>13.4f} {r05_mean:>13.4f} {pct_low:>13.1f}%")

    # Per-band breakdown
    print(f"\n[Multi-band per-band summary (K={args.K})]")
    print(f"  {'band':<6} {'ω₀':>10} {'channels':>12} {'median r (mean)':>16} {'r_05 (mean)':>13}")
    rows = _per_band_summary(stats_multi, multiband.channels_per_band)
    for r, o0 in zip(rows, multiband.omega0_per_band):
        s, e = r["channels"]
        print(
            f"  {r['band']:<6} {o0:>10.3f} {f'[{s:>4d},{e:>4d})':>12} {r['median_mean']:>16.4f} {r['r05_mean']:>13.4f}"
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    tag = "withmask" if args.with_mask else "nomask"
    out_path = args.output_dir / f"multiband_K{args.K}_{tag}.png"
    _plot_channel_band_diagnostics(
        stats_baseline=stats_baseline,
        stats_multiband=stats_multi,
        channels_per_band=multiband.channels_per_band,
        omega0_per_band=multiband.omega0_per_band,
        out_path=out_path,
        title_suffix=f"K={args.K}, ω₀ ∈ [{args.omega0_min}, {args.omega0_max}], {tag}",
    )
    print(f"\n→ wrote {out_path}")


if __name__ == "__main__":
    main()
