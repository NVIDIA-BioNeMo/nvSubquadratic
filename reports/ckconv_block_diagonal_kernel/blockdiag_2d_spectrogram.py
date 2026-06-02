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

"""2D spectrograms comparing a vanilla SIREN baseline (single ω₀, no block-diag,
default mask) against the linear-schedule block-diag (off=0) kernel for a sweep
of ω₀_max values, with aligned mask, K=8 blocks, on the PR's N=29 grid.

For each variant we compute the kernel, take the 2D FFT (fftshift'd so DC is
centered), square the magnitude (energy), and average across:
  - the channels in each block (per-block mean spectrum) → shape [N, N, K]
  - then across n_seeds for cleaner pictures.

We also compute the all-channels mean spectrum.

Layout:
  row 0   = vanilla SIREN baseline (single ω₀, no block, default mask)
  rows 1+ = linear schedule + block-diag off=0, aligned mask, one ω₀_max per row
  cols    = block 0..K-1 + "all channels mean"

For the baseline row the per-block columns are just arbitrary consecutive
groups of channels (no structural meaning) — they all sample the same
distribution and exist only to fill the same layout for visual comparison.

Run::

    PYTHONPATH=. python _tmp/spectrum_analysis/blockdiag_2d_spectrogram.py \\
        --omega0-max-list 8 10 12 16 20 --n-seeds 5 --baseline-omega0 10 \\
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
    build_mask,
    build_siren,
)
from multiband_omega0 import build_aligned_mask
from multiomega_classes import BlockDiagonalMultiOmegaSIRENKernelND


NUM_BLOCKS = 8
BASELINE_OMEGA0 = 10.0


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


def _kernel_2d(siren, mask, L: int) -> torch.Tensor:
    with torch.no_grad():
        k, grid = siren(seq_lens=(L, L))
        if mask is not None:
            k = mask(grid=grid, x=k)
    return k.squeeze(0)  # [N, N, C]


def _power_spectrum(kernel_hwc: torch.Tensor) -> np.ndarray:
    """Return per-channel power spectrum, fftshift'd so DC is centered.

    Output shape: [N, N, C]  (numpy).
    """
    spec = torch.fft.fft2(kernel_hwc.float(), dim=(0, 1))
    spec = torch.fft.fftshift(spec, dim=(0, 1))
    return (spec.real**2 + spec.imag**2).cpu().numpy()


def _accumulate_per_block(
    siren_factory, mask_factory, *, L: int, K: int, channels_per_block: int, n_seeds: int, omega_per_block: np.ndarray
) -> dict:
    """Average power spectrum per block (and total) across n_seeds.

    Returns dict: {
        "per_block": [N, N, K],
        "total":     [N, N],
        "omega_per_block": [K],
    }
    """
    accum_block = None  # [N, N, K]
    accum_total = None  # [N, N]
    for s in range(n_seeds):
        siren = siren_factory(s)
        mask = mask_factory()
        k = _kernel_2d(siren, mask, L)
        spec = _power_spectrum(k)  # [N, N, C]
        block = np.zeros((spec.shape[0], spec.shape[1], K), dtype=np.float64)
        for b in range(K):
            c0, c1 = b * channels_per_block, (b + 1) * channels_per_block
            block[..., b] = spec[..., c0:c1].mean(axis=-1)
        total = spec.mean(axis=-1)
        if accum_block is None:
            accum_block = block.astype(np.float64)
            accum_total = total.astype(np.float64)
        else:
            accum_block += block
            accum_total += total
    return dict(
        per_block=accum_block / n_seeds,
        total=accum_total / n_seeds,
        omega_per_block=omega_per_block,
    )


def _per_block_and_total_mean_spec(omega_max: float, *, L: int, K: int, channels_per_block: int, n_seeds: int) -> dict:
    """Block-diag (off=0) variant with the linear ω₀ schedule and aligned mask."""
    omega_per_block = np.linspace(1.0, omega_max, K)
    siren_factory = lambda s: _build_block_diag_linear(omega_max, L=L, num_blocks=K, seed=s)
    mask_factory = lambda: build_aligned_mask(L_cache=L, K=K, channels_per_band=[channels_per_block] * K)
    return _accumulate_per_block(
        siren_factory,
        mask_factory,
        L=L,
        K=K,
        channels_per_block=channels_per_block,
        n_seeds=n_seeds,
        omega_per_block=omega_per_block,
    )


def _baseline_per_block_spec(*, L: int, K: int, channels_per_block: int, n_seeds: int, omega_0: float) -> dict:
    """Vanilla SIRENKernelND (single ω₀, no block-diag, default mask).

    The 8 "blocks" here are just consecutive groups of 48 channels with no
    structural meaning — included only to fill the same figure layout for an
    apples-to-apples visual comparison.
    """
    omega_per_block = np.full(K, omega_0, dtype=np.float64)
    siren_factory = lambda s: build_siren(omega_0=omega_0, L_cache=L, seed=s)
    mask_factory = lambda: build_mask(L_cache=L)
    return _accumulate_per_block(
        siren_factory,
        mask_factory,
        L=L,
        K=K,
        channels_per_block=channels_per_block,
        n_seeds=n_seeds,
        omega_per_block=omega_per_block,
    )


def _draw_nyquist_circle(ax, N: int):
    """Overlay a unit-Nyquist circle (radius = N/2 in pixel coords)."""
    theta = np.linspace(0, 2 * np.pi, 256)
    cx, cy = (N - 1) / 2, (N - 1) / 2
    r = N / 2
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), color="white", linewidth=0.8, alpha=0.7)
    # Half-Nyquist for reference
    ax.plot(
        cx + 0.5 * r * np.cos(theta),
        cy + 0.5 * r * np.sin(theta),
        color="white",
        linewidth=0.5,
        alpha=0.4,
        linestyle=":",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--omega0-max-list", type=float, nargs="+", default=[12.0, 16.0, 20.0])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--num-blocks", type=int, default=NUM_BLOCKS)
    parser.add_argument(
        "--baseline-omega0",
        type=float,
        default=BASELINE_OMEGA0,
        help="ω₀ of the vanilla SIREN baseline row (no block-diag, no per-row schedule, default mask).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    L = PR_BASE_L
    K = args.num_blocks
    channels_per_block = PR_HIDDEN_DIM // K
    N = 2 * L - 1
    print(f"L = {L}, N = {N}, K = {K}, channels/block = {channels_per_block}")
    print(f"Computing 2D spectrograms for ω₀_max ∈ {args.omega0_max_list}, n_seeds = {args.n_seeds}")
    print(f"Computing baseline (vanilla SIREN, ω₀ = {args.baseline_omega0}, default mask)")

    baseline = _baseline_per_block_spec(
        L=L,
        K=K,
        channels_per_block=channels_per_block,
        n_seeds=args.n_seeds,
        omega_0=args.baseline_omega0,
    )
    results = {
        om: _per_block_and_total_mean_spec(om, L=L, K=K, channels_per_block=channels_per_block, n_seeds=args.n_seeds)
        for om in args.omega0_max_list
    }

    # Common log-scale color limits across all panels for easy visual comparison.
    all_per_block = [baseline["per_block"]] + [r["per_block"] for r in results.values()]
    all_totals = [baseline["total"]] + [r["total"] for r in results.values()]
    all_block_specs = np.concatenate(all_per_block, axis=-1)
    all_total_specs = np.stack(all_totals, axis=0)
    eps = 1e-12
    log_block = np.log10(all_block_specs + eps)
    log_total = np.log10(all_total_specs + eps)
    # Use the union range
    vmin_block = float(np.percentile(log_block, 5))
    vmax_block = float(log_block.max())
    vmin_total = float(np.percentile(log_total, 5))
    vmax_total = float(log_total.max())

    # ── Plot grid ────────────────────────────────────────────────────────────
    # Top row = vanilla SIREN baseline; subsequent rows = block-diag ω₀_max sweep.
    row_specs: list[tuple[str, str, dict]] = [
        ("baseline", f"baseline\nω₀ = {args.baseline_omega0:g}\n(no block, default mask)", baseline),
    ]
    for om in args.omega0_max_list:
        row_specs.append(("blockdiag", f"ω₀_max = {int(om)}", results[om]))

    nrows = len(row_specs)
    ncols = K + 1  # K blocks + 1 total
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.4 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for i, (kind, ylabel, r) in enumerate(row_specs):
        for b in range(K):
            ax = axes[i, b]
            spec = np.log10(r["per_block"][..., b] + eps)
            im = ax.imshow(spec, cmap="magma", origin="upper", vmin=vmin_block, vmax=vmax_block)
            _draw_nyquist_circle(ax, N)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                # block index titles only meaningful for block-diag rows; show
                # both "block b" and "ch group b" hints once at the top.
                ax.set_title(f"block {b}", fontsize=9)
            if b == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            # Annotate the per-block ω₀ value. For the baseline this is the
            # single shared ω₀ (every group has the same value).
            ax.text(
                0.04,
                0.96,
                f"ω₀={r['omega_per_block'][b]:.1f}",
                transform=ax.transAxes,
                fontsize=7,
                color="white",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, edgecolor="none"),
            )

        # Last column: total mean across all channels
        ax = axes[i, K]
        spec_tot = np.log10(r["total"] + eps)
        im_tot = ax.imshow(spec_tot, cmap="magma", origin="upper", vmin=vmin_total, vmax=vmax_total)
        _draw_nyquist_circle(ax, N)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title("all channels\nmean", fontsize=9)

    # Single colorbar at the right of the per-block panels
    cbar_ax = fig.add_axes([0.04, 0.08, 0.012, 0.84])
    fig.colorbar(im, cax=cbar_ax, label="log₁₀ power")
    cbar_ax.yaxis.set_ticks_position("left")
    cbar_ax.yaxis.set_label_position("left")

    fig.suptitle(
        f"Per-block + total 2D power spectrum  (K={K}, N={N}, mean over {args.n_seeds} seeds)\n"
        f"top row = vanilla SIREN baseline (single ω₀ = {args.baseline_omega0:g}, default mask) — "
        "channel groups are arbitrary\n"
        "remaining rows = linear schedule + block-diag off=0, aligned mask\n"
        "white solid = Nyquist circle  •  white dotted = half-Nyquist  •  DC at center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.96])
    out_path = args.output_dir / "blockdiag_2d_spectrogram.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ wrote {out_path}")


if __name__ == "__main__":
    main()
