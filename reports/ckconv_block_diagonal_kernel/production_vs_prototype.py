"""Verify the upstreamed production classes match the _tmp prototype.

Builds the kernel + mask under the user's chosen defaults
(K=8, linear[1, 12], off_block_scale=0.1, aligned mask) using:

  PROTOTYPE: ``_tmp/spectrum_analysis/multiomega_classes.BlockDiagonalMultiOmegaSIRENKernelND``
             + ``multiband_omega0.build_aligned_mask``
  PRODUCTION: ``nvsubquadratic.modules.kernels_nd.BlockDiagonalMultiOmegaSIRENKernelND``
              + ``nvsubquadratic.modules.masks_nd.BlockAlignedGaussianModulationND``

For each of N seeds we:

  1.  Compare all kernel parameters byte-for-byte (atol=0, rtol=0).
  2.  Compare the mask's ``std_param`` byte-for-byte.
  3.  Compute per-channel spectral medians from the masked kernel and verify
      they match exactly.

Then we pool medians across seeds and plot the production vs prototype
histograms side by side, identical to the format used in
``block_diag_scaling_verify.py``.

Run::

    PYTHONPATH=. python _tmp/spectrum_analysis/production_vs_prototype.py
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
    PR_MASK_INIT_EXTENT,
    PR_MASK_MAX_ATTENUATION_AT_LIMIT,
    PR_MASK_MIN_ATTENUATION_AT_STEP,
    PR_MASK_PARAMETRIZATION,
    compute_spectrum_stats,
)
from multiband_omega0 import build_aligned_mask as proto_build_aligned_mask
from multiomega_classes import (
    BlockDiagonalMultiOmegaSIRENKernelND as ProtoBlockDiag,
)

from nvsubquadratic.modules.kernels_nd import (
    BlockDiagonalMultiOmegaSIRENKernelND as ProdBlockDiag,
)
from nvsubquadratic.modules.masks_nd import (
    BlockAlignedGaussianModulationND as ProdAlignedMask,
)


NUM_BLOCKS = 8
OMEGA0_MIN = 1.0
OMEGA0_MAX = 12.0
OFF_BLOCK_SCALE = 0.1


# ── Builders ────────────────────────────────────────────────────────────────


def build_proto_kernel(seed: int, L: int) -> torch.nn.Module:
    omega_per_block = np.linspace(OMEGA0_MIN, OMEGA0_MAX, NUM_BLOCKS)
    torch.manual_seed(seed)
    return ProtoBlockDiag(
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


def build_prod_kernel(seed: int, L: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return ProdBlockDiag(
        out_dim=PR_HIDDEN_DIM,
        data_dim=PR_DATA_DIM,
        mlp_hidden_dim=PR_KERNEL_MLP_HIDDEN_DIM,
        num_layers=PR_KERNEL_NUM_LAYERS,
        embedding_dim=PR_KERNEL_EMBEDDING_DIM,
        L_cache=L,
        use_bias=True,
        num_blocks=NUM_BLOCKS,
        omega_0_min=OMEGA0_MIN,
        omega_0_max=OMEGA0_MAX,
        schedule="linear",
        off_block_scale=OFF_BLOCK_SCALE,
        hidden_omega_0=PR_KERNEL_HIDDEN_OMEGA_0,
    )


def build_proto_mask(L: int) -> torch.nn.Module:
    return proto_build_aligned_mask(
        L_cache=L,
        K=NUM_BLOCKS,
        channels_per_band=[PR_HIDDEN_DIM // NUM_BLOCKS] * NUM_BLOCKS,
    )


def build_prod_mask(L: int) -> torch.nn.Module:
    return ProdAlignedMask(
        data_dim=PR_DATA_DIM,
        num_channels=PR_HIDDEN_DIM,
        grid_size=2 * L - 1,
        min_attenuation_at_step=PR_MASK_MIN_ATTENUATION_AT_STEP,
        max_attenuation_at_limit=PR_MASK_MAX_ATTENUATION_AT_LIMIT,
        init_extent=PR_MASK_INIT_EXTENT,
        parametrization=PR_MASK_PARAMETRIZATION,
    )


# ── Comparison ──────────────────────────────────────────────────────────────


def _compare_state(proto: torch.nn.Module, prod: torch.nn.Module, label: str) -> None:
    """Bit-identical comparison of every parameter and persistent buffer."""
    proto_params = dict(proto.named_parameters())
    prod_params = dict(prod.named_parameters())
    common = set(proto_params) & set(prod_params)
    only_proto = set(proto_params) - set(prod_params)
    only_prod = set(prod_params) - set(proto_params)
    if only_proto or only_prod:
        print(f"  [{label}] WARN: parameter set differs.")
        print(f"      only in prototype: {sorted(only_proto)}")
        print(f"      only in production: {sorted(only_prod)}")

    for name in sorted(common):
        a = proto_params[name].detach().cpu()
        b = prod_params[name].detach().cpu()
        if a.shape != b.shape:
            raise AssertionError(f"[{label}] shape mismatch on {name}: {a.shape} vs {b.shape}")
        # atol=rtol=0 → require exact bit-for-bit equality.
        torch.testing.assert_close(
            a, b, atol=0.0, rtol=0.0, msg=f"[{label}] {name} differs (max |Δ|={(a - b).abs().max().item()})"
        )


def _kernel_2d(siren: torch.nn.Module, mask: torch.nn.Module, L: int) -> torch.Tensor:
    with torch.no_grad():
        k, grid = siren(seq_lens=(L, L))
        if mask is not None:
            k = mask(grid=grid, x=k)
    return k.squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--L", type=int, default=PR_BASE_L)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Comparing PRODUCTION vs PROTOTYPE")
    print(f"  K = {NUM_BLOCKS}  schedule = linear[{OMEGA0_MIN}, {OMEGA0_MAX}]  off_block_scale = {OFF_BLOCK_SCALE}")
    print(f"  L = {args.L}  (kernel grid = {2 * args.L - 1} per axis)")
    print(
        f"  embedding_dim = {PR_KERNEL_EMBEDDING_DIM}  hidden = {PR_KERNEL_MLP_HIDDEN_DIM}  "
        f"num_layers = {PR_KERNEL_NUM_LAYERS}  out_dim = {PR_HIDDEN_DIM}"
    )
    print(f"  n_seeds = {args.n_seeds}\n")

    proto_medians, prod_medians = [], []
    proto_r05, prod_r05 = [], []

    for s in range(args.n_seeds):
        # ── Build both stacks at the same seed.
        proto_k = build_proto_kernel(seed=s, L=args.L)
        prod_k = build_prod_kernel(seed=s, L=args.L)
        proto_m = build_proto_mask(L=args.L)
        prod_m = build_prod_mask(L=args.L)

        # ── Bit-identical parameter comparison.
        _compare_state(proto_k, prod_k, label=f"kernel seed={s}")
        _compare_state(proto_m, prod_m, label=f"mask seed={s}")

        # Schedule fields (ω₀_per_block on kernel) — production stores as float32 buffer,
        # prototype as float64 numpy array.  Compare on float64.
        proto_omega = torch.tensor(proto_k.omega_0_per_block, dtype=torch.float64)
        prod_omega = prod_k.omega_0_per_block.detach().to(torch.float64)
        torch.testing.assert_close(proto_omega, prod_omega, atol=1e-6, rtol=0)

        # ── Spectrum stats on the masked kernel.
        proto_kk = _kernel_2d(proto_k, proto_m, args.L)
        prod_kk = _kernel_2d(prod_k, prod_m, args.L)
        torch.testing.assert_close(proto_kk, prod_kk, atol=0.0, rtol=0.0, msg=f"masked kernel differs at seed={s}")

        proto_st = compute_spectrum_stats(proto_kk)
        prod_st = compute_spectrum_stats(prod_kk)

        # Per-channel median radii must be identical (they're derived from
        # identical kernels via identical code paths).
        np.testing.assert_array_equal(
            proto_st.median_radius_per_channel,
            prod_st.median_radius_per_channel,
        )
        np.testing.assert_array_equal(
            proto_st.r05_per_channel,
            prod_st.r05_per_channel,
        )

        proto_medians.append(proto_st.median_radius_per_channel)
        prod_medians.append(prod_st.median_radius_per_channel)
        proto_r05.append(proto_st.r05_per_channel)
        prod_r05.append(prod_st.r05_per_channel)

        proto_mu = float(proto_st.median_radius_per_channel.mean())
        prod_mu = float(prod_st.median_radius_per_channel.mean())
        proto_sg = float(proto_st.median_radius_per_channel.std())
        prod_sg = float(prod_st.median_radius_per_channel.std())
        print(
            f"  seed={s:>2d}  prod μ={prod_mu:.6f} σ={prod_sg:.6f}  "
            f"proto μ={proto_mu:.6f} σ={proto_sg:.6f}  ✓ identical"
        )

    proto_medians = np.concatenate(proto_medians, axis=0)
    prod_medians = np.concatenate(prod_medians, axis=0)
    proto_r05 = np.concatenate(proto_r05, axis=0)
    prod_r05 = np.concatenate(prod_r05, axis=0)

    print("\n" + "=" * 80)
    print(" Pooled across seeds")
    print("-" * 80)
    print("  median freq / Nyquist:")
    print(f"    prototype : μ={proto_medians.mean():.6f}  σ={proto_medians.std():.6f}")
    print(f"    production: μ={prod_medians.mean():.6f}  σ={prod_medians.std():.6f}")
    print("  r05 / Nyquist:")
    print(f"    prototype : μ={proto_r05.mean():.6f}")
    print(f"    production: μ={prod_r05.mean():.6f}")
    print("=" * 80)

    np.testing.assert_array_equal(proto_medians, prod_medians)
    np.testing.assert_array_equal(proto_r05, prod_r05)
    print("\n  ✓ All per-channel statistics match bit-for-bit across all seeds.")

    # ── Plot: overlay histograms for visual sanity. ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(0.0, 1.0, 41)

    ax = axes[0]
    ax.hist(
        proto_medians,
        bins=bins,
        density=True,
        color="#1f77b4",
        alpha=0.6,
        edgecolor="#1f77b4",
        linewidth=1.0,
        histtype="stepfilled",
        label=f"PROTOTYPE  μ={proto_medians.mean():.4f}  σ={proto_medians.std():.4f}",
    )
    ax.hist(
        prod_medians,
        bins=bins,
        density=True,
        color="#d62728",
        linewidth=1.6,
        histtype="step",
        label=f"PRODUCTION  μ={prod_medians.mean():.4f}  σ={prod_medians.std():.4f}",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Per-channel median radial freq / Nyquist")
    ax.set_ylabel("Density")
    ax.set_title("Median radial freq (pooled across seeds)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(
        proto_r05,
        bins=bins,
        density=True,
        color="#2ca02c",
        alpha=0.6,
        edgecolor="#2ca02c",
        linewidth=1.0,
        histtype="stepfilled",
        label=f"PROTOTYPE  μ={proto_r05.mean():.4f}",
    )
    ax.hist(
        prod_r05,
        bins=bins,
        density=True,
        color="#d62728",
        linewidth=1.6,
        histtype="step",
        label=f"PRODUCTION  μ={prod_r05.mean():.4f}",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Per-channel r₀₅ / Nyquist")
    ax.set_title("Lower-tail (5%) radial freq (pooled across seeds)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Production vs Prototype — block-diag SIREN + aligned mask\n"
        f"K={NUM_BLOCKS}, linear[{OMEGA0_MIN}, {OMEGA0_MAX}], off={OFF_BLOCK_SCALE}, "
        f"L={args.L} (N={2 * args.L - 1}), n_seeds={args.n_seeds}  •  "
        "production curve overlaid in red — must trace prototype exactly",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    out_path = args.output_dir / "production_vs_prototype.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ wrote {out_path}")


if __name__ == "__main__":
    main()
