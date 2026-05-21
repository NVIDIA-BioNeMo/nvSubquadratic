"""omega_0 scaling experiment for SIREN kernels.

Goal: find the multiplier X such that when L_cache doubles, using omega_0 * X
keeps the kernel's frequency content at the same *relative* bandwidth (fraction
of Nyquist).

Protocol
--------
1.  Instantiate a SIRENKernelND at a base L_cache with a low omega_0 chosen so
    that the spectral support occupies ~half the available bandwidth.
2.  Evaluate the kernel and measure its "spectral occupancy" (fraction of the
    spectrum that contains, e.g., 95% of the energy).
3.  Double L_cache.  With the *same* omega_0 the spectral support now occupies
    only ~1/4 of the new bandwidth.
4.  Sweep omega_0 multipliers and find the one that restores occupancy to 1/2.
5.  Repeat for 4x, 8x, … to see if the relationship is consistent (expect X≈2
    per doubling, i.e., omega_0 ∝ L_cache).

Usage
-----
    conda run -n nv-subq python reports/siren_omega0_dimensional_scaling/omega0_scaling.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nvsubquadratic.modules.kernels_nd import SIRENKernelND


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def spectral_occupancy(kernel_1d: np.ndarray, energy_threshold: float = 0.95) -> float:
    """Fraction of the spectrum containing `energy_threshold` of total energy.

    Args:
        kernel_1d: 1-D real signal (one output channel of the SIREN kernel).
        energy_threshold: cumulative energy fraction (default 95%).

    Returns:
        Fraction in [0, 1] — how much of the one-sided spectrum is needed.
    """
    spectrum = np.abs(np.fft.rfft(kernel_1d))
    power = spectrum**2
    total = power.sum()
    if total < 1e-30:
        return 0.0
    cumulative = np.cumsum(power) / total
    # Index where we first exceed the threshold
    k = int(np.searchsorted(cumulative, energy_threshold)) + 1
    n_bins = len(power)
    return k / n_bins


def build_kernel(
    omega_0: float,
    hidden_omega_0: float,
    L_cache: int,
    data_dim: int = 1,
    seed: int = 42,
    **siren_kwargs,
) -> SIRENKernelND:
    """Build a SIRENKernelND with deterministic init."""
    torch.manual_seed(seed)
    defaults = dict(
        out_dim=1,
        data_dim=data_dim,
        mlp_hidden_dim=32,
        num_layers=3,
        embedding_dim=32,
        omega_0=omega_0,
        hidden_omega_0=hidden_omega_0,
        L_cache=L_cache,
        use_bias=True,
    )
    defaults.update(siren_kwargs)
    return SIRENKernelND(**defaults)


@torch.no_grad()
def evaluate_kernel(model: SIRENKernelND, L: int, data_dim: int = 1) -> np.ndarray:
    """Evaluate the kernel on a grid of size 2L-1 and return as numpy."""
    seq_lens = (L,) * data_dim
    kernel, _ = model(seq_lens)
    # kernel shape: [1, 2L-1, ..., out_dim] — take channel 0
    k = kernel[0, ..., 0].cpu().numpy()
    return k


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    base_L: int = 32,
    base_omega_0: float = 5.0,
    hidden_omega_0: float = 1.0,
    n_doublings: int = 4,
    energy_threshold: float = 0.95,
    multiplier_range: tuple[float, float] = (0.5, 8.0),
    n_multipliers: int = 200,
    data_dim: int = 1,
    seed: int = 42,
):
    """Run the omega_0 scaling experiment.

    For each doubling of L_cache, sweep omega_0 multipliers and find the one
    that matches the base spectral occupancy.
    """
    device = "cpu"

    # --- Step 1: base kernel ---
    base_model = build_kernel(base_omega_0, hidden_omega_0, base_L, data_dim, seed).to(device)
    base_kernel = evaluate_kernel(base_model, base_L, data_dim)

    if data_dim == 1:
        base_occ = spectral_occupancy(base_kernel, energy_threshold)
    else:
        # For nD, flatten and measure (or measure per-axis — flatten is simpler)
        base_occ = spectral_occupancy(base_kernel.ravel(), energy_threshold)

    print(f"{'=' * 70}")
    print(f"omega_0 scaling experiment  (data_dim={data_dim})")
    print(f"{'=' * 70}")
    print(
        f"Base: L_cache={base_L}, omega_0={base_omega_0:.2f}, "
        f"kernel len={2 * base_L - 1}, spectral occupancy={base_occ:.4f}"
    )
    print(f"Target occupancy: {base_occ:.4f}  (energy threshold={energy_threshold})")
    print()

    multipliers = np.linspace(multiplier_range[0], multiplier_range[1], n_multipliers)
    results = []

    for d in range(1, n_doublings + 1):
        L = base_L * (2**d)
        scale_factor = 2**d
        print(f"--- L_cache = {L}  ({scale_factor}x base) ---")

        # Control: same omega_0
        control_model = build_kernel(base_omega_0, hidden_omega_0, L, data_dim, seed).to(device)
        control_kernel = evaluate_kernel(control_model, L, data_dim)
        signal = control_kernel.ravel() if data_dim > 1 else control_kernel
        control_occ = spectral_occupancy(signal, energy_threshold)
        print(
            f"  Same omega_0={base_omega_0:.2f}: occupancy={control_occ:.4f} "
            f"(ratio vs base: {control_occ / base_occ:.3f})"
        )

        # Sweep multipliers
        best_mult = 1.0
        best_err = float("inf")
        sweep_data = []

        for mult in multipliers:
            w0 = base_omega_0 * mult
            model = build_kernel(w0, hidden_omega_0, L, data_dim, seed).to(device)
            kernel = evaluate_kernel(model, L, data_dim)
            signal = kernel.ravel() if data_dim > 1 else kernel
            occ = spectral_occupancy(signal, energy_threshold)
            err = abs(occ - base_occ)
            sweep_data.append((mult, w0, occ, err))
            if err < best_err:
                best_err = err
                best_mult = mult

        best_w0 = base_omega_0 * best_mult
        best_entry = min(sweep_data, key=lambda x: x[3])
        print(
            f"  Best match: mult={best_mult:.4f}, omega_0={best_w0:.2f}, "
            f"occupancy={best_entry[2]:.4f}, error={best_err:.5f}"
        )

        # Theoretical: if omega_0 ∝ L, multiplier should equal scale_factor
        print(f"  Expected if ω₀∝L: mult={scale_factor:.1f}  |  Expected if ω₀∝√L: mult={math.sqrt(scale_factor):.3f}")
        print()

        results.append(
            {
                "L_cache": L,
                "scale_factor": scale_factor,
                "control_occupancy": float(control_occ),
                "best_multiplier": float(best_mult),
                "best_omega_0": float(best_w0),
                "best_occupancy": float(best_entry[2]),
                "target_occupancy": float(base_occ),
            }
        )

    # --- Summary ---
    print(f"{'=' * 70}")
    print("Summary: best omega_0 multiplier per doubling")
    print(f"{'=' * 70}")
    print(f"{'L_cache':>8} {'scale':>6} {'mult':>8} {'omega_0':>8} {'occupancy':>10} {'target':>10}")
    print(f"{base_L:>8} {'1x':>6} {'1.000':>8} {base_omega_0:>8.2f} {base_occ:>10.4f} {base_occ:>10.4f}")
    for r in results:
        print(
            f"{r['L_cache']:>8} {r['scale_factor']:>5}x {r['best_multiplier']:>8.4f} "
            f"{r['best_omega_0']:>8.2f} {r['best_occupancy']:>10.4f} {r['target_occupancy']:>10.4f}"
        )

    # Log-space regression: mult = scale^alpha => log(mult) = alpha * log(scale)
    if len(results) >= 2:
        log_scales = np.array([math.log2(r["scale_factor"]) for r in results])
        log_mults = np.array([math.log2(r["best_multiplier"]) for r in results])
        alpha = np.polyfit(log_scales, log_mults, 1)[0]
        print(f"\nFitted scaling law: omega_0 ∝ L_cache^{alpha:.3f}")
        print("  (alpha=1.0 means linear, alpha=0.5 means sqrt)")

    # Save results
    out_path = Path(__file__).parent / "omega0_scaling_results.json"
    out = {
        "base_L": base_L,
        "base_omega_0": base_omega_0,
        "hidden_omega_0": hidden_omega_0,
        "energy_threshold": energy_threshold,
        "base_occupancy": float(base_occ),
        "data_dim": data_dim,
        "seed": seed,
        "results": results,
    }
    if len(results) >= 2:
        out["fitted_alpha"] = float(alpha)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    # 1D experiment
    run_experiment(
        base_L=32,
        base_omega_0=5.0,
        hidden_omega_0=1.0,
        n_doublings=4,
        data_dim=1,
    )

    print("\n\n")

    # 2D experiment (smaller range since grids grow quadratically)
    run_experiment(
        base_L=16,
        base_omega_0=5.0,
        hidden_omega_0=1.0,
        n_doublings=3,
        data_dim=2,
    )
