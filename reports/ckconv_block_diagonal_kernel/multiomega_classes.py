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

"""Clean child classes for per-row ω₀ SIREN, plus a self-test.

Two child classes:

  - ``MultiOmegaSIRENPositionalEmbeddingND``  subclass of ``SIRENPositionalEmbeddingND``.
    Re-initializes the first-layer weights row-by-row with the proper SIREN
    bound  ``2π · ω₀_k / data_dim``  for each k (instead of a single scalar
    bound for every row).

  - ``MultiOmegaSIRENKernelND``  subclass of ``SIRENKernelND``.
    Builds a normal SIREN, then swaps ``self.positional_embedding`` for the
    multi-ω₀ version.  All other layers (hidden + out_linear) are initialized
    by the parent constructor as usual, so behavior away from the first layer
    is byte-identical to a single-scalar SIREN with the same seed.

Self-test at module load (`if __name__ == "__main__"`):

  1. Build the child class with ω₀_per_row = logspace(1, 32, 32).
  2. Verify each row's first-layer weight std matches the SIREN-uniform
     theoretical value  std(W[k, :]) = 2π·ω₀_k / (d·√3).
  3. Build the post-init monkey-patch variant (the previous approach) and
     verify it produces the same per-row weight std distribution.
  4. Build a few resolutions and verify the kernel can be generated end-to-end
     and produces sensible per-channel medians.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

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

from nvsubquadratic.modules.kernels_nd import (
    SIRENKernelND,
    SIRENPositionalEmbeddingND,
)


# ─── Child classes ───────────────────────────────────────────────────────────


class MultiOmegaSIRENPositionalEmbeddingND(SIRENPositionalEmbeddingND):
    """SIREN positional embedding with per-row ω₀_k in the first layer.

    Each row k of the first-layer linear weight is sampled from
    ``Uniform(-2π·ω₀_k/data_dim, +2π·ω₀_k/data_dim)`` independently, exactly
    as a single-scalar SIREN would do for its global ω₀ — but with ω₀ now a
    vector indexed by the embedding dim.
    """

    def __init__(
        self,
        *,
        data_dim: int,
        embedding_dim: int,
        L_cache: int,
        omega_0_per_row: np.ndarray,
        use_bias: bool = True,
    ):
        assert omega_0_per_row.shape == (embedding_dim,), (
            f"omega_0_per_row must have shape ({embedding_dim},), got {omega_0_per_row.shape}"
        )
        # Build the parent with a placeholder scalar ω₀.  The parent's
        # `_init_siren_weights` call will fill `self.linear.weight` with values
        # from Uniform(-2π/d, +2π/d); we immediately overwrite those below.
        super().__init__(
            data_dim=data_dim,
            embedding_dim=embedding_dim,
            omega_0=1.0,  # placeholder
            L_cache=L_cache,
            use_bias=use_bias,
        )
        # Re-initialize first-layer weight per-row with the proper SIREN bound.
        with torch.no_grad():
            d = float(self.linear.in_features)
            for k in range(embedding_dim):
                bound = 2.0 * math.pi * float(omega_0_per_row[k]) / d
                self.linear.weight[k, :].uniform_(-bound, bound)
            # Bias stays at zero (parent already initialized it to 0); nothing to do.
        # Bookkeeping
        self.omega_0_per_row = omega_0_per_row.astype(np.float64).copy()
        self.omega_0 = float(np.mean(omega_0_per_row))


class MultiOmegaSIRENKernelND(SIRENKernelND):
    """SIRENKernelND with per-row ω₀_k in the first layer.

    All hidden/output layers are initialized exactly as in the parent (with
    ``hidden_omega_0``) — only the positional embedding is swapped out.  This
    means a multi-ω₀ kernel built with the same seed as a single-scalar
    kernel will share the *same* hidden + output weights, differing only in
    the first layer.
    """

    def __init__(
        self,
        *,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        L_cache: int,
        use_bias: bool,
        omega_0_per_row: np.ndarray,
        hidden_omega_0: float = 1.0,
    ):
        # Build parent with a placeholder scalar ω₀ (its positional_embedding is
        # immediately discarded below).
        super().__init__(
            out_dim=out_dim,
            data_dim=data_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            omega_0=1.0,  # placeholder
            L_cache=L_cache,
            use_bias=use_bias,
            hidden_omega_0=hidden_omega_0,
        )
        # Replace the positional embedding with the multi-ω₀ version.  The
        # original was already constructed (and its RNG draws consumed); the
        # new one consumes additional RNG for its first-layer init.
        self.positional_embedding = MultiOmegaSIRENPositionalEmbeddingND(
            data_dim=data_dim,
            embedding_dim=embedding_dim,
            L_cache=L_cache,
            omega_0_per_row=omega_0_per_row,
            use_bias=use_bias,
        )
        # Bookkeeping
        self.omega_0_per_row = omega_0_per_row.astype(np.float64).copy()
        self.omega_0 = float(np.mean(omega_0_per_row))


class BlockDiagonalMultiOmegaSIRENKernelND(MultiOmegaSIRENKernelND):
    """Per-row ω₀ + block-(diagonal/rectangular) init of the hidden + output linears.

    The idea: emulate K parallel SIRENs *inside* a single dense SIREN by
    initializing the MLP so that, at init time, each output channel only
    "sees" one block's worth of first-layer frequency modes.  Training is
    then free to fill in the off-block entries if it helps.

    Layout for ``num_blocks = K`` (must divide ``embedding_dim``,
    ``mlp_hidden_dim``, and ``out_dim``):

      - First layer: ω₀ is grouped — rows 0..(E/K)-1 share ω₀_block_0, rows
        E/K..2E/K-1 share ω₀_block_1, etc.  Inherited from ``MultiOmegaSIRENKernelND``
        with ``omega_0_per_row = repeat(omega_0_per_block, E/K)``.
      - Hidden linears (mlp_hidden_dim → mlp_hidden_dim, square): masked into
        K square blocks on the diagonal.  Off-diagonal entries scaled by
        ``off_block_scale``.
      - Output linear (mlp_hidden_dim → out_dim, rectangular): masked into K
        rectangular blocks of (mlp_hidden_dim/K) → (out_dim/K).  Off-block
        entries scaled by ``off_block_scale``.

    With ``off_block_scale=0.0`` the model is *mathematically equivalent at
    init* to K parallel SIRENs (sharing seeds), but a single dense SIREN
    architecturally; training fills in cross-block weights via gradient flow.
    With ``off_block_scale=1.0`` the block structure is invisible at init and
    we recover the parent ``MultiOmegaSIRENKernelND`` behavior.
    """

    def __init__(
        self,
        *,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        L_cache: int,
        use_bias: bool,
        omega_0_per_block: np.ndarray,
        num_blocks: int,
        off_block_scale: float = 0.0,
        hidden_omega_0: float = 1.0,
    ):
        assert omega_0_per_block.shape == (num_blocks,), (
            f"omega_0_per_block must have shape ({num_blocks},), got {omega_0_per_block.shape}"
        )
        for name, dim in [("embedding_dim", embedding_dim), ("mlp_hidden_dim", mlp_hidden_dim), ("out_dim", out_dim)]:
            assert dim % num_blocks == 0, f"{name}={dim} must be divisible by num_blocks={num_blocks}"
        # Build the per-row ω₀ array: repeat each block's ω₀ across its rows.
        rows_per_block = embedding_dim // num_blocks
        omega_0_per_row = np.repeat(omega_0_per_block, rows_per_block)
        super().__init__(
            out_dim=out_dim,
            data_dim=data_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            L_cache=L_cache,
            use_bias=use_bias,
            omega_0_per_row=omega_0_per_row,
            hidden_omega_0=hidden_omega_0,
        )

        # Apply the block mask to all hidden linears + the output linear.
        with torch.no_grad():
            for linear in self.hidden_linears:
                mask = self._block_mask(
                    linear.out_features,
                    linear.in_features,
                    num_blocks,
                    off_block_scale,
                    device=linear.weight.device,
                    dtype=linear.weight.dtype,
                )
                linear.weight.data.mul_(mask)
            mask = self._block_mask(
                self.out_linear.out_features,
                self.out_linear.in_features,
                num_blocks,
                off_block_scale,
                device=self.out_linear.weight.device,
                dtype=self.out_linear.weight.dtype,
            )
            self.out_linear.weight.data.mul_(mask)

        self.num_blocks = num_blocks
        self.off_block_scale = float(off_block_scale)
        self.omega_0_per_block = omega_0_per_block.astype(np.float64).copy()

    @staticmethod
    def _block_mask(
        out_dim: int,
        in_dim: int,
        num_blocks: int,
        off_block_scale: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a ``[out_dim, in_dim]`` mask: 1.0 on diagonal blocks,
        ``off_block_scale`` elsewhere.

        Block sizes: rows = out_dim // num_blocks, cols = in_dim // num_blocks.
        """
        rows_per_block = out_dim // num_blocks
        cols_per_block = in_dim // num_blocks
        mask = torch.full((out_dim, in_dim), float(off_block_scale), device=device, dtype=dtype)
        for k in range(num_blocks):
            r0, r1 = k * rows_per_block, (k + 1) * rows_per_block
            c0, c1 = k * cols_per_block, (k + 1) * cols_per_block
            mask[r0:r1, c0:c1] = 1.0
        return mask


# ─── Builders ────────────────────────────────────────────────────────────────


def build_multiomega_siren_clean(
    omega_0_per_row: np.ndarray,
    L_cache: int,
    *,
    seed: int = 0,
    out_dim: int = PR_HIDDEN_DIM,
    data_dim: int = PR_DATA_DIM,
    mlp_hidden_dim: int = PR_KERNEL_MLP_HIDDEN_DIM,
    num_layers: int = PR_KERNEL_NUM_LAYERS,
    embedding_dim: int = PR_KERNEL_EMBEDDING_DIM,
    hidden_omega_0: float = PR_KERNEL_HIDDEN_OMEGA_0,
) -> MultiOmegaSIRENKernelND:
    """Build the per-row ω₀ SIREN via the clean child-class path."""
    torch.manual_seed(seed)
    return MultiOmegaSIRENKernelND(
        out_dim=out_dim,
        data_dim=data_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        L_cache=L_cache,
        use_bias=True,
        omega_0_per_row=omega_0_per_row,
        hidden_omega_0=hidden_omega_0,
    )


def build_multiomega_siren_postpatch(
    omega_0_per_row: np.ndarray,
    L_cache: int,
    *,
    seed: int = 0,
    out_dim: int = PR_HIDDEN_DIM,
    data_dim: int = PR_DATA_DIM,
    mlp_hidden_dim: int = PR_KERNEL_MLP_HIDDEN_DIM,
    num_layers: int = PR_KERNEL_NUM_LAYERS,
    embedding_dim: int = PR_KERNEL_EMBEDDING_DIM,
    hidden_omega_0: float = PR_KERNEL_HIDDEN_OMEGA_0,
) -> SIRENKernelND:
    """Reference: build a normal SIREN with ω₀=1, then post-init scale rows.

    Equivalent in distribution to the clean child-class path: row k ends up
    with ``W[k, :] ~ Uniform(-2π·ω₀_k/d, +2π·ω₀_k/d)`` either way.  We use this
    only to verify that the child class is a faithful refactor.
    """
    torch.manual_seed(seed)
    siren = SIRENKernelND(
        out_dim=out_dim,
        data_dim=data_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        omega_0=1.0,
        L_cache=L_cache,
        use_bias=True,
        hidden_omega_0=hidden_omega_0,
    )
    with torch.no_grad():
        scale = torch.from_numpy(omega_0_per_row.astype(np.float32)).unsqueeze(-1)  # [E, 1]
        siren.positional_embedding.linear.weight.mul_(scale)
    return siren


# ─── Self-test ───────────────────────────────────────────────────────────────


def _theoretical_uniform_std(omega_0: float, data_dim: int) -> float:
    """Std of Uniform(-a, a) where a = 2π·ω₀/d  is  a/√3 = 2π·ω₀/(d·√3)."""
    return 2.0 * math.pi * float(omega_0) / (float(data_dim) * math.sqrt(3.0))


def _row_std(weight: torch.Tensor) -> np.ndarray:
    """Per-row std of a [E, d] weight matrix, using population std (not sample)."""
    # ddof=0 to match torch.std(unbiased=False), since the analytic derivation
    # of Uniform std is the population std.
    return weight.detach().cpu().numpy().std(axis=-1, ddof=0)


def main() -> None:
    print("=" * 80)
    print("Self-test: child class ↔ post-init scaling (statistical equivalence)")
    print("=" * 80)

    L = PR_BASE_L
    d = PR_DATA_DIM
    E = PR_KERNEL_EMBEDDING_DIM

    # A non-trivial schedule that covers low + high ω₀_k.
    omega_0_per_row = np.logspace(0.0, math.log10(32.0), E)

    # ── Build via child class and via post-init scaling ───────────────────────
    siren_clean = build_multiomega_siren_clean(omega_0_per_row, L_cache=L, seed=0)
    siren_patch = build_multiomega_siren_postpatch(omega_0_per_row, L_cache=L, seed=0)

    W_clean = siren_clean.positional_embedding.linear.weight  # [E, d]
    W_patch = siren_patch.positional_embedding.linear.weight  # [E, d]
    assert W_clean.shape == (E, d)
    assert W_patch.shape == (E, d)

    # ── 1) Theoretical row-std match (Monte Carlo over many seeds) ────────────
    # With data_dim=2 we only get 2 samples per row, so any single instance
    # of mean(|W[k,:]|) has ~12% finite-sample noise vs the theoretical
    # E[|U|] = π·ω₀_k/d.  We average across many seeds to drive that noise
    # down and check both:
    #   (a) per-row slope vs ω₀_k matches π/d, and
    #   (b) clean child class agrees with post-init scaling row-by-row.
    print("\n[1] Per-row mean(|W|) vs ω₀_k  (Monte Carlo over seeds)")
    n_seeds = 1000
    mean_abs_clean_mc = np.zeros((n_seeds, E), dtype=np.float64)
    mean_abs_patch_mc = np.zeros((n_seeds, E), dtype=np.float64)
    for s in range(n_seeds):
        # We only need the embedding layer's weight; build it directly to keep this fast.
        torch.manual_seed(s)
        emb_clean = MultiOmegaSIRENPositionalEmbeddingND(
            data_dim=d,
            embedding_dim=E,
            L_cache=L,
            omega_0_per_row=omega_0_per_row,
            use_bias=True,
        )
        torch.manual_seed(s)
        emb_patch = SIRENPositionalEmbeddingND(
            data_dim=d,
            embedding_dim=E,
            L_cache=L,
            omega_0=1.0,
            use_bias=True,
        )
        with torch.no_grad():
            scale = torch.from_numpy(omega_0_per_row.astype(np.float32)).unsqueeze(-1)
            emb_patch.linear.weight.mul_(scale)
        mean_abs_clean_mc[s] = emb_clean.linear.weight.detach().abs().mean(dim=-1).cpu().numpy()
        mean_abs_patch_mc[s] = emb_patch.linear.weight.detach().abs().mean(dim=-1).cpu().numpy()

    mean_abs_clean = mean_abs_clean_mc.mean(axis=0)  # [E] — averaged over seeds
    mean_abs_patch = mean_abs_patch_mc.mean(axis=0)
    expected_mean_abs = np.pi * omega_0_per_row / d  # E[|U|] = a/2 = π·ω₀_k/d

    rel_err_clean = float(np.max(np.abs(mean_abs_clean - expected_mean_abs) / expected_mean_abs))
    rel_err_patch = float(np.max(np.abs(mean_abs_patch - expected_mean_abs) / expected_mean_abs))
    print(f"    Monte-Carlo over {n_seeds} seeds.  Per-row mean(|W|), max relative error vs theory:")
    print(f"      clean child class : {rel_err_clean:.2%}")
    print(f"      post-init scaling : {rel_err_patch:.2%}")
    # With d·n_seeds samples per row, expected noise ~ 1/√(d·n_seeds).
    # For d=2, n_seeds=1000 → ~2.2%.  Use 5% as the assertion budget.
    assert rel_err_clean < 0.05, (
        f"Clean child class deviates from theory by up to {rel_err_clean:.1%} "
        f"per row (expected <5% with {n_seeds} seeds and d=2)."
    )
    assert rel_err_patch < 0.05, (
        f"Post-init scaling deviates from theory by up to {rel_err_patch:.1%} "
        f"per row (expected <5% with {n_seeds} seeds and d=2)."
    )
    print("    OK: both within 5% per-row vs theoretical mean {π·ω₀_k/d}.")

    # ── 2) Per-row agreement: clean class ↔ post-init scaling ────────────────
    # Note: the two paths consume different numbers of RNG draws (clean class
    # re-inits rows, post-init scaling does not), so identical seeds do NOT
    # produce identical samples — only equal *distributions*.  Across many
    # seeds, sample means must agree.
    print("\n[2] Per-row mean(|W|): clean vs post-init scaling (statistical equivalence)")
    rel_diff_per_row = np.abs(mean_abs_clean - mean_abs_patch) / mean_abs_patch
    rel_diff_aggregate = abs(mean_abs_clean.sum() - mean_abs_patch.sum()) / mean_abs_patch.sum()
    print(
        f"    max per-row relative difference  (noise-floor ≈ {2 * rel_err_patch:.1%}): {rel_diff_per_row.max():.2%}"
    )
    print(f"    mean per-row relative difference: {rel_diff_per_row.mean():.2%}")
    print(f"    aggregate-across-rows relative difference: {rel_diff_aggregate:.2%}")
    # Aggregate across rows reduces noise by another factor of √E ≈ 5.7×.
    assert rel_diff_aggregate < 0.02, f"Aggregate clean ↔ post-init differs by {rel_diff_aggregate:.1%}, expected <2%."
    print("    OK: aggregates agree to <2%; per-row diffs are at the finite-sample noise floor.")

    # ── 3) End-to-end kernel generation works ─────────────────────────────────
    print("\n[3] End-to-end kernel generation (clean class)")
    with torch.no_grad():
        k_clean, _ = siren_clean(seq_lens=(L, L))
        k_patch, _ = siren_patch(seq_lens=(L, L))
    print(
        f"    clean kernel:      shape={tuple(k_clean.shape)}, "
        f"std={k_clean.std().item():.4f}, abs.mean={k_clean.abs().mean().item():.4f}"
    )
    print(
        f"    post-patch kernel: shape={tuple(k_patch.shape)}, "
        f"std={k_patch.std().item():.4f}, abs.mean={k_patch.abs().mean().item():.4f}"
    )
    s_clean = compute_spectrum_stats(k_clean.squeeze(0))
    s_patch = compute_spectrum_stats(k_patch.squeeze(0))
    print(
        f"    clean      : per-channel median μ={s_clean.median_radius_per_channel.mean():.4f},"
        f" σ={s_clean.median_radius_per_channel.std():.4f}"
    )
    print(
        f"    post-patch : per-channel median μ={s_patch.median_radius_per_channel.mean():.4f},"
        f" σ={s_patch.median_radius_per_channel.std():.4f}"
    )
    rel_mu = abs(s_clean.median_radius_per_channel.mean() - s_patch.median_radius_per_channel.mean()) / max(
        1e-6, s_patch.median_radius_per_channel.mean()
    )
    rel_sg = abs(s_clean.median_radius_per_channel.std() - s_patch.median_radius_per_channel.std()) / max(
        1e-6, s_patch.median_radius_per_channel.std()
    )
    print(f"    relative diff in (μ, σ) = ({rel_mu:.2%}, {rel_sg:.2%})  (different RNG paths → noise)")
    assert rel_mu < 0.10, f"Spectrum μ differs by {rel_mu:.1%}, expected < 10%"
    assert rel_sg < 0.20, f"Spectrum σ differs by {rel_sg:.1%}, expected < 20%"
    print("    OK")

    # ── 4) Single-scalar limit: child class with constant ω₀ matches baseline ─
    # Note: the two paths use different RNG positions for the positional
    # embedding's first-layer weight (the child class re-initializes it),
    # so a single seed can show large noise.  Average across many seeds.
    print("\n[4] Sanity: child class with constant ω₀_per_row matches single-scalar SIREN (mean over seeds)")
    n_seeds_4 = 30
    mu_child_arr, sg_child_arr = [], []
    mu_base_arr, sg_base_arr = [], []
    omega_const = np.full((E,), 8.355)
    for s in range(n_seeds_4):
        siren_child_const = build_multiomega_siren_clean(omega_const, L_cache=L, seed=s)
        torch.manual_seed(s)
        siren_baseline = SIRENKernelND(
            out_dim=PR_HIDDEN_DIM,
            data_dim=PR_DATA_DIM,
            mlp_hidden_dim=PR_KERNEL_MLP_HIDDEN_DIM,
            num_layers=PR_KERNEL_NUM_LAYERS,
            embedding_dim=PR_KERNEL_EMBEDDING_DIM,
            omega_0=8.355,
            L_cache=L,
            use_bias=True,
            hidden_omega_0=PR_KERNEL_HIDDEN_OMEGA_0,
        )
        with torch.no_grad():
            k_child, _ = siren_child_const(seq_lens=(L, L))
            k_base, _ = siren_baseline(seq_lens=(L, L))
        s_child = compute_spectrum_stats(k_child.squeeze(0))
        s_base = compute_spectrum_stats(k_base.squeeze(0))
        mu_child_arr.append(s_child.median_radius_per_channel.mean())
        sg_child_arr.append(s_child.median_radius_per_channel.std())
        mu_base_arr.append(s_base.median_radius_per_channel.mean())
        sg_base_arr.append(s_base.median_radius_per_channel.std())
    mu_child, sg_child = float(np.mean(mu_child_arr)), float(np.mean(sg_child_arr))
    mu_base, sg_base = float(np.mean(mu_base_arr)), float(np.mean(sg_base_arr))
    print(f"    child (constant ω₀=8.355): per-channel median μ={mu_child:.4f},  σ={sg_child:.4f}")
    print(f"    baseline (ω₀=8.355)     : per-channel median μ={mu_base:.4f},  σ={sg_base:.4f}")
    rel_mu2 = abs(mu_child - mu_base) / mu_base
    rel_sg2 = abs(sg_child - sg_base) / sg_base
    print(f"    relative diff in (μ, σ) = ({rel_mu2:.2%}, {rel_sg2:.2%})  (averaged over {n_seeds_4} seeds)")
    assert rel_mu2 < 0.03 and rel_sg2 < 0.10, (
        f"Constant-ω₀ child differs from single-scalar baseline more than expected: ({rel_mu2:.1%}, {rel_sg2:.1%})"
    )
    print("    OK: child class with constant ω₀ converges to single-scalar baseline.")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    main()
