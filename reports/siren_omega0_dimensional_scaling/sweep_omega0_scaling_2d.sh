#!/bin/bash
# omega_0 scaling sweep — 2D Hyena Gaussian mask
#
# Linear scaling: omega_0 = 10.0 * (L_cache / 64)
# where L_cache = canvas_size / patch_size = 64 / P
#
# | Variant   | L_cache | omega_0 (scaled) | omega_0 (original) |
# |-----------|---------|------------------|--------------------|
# | No patch  | 64      | 10.0             | 10.0               |
# | p=2       | 32      | 5.0              | 10.0               |
# | p=4       | 16      | 2.5              | 10.0               |
# | p=8       | 8       | 1.25             | 10.0               |
# | p=16      | 4       | 0.625            | 10.0               |
#
# All runs use Gaussian mask configs. 50k iters, 1 GPU.
# The no-patch case has omega_0=10 which matches the original, so we
# skip it (already have that result).
#
# Usage:
#   bash reports/siren_omega0_dimensional_scaling/sweep_omega0_scaling_2d.sh

set -euo pipefail
cd /home/dwromero/projects/nvSubquadratic-private

SUBMIT="scripts/slurm/submit_1gpu.sh"
W0_OVERRIDE="net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0"

# ── simple_copy_2d ────────────────────────────────────────────────────

# No patch (L_cache=64, omega_0=10.0) — same as existing, skip

# p=2, L_cache=32, omega_0=5.0
sbatch --job-name=sc2p2-hg-w5 "$SUBMIT" \
    examples/spatial_recall_v2/simple_copy_2d/hyena_gaussian_mask_patch.py \
    net.in_proj_cfg.patch_size=2 \
    "${W0_OVERRIDE}=5.0"

# p=4, L_cache=16, omega_0=2.5
sbatch --job-name=sc2p4-hg-w2.5 "$SUBMIT" \
    examples/spatial_recall_v2/simple_copy_2d/hyena_gaussian_mask_patch.py \
    net.in_proj_cfg.patch_size=4 \
    "${W0_OVERRIDE}=2.5"

# p=8, L_cache=8, omega_0=1.25
sbatch --job-name=sc2p8-hg-w1.25 "$SUBMIT" \
    examples/spatial_recall_v2/simple_copy_2d/hyena_gaussian_mask_patch.py \
    net.in_proj_cfg.patch_size=8 \
    "${W0_OVERRIDE}=1.25"

# p=16, L_cache=4, omega_0=0.625
sbatch --job-name=sc2p16-hg-w0.6 "$SUBMIT" \
    examples/spatial_recall_v2/simple_copy_2d/hyena_gaussian_mask_patch.py \
    net.in_proj_cfg.patch_size=16 \
    "${W0_OVERRIDE}=0.625"

# ── color_conditioning_2d ─────────────────────────────────────────────

# No patch — skip (same omega_0=10)

# p=2, omega_0=5.0
sbatch --job-name=cc2p2-hg-w5 "$SUBMIT" \
    examples/spatial_recall_v2/color_conditioning_2d/hyena_gaussian_mask_patch.py \
    net.in_proj_cfg.patch_size=2 \
    "${W0_OVERRIDE}=5.0"

# p=4, omega_0=2.5
sbatch --job-name=cc2p4-hg-w2.5 "$SUBMIT" \
    examples/spatial_recall_v2/color_conditioning_2d/hyena_gaussian_mask_patch.py \
    net.in_proj_cfg.patch_size=4 \
    "${W0_OVERRIDE}=2.5"

# p=8, omega_0=1.25
sbatch --job-name=cc2p8-hg-w1.25 "$SUBMIT" \
    examples/spatial_recall_v2/color_conditioning_2d/hyena_gaussian_mask_patch.py \
    net.in_proj_cfg.patch_size=8 \
    "${W0_OVERRIDE}=1.25"

# p=16, omega_0=0.625
sbatch --job-name=cc2p16-hg-w0.6 "$SUBMIT" \
    examples/spatial_recall_v2/color_conditioning_2d/hyena_gaussian_mask_patch.py \
    net.in_proj_cfg.patch_size=16 \
    "${W0_OVERRIDE}=0.625"

echo ""
echo "Submitted 8 jobs (4 simple_copy + 4 color_cond, all patched + Gaussian mask + scaled omega_0)"
