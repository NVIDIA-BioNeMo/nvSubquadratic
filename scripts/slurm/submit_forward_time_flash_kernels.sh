#!/bin/bash
# =============================================================================
# Attention-KERNEL comparison: SDPA vs FlexAttention vs FlashAttention-4 vs
# HyenaND — a ready-to-submit wrapper over submit_forward_time_nd.sh.
#
# This is the SECOND of the two forward-time runs, and deliberately a different
# config from the "reach-to-16M" sweep:
#
#   * reach sweep  (submit_forward_time_nd.sh defaults): hidden 8, head_dim 4 —
#     the largest shared width that keeps the qkv tensor under torch's 2^31 index
#     limit out to 16M tokens. flex/fa4 CANNOT run there (they require head_dim
#     >= 16), and SDPA there is a weak non-flash fallback.
#   * this sweep: hidden 512 / 4 heads => head_dim 128 — the regime FlashAttention
#     (2/3/4), cuDNN SDPA, and FlexAttention are actually OPTIMIZED for (FA3/FA4
#     on Hopper/Blackwell are tuned for head_dim 128). head_dim 16 is only the
#     eligibility floor; 64/128 is where these kernels — and real models — live.
#     At hidden 512 the 32-bit-index wall lands at ~1M tokens (2D), which is far
#     past where attention time-walls, so the comparison is complete.
#
# Runs HyenaND and bidirectional Mamba2 alongside the three attention kernels, so
# the plot keeps the Figure-1 punchline (subquadratic scaling past every attention
# kernel's O(L^2) wall, with Mamba2 as the O(L) SSM reference that walls on its
# causal_conv1d grid limit ~1-2M). The fa4 series needs flash-attn-4 in the image
# (rebuild the sqsh with the version-locked layer); absent or mismatched, it
# records 'unavailable'/'error' and the plotter simply omits it. Mamba2 needs
# mamba-ssm (already in the image); absent, it is likewise omitted.
#
# Usage (defaults to 2D; override DATA_DIM / any submit_forward_time_nd.sh var):
#   scripts/slurm/submit_forward_time_flash_kernels.sh
#   DATA_DIM=1 scripts/slurm/submit_forward_time_flash_kernels.sh
#   # lighter head_dim 64 instead of 128:
#   HIDDEN_DIM=256 scripts/slurm/submit_forward_time_flash_kernels.sh
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIM="${DATA_DIM:-2}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"   # /4 heads => head_dim 128 (flash-optimized)
NUM_HEADS="${NUM_HEADS:-4}"
MAMBA_HEADDIM="${MAMBA_HEADDIM:-64}"   # d_inner = 512*expand(2) = 1024 / 64 => 16 heads
MIXERS="${MIXERS:-hyena attention flex fa4 mamba}"

# Per-dim reach: largest R keeping 3*HIDDEN_DIM*R^DATA_DIM under 2^31 at hidden 512.
case "${DATA_DIM}" in
    1) RESOLUTIONS="${RESOLUTIONS:-4096 16384 65536 262144 1048576}" ;;
    2) RESOLUTIONS="${RESOLUTIONS:-64 128 256 512 1024}" ;;
    3) RESOLUTIONS="${RESOLUTIONS:-16 32 64}" ;;
    *) echo "DATA_DIM must be 1, 2, or 3 (got '${DATA_DIM}')"; exit 1 ;;
esac

echo "[flash-kernels] DATA_DIM=${DATA_DIM} hidden=${HIDDEN_DIM} heads=${NUM_HEADS} " \
     "(head_dim=$((HIDDEN_DIM / NUM_HEADS))) mamba_headdim=${MAMBA_HEADDIM} mixers='${MIXERS}'"
echo "[flash-kernels] resolutions='${RESOLUTIONS}'"

# Distinct output stem so results/plots don't clobber the reach sweep's
# forward_time_${DATA_DIM}d.* files.
OUT="forward_time_flash_${DATA_DIM}d"

# Hand off to the main sbatch script. Export the knobs into the environment and
# propagate the whole env with --export=ALL, rather than packing space-containing
# values (MIXERS, RESOLUTIONS) into an --export=KEY=VAL,... string — Slurm splits
# that list on commas and mishandles embedded spaces on some versions.
export DATA_DIM MIXERS HIDDEN_DIM NUM_HEADS MAMBA_HEADDIM RESOLUTIONS OUT
exec sbatch \
    --job-name="nvsubq-fwd-flash-${DATA_DIM}d" \
    --export=ALL \
    "${HERE}/submit_forward_time_nd.sh"
