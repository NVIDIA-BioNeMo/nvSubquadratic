#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --partition=36x2-a01r
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=0
#SBATCH --job-name=nvsubq-fwdtime-2d
#SBATCH --mail-type=FAIL
#SBATCH --exclusive

set -x

# =============================================================================
# 2D forward-time-vs-resolution benchmark (the 2D analogue of Figure 1, right).
#
# Times a single HyenaND / Attention / Mamba2 layer at growing square
# resolutions on ONE GPU and writes a JSONL, then renders the paper-style
# log-log plot. Single-GPU microbenchmark (no torchrun / CP): we pin
# CUDA_VISIBLE_DEVICES=0 even on an exclusive multi-GPU node.
#
# Usage (defaults target the GB200 arm64 image + partition below):
#   sbatch scripts/slurm/submit_forward_time_nd.sh
#
#   # H100 instead: use the x86_64 image and a Hopper partition (the defaults
#   # sweep unchanged — the ~15 GB peak at 2048^2 fits an 80 GB H100, and the
#   # ceiling is the 4096^2 torch 32-bit-index wall, not memory):
#   SQSH_PATH=/lustre/.../enroot/nvsubquadratic-x86_64.sqsh \
#       sbatch --partition=<h100_partition> scripts/slurm/submit_forward_time_nd.sh
#
# The defaults sweep 64x64 .. 8192x8192 (4K .. 64M tokens) at hidden_dim=64 with
# the circular (single-grid) kernel, so a *single* run produces the whole story:
# the HyenaND/Attention (+Mamba2 if installed) comparison where they overlap, and
# HyenaND scaling far past attention's O(L^2) wall. Observed reach at hidden_dim=64
# is ~2048^2 (4M tokens, ~1200x faster than attention there); at 4096^2+ every
# operator hits torch's 2^31-element 32-bit-indexing limit and errors (that top-end
# 'x' is a torch limit, not subq_ops or memory). Adaptive timing keeps the slow
# high-R points cheap; MAX_SECONDS marks where attention becomes an 'x'.
#
# All mixers ALWAYS share one hidden_dim (one comparable plot). Peak memory at
# 2048^2 is ~15 GB, so hidden_dim=64 fits an 80 GB H100 and a GB200 alike; the
# ceiling is the 32-bit-index wall above, not memory. Every parameter is an
# environment override (one script, not modes); these change the shared value /
# range for the WHOLE run, they do not mix dims:
#   # all three at hidden 256, non-circular (whole run) — caps ~4096^2:
#   HIDDEN_DIM=256 GRID_TYPE=double RESOLUTIONS="64 128 256 512 1024 2048 4096" \
#       sbatch scripts/slurm/submit_forward_time_nd.sh
#   # HyenaND-reach-to-8K, apple-to-apple: all three at hidden 8 (the largest
#   # shared width whose qkv tensor 3*hidden*R^2 stays under 2^31 at 8192^2), so
#   # HyenaND continues to 64M while Attention (~16M) and Mamba (~4M) x out at the
#   # SAME config. num_heads/mamba_headdim reduced to divide hidden 8:
#   HIDDEN_DIM=8 NUM_HEADS=2 MAMBA_HEADDIM=8 \
#       sbatch scripts/slurm/submit_forward_time_nd.sh
#   # Attention-KERNEL comparison (SDPA vs FlexAttention vs FlashAttention-4 vs
#   # HyenaND). flex/fa4 REQUIRE head_dim >= 16, so this is a SEPARATE run from the
#   # reach story above: bump the width so head_dim = HIDDEN_DIM/NUM_HEADS >= 16
#   # (256/8 = 32). Caps ~2048^2 (4M) at the 32-bit-index wall — plenty, since every
#   # attention kernel walls on time/memory well before that. Needs flash-attn-4 in
#   # the image for the fa4 series (absent -> it shows 'unavailable' and is omitted):
#   MIXERS="hyena attention flex fa4" HIDDEN_DIM=256 NUM_HEADS=8 GRID_TYPE=double \
#       RESOLUTIONS="64 128 256 512 1024 2048" \
#       sbatch scripts/slurm/submit_forward_time_nd.sh
#   # HyenaND only:
#   MIXERS=hyena sbatch scripts/slurm/submit_forward_time_nd.sh
#
# Prerequisites:
#   * The repo (with these benchmark scripts) is on the cluster at CODE_PATH; it
#     is mounted and run with PYTHONPATH=. so the latest code is always used.
#   * The image (SQSH_PATH) has the subq_ops v2 CUDA kernels ([cuda] extra).
#     Optional: mamba-ssm for the Mamba2 series — absent, it shows as 'x' and
#     the other two still render.
# =============================================================================

# ── Paths (adjust to your cluster layout) ────────────────────────────────────
SQSH_PATH="${SQSH_PATH:-/lustre/fsw/healthcareeng_bionemo/farhadr/enroot/nvsubquadratic-arm64.sqsh}"
CODE_PATH="${CODE_PATH:-/lustre/fsw/healthcareeng_bionemo/farhadr/nvsubquadratic_workdir/nvSubquadratic-private}"
CODE_MOUNT=/workspace/nvsubq
RESULTS_HOST="${CODE_PATH}/benchmarks/results"
MOUNTS="${CODE_PATH}:${CODE_MOUNT}"

mkdir -p "${RESULTS_HOST}"

# ── Benchmark parameters (override any from the environment) ──────────────────
DATA_DIM="${DATA_DIM:-2}"          # 1 | 2 | 3   (L = R^DATA_DIM)
MIXERS="${MIXERS:-attention hyena mamba}"
# Shared width. hidden 8 (num_heads 2, mamba_headdim 8) keeps the qkv tensor
# 3*hidden*R^N under torch's 2^31 index limit through ~16M tokens in every dim.
HIDDEN_DIM="${HIDDEN_DIM:-8}"
NUM_HEADS="${NUM_HEADS:-2}"
MAMBA_HEADDIM="${MAMBA_HEADDIM:-8}"
GRID_TYPE="${GRID_TYPE:-single}"
# Per-dim resolution sweep (R), each topping out near 16M tokens. Override with RESOLUTIONS=.
#   1D: L = R      2D: L = R^2      3D: L = R^3
case "${DATA_DIM}" in
    1) RESOLUTIONS="${RESOLUTIONS:-4096 16384 65536 262144 1048576 4194304 16777216}" ;;
    2) RESOLUTIONS="${RESOLUTIONS:-64 128 256 512 1024 2048 4096}" ;;
    3) RESOLUTIONS="${RESOLUTIONS:-16 32 64 128 256}" ;;
    *) echo "DATA_DIM must be 1, 2, or 3 (got '${DATA_DIM}')"; exit 1 ;;
esac
FFT_BACKEND="${FFT_BACKEND:-subq_ops}"   # 1D=causal fused, 2D=fused; 3D auto-falls back to torch_fft
DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WARMUP="${NUM_WARMUP:-10}"
NUM_ITERS="${NUM_ITERS:-30}"
MAX_SECONDS="${MAX_SECONDS:-300}"
OUT="${OUT:-forward_time_${DATA_DIM}d}"

JSONL="${CODE_MOUNT}/benchmarks/results/${OUT}.jsonl"
PNG="${CODE_MOUNT}/benchmarks/results/${OUT}.png"
MEM_PNG="${CODE_MOUNT}/benchmarks/results/${OUT}_memory.png"

# ── In-container command ─────────────────────────────────────────────────────
read -r -d '' COMMAND <<EOF
source /opt/conda/etc/profile.d/conda.sh && conda activate base
export CUDA_VISIBLE_DEVICES=0
cd ${CODE_MOUNT}

echo "======================================================"
echo "${DATA_DIM}D forward-time benchmark  (L = R^${DATA_DIM})"
echo "mixers='${MIXERS}' hidden_dim=${HIDDEN_DIM} grid=${GRID_TYPE} backend=${FFT_BACKEND}"
echo "resolutions='${RESOLUTIONS}' dtype=${DTYPE} batch=${BATCH_SIZE} max_s=${MAX_SECONDS}"
echo "======================================================"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
python -c "import subquadratic_ops_torch; print('subq_ops: available')" \
    || echo "WARNING: subquadratic_ops_torch missing — hyena subq_ops points will error."

PYTHONPATH=. python benchmarks/benchmark_forward_time_nd_resolution.py \
    --data-dim ${DATA_DIM} \
    --fft-backend ${FFT_BACKEND} \
    --grid-type ${GRID_TYPE} \
    --dtype ${DTYPE} \
    --batch-size ${BATCH_SIZE} \
    --hidden-dim ${HIDDEN_DIM} \
    --num-heads ${NUM_HEADS} \
    --mamba-headdim ${MAMBA_HEADDIM} \
    --mixers ${MIXERS} \
    --resolutions ${RESOLUTIONS} \
    --num-warmup ${NUM_WARMUP} \
    --num-iters ${NUM_ITERS} \
    --max-seconds-per-point ${MAX_SECONDS} \
    --output ${JSONL}

# Render the time and memory plots in-job (matplotlib ships in the dev image).
if python -c "import matplotlib" 2>/dev/null; then
    PYTHONPATH=. python scripts/visualization/visualize_forward_time_nd.py \
        --input ${JSONL} --out ${PNG} --no-fail-markers
    PYTHONPATH=. python scripts/visualization/visualize_forward_time_nd.py \
        --input ${JSONL} --out ${MEM_PNG} --metric memory --no-fail-markers
else
    echo "[plot] matplotlib unavailable — plot on the login node from ${OUT}.jsonl."
fi
EOF

srun \
    --output "${RESULTS_HOST}/${OUT}-%j.out" \
    --error  "${RESULTS_HOST}/${OUT}-error-%j.out" \
    --container-image="${SQSH_PATH}" \
    --container-mounts "${MOUNTS}" \
    bash -c "${COMMAND}"

set +x
