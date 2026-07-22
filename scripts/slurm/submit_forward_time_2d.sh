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
#   sbatch scripts/slurm/submit_forward_time_2d.sh
#
#   # H100 instead: use the x86_64 image and a Hopper partition (the defaults
#   # sweep unchanged — the ~15 GB peak at 2048^2 fits an 80 GB H100, and the
#   # ceiling is the 4096^2 torch 32-bit-index wall, not memory):
#   SQSH_PATH=/lustre/.../enroot/nvsubquadratic-x86_64.sqsh \
#       sbatch --partition=<h100_partition> scripts/slurm/submit_forward_time_2d.sh
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
#       sbatch scripts/slurm/submit_forward_time_2d.sh
#   # HyenaND-reach-to-8K, apple-to-apple: all three at hidden 8 (the largest
#   # shared width whose qkv tensor 3*hidden*R^2 stays under 2^31 at 8192^2), so
#   # HyenaND continues to 64M while Attention (~16M) and Mamba (~4M) x out at the
#   # SAME config. num_heads/mamba_headdim reduced to divide hidden 8:
#   HIDDEN_DIM=8 NUM_HEADS=2 MAMBA_HEADDIM=8 \
#       sbatch scripts/slurm/submit_forward_time_2d.sh
#   # HyenaND only:
#   MIXERS=hyena sbatch scripts/slurm/submit_forward_time_2d.sh
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
MIXERS="${MIXERS:-attention hyena mamba}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
# Attention/Mamba shape params — must divide hidden_dim. Reduce these with
# hidden_dim: at hidden 8 use NUM_HEADS=2 (head_dim 4, valid for 2D RoPE) and
# MAMBA_HEADDIM=8 (d_inner=hidden*2=16 must be divisible by it).
NUM_HEADS="${NUM_HEADS:-8}"
MAMBA_HEADDIM="${MAMBA_HEADDIM:-64}"
GRID_TYPE="${GRID_TYPE:-single}"
RESOLUTIONS="${RESOLUTIONS:-64 128 256 512 1024 2048 4096 8192}"
FFT_BACKEND="${FFT_BACKEND:-subq_ops}"
DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WARMUP="${NUM_WARMUP:-10}"
NUM_ITERS="${NUM_ITERS:-30}"
MAX_SECONDS="${MAX_SECONDS:-300}"
OUT="${OUT:-forward_time_2d}"

JSONL="${CODE_MOUNT}/benchmarks/results/${OUT}.jsonl"
PNG="${CODE_MOUNT}/benchmarks/results/${OUT}.png"

# ── In-container command ─────────────────────────────────────────────────────
read -r -d '' COMMAND <<EOF
source /opt/conda/etc/profile.d/conda.sh && conda activate base
export CUDA_VISIBLE_DEVICES=0
cd ${CODE_MOUNT}

echo "======================================================"
echo "2D forward-time benchmark"
echo "mixers='${MIXERS}' hidden_dim=${HIDDEN_DIM} grid=${GRID_TYPE} backend=${FFT_BACKEND}"
echo "resolutions='${RESOLUTIONS}' dtype=${DTYPE} batch=${BATCH_SIZE} max_s=${MAX_SECONDS}"
echo "======================================================"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
python -c "import subquadratic_ops_torch; print('subq_ops: available')" \
    || echo "WARNING: subquadratic_ops_torch missing — hyena subq_ops points will error."

PYTHONPATH=. python benchmarks/benchmark_forward_time_2d_resolution.py \
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

# Render the plot in-job (matplotlib ships in the dev image); harmless if absent.
python -c "import matplotlib" 2>/dev/null \
    && PYTHONPATH=. python scripts/visualization/visualize_forward_time_2d.py \
           --input ${JSONL} --out ${PNG} \
    || echo "[plot] matplotlib unavailable — plot on the login node from ${OUT}.jsonl."
EOF

srun \
    --output "${RESULTS_HOST}/${OUT}-%j.out" \
    --error  "${RESULTS_HOST}/${OUT}-error-%j.out" \
    --container-image="${SQSH_PATH}" \
    --container-mounts "${MOUNTS}" \
    bash -c "${COMMAND}"

set +x
