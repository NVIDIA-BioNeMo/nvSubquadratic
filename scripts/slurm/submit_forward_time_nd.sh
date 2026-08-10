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
#   # HyenaND) — use the ready-made wrapper, which sets head_dim 128 (hidden 512 /
#   # 4 heads), the regime these flash kernels are OPTIMIZED for. flex/fa4 only
#   # *accept* head_dim >= 16, but 64/128 is where they (and real models) run fast;
#   # 16/32 sits on a slow small-tile path. Needs flash-attn-4 in the image for the
#   # fa4 series (absent -> 'unavailable', omitted from the plot):
#   scripts/slurm/submit_forward_time_flash_kernels.sh          # 2D, head_dim 128
#   DATA_DIM=1 scripts/slurm/submit_forward_time_flash_kernels.sh
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
CODE_PATH="${CODE_PATH:-/lustre/fsw/healthcareeng_bionemo/farhadr/nvsubquadratic_workdir/nvSubquadratic}"
CODE_MOUNT=/workspace/nvsubq
RESULTS_HOST="${CODE_PATH}/benchmarks/results"
MOUNTS="${CODE_PATH}:${CODE_MOUNT}"

# Optional kernel override: a host directory of pre-downloaded aarch64 wheels for
# subquadratic-ops-torch-cu13. The image bakes in whatever the [cuda] extra
# resolved at build time (0.2.1 from public PyPI); FFT_BACKEND=subq_ops_fused
# needs >= 0.2.2, which today ships only from the internal NVIDIA GitLab registry.
# Rather than rebuild the .sqsh (no docker on the login node), stage the wheels
# once and pip-install them into the container at job start:
#
#   pip download --no-deps --only-binary=:all: \
#       --platform manylinux_2_28_aarch64 --python-version 3.12 --abi cp312 \
#       -d /lustre/.../subq_ops_wheels 'subquadratic-ops-torch-cu13>=0.2.2' \
#       --index-url https://__token__:<TOKEN>@gitlab-master.nvidia.com/api/v4/projects/180496/packages/pypi/simple
#   SUBQ_OPS_WHEEL_DIR=/lustre/.../subq_ops_wheels sbatch scripts/slurm/submit_forward_time_nd.sh
#
# Installed with --no-index so the compute node needs no network access.
SUBQ_OPS_WHEEL_DIR="${SUBQ_OPS_WHEEL_DIR:-}"
WHEEL_MOUNT=/workspace/subq_wheels
if [[ -n "${SUBQ_OPS_WHEEL_DIR}" ]]; then
    if [[ ! -d "${SUBQ_OPS_WHEEL_DIR}" ]]; then
        echo "SUBQ_OPS_WHEEL_DIR='${SUBQ_OPS_WHEEL_DIR}' is not a directory"; exit 1
    fi
    MOUNTS="${MOUNTS},${SUBQ_OPS_WHEEL_DIR}:${WHEEL_MOUNT}:ro"
fi

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
# Per-dim resolution sweep (R): every power of two from a 16-wide grid up to ~16M
# tokens. Override with RESOLUTIONS=.
#   1D: L = R      2D: L = R^2      3D: L = R^3
case "${DATA_DIM}" in
    1) RESOLUTIONS="${RESOLUTIONS:-16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216}" ;;
    2) RESOLUTIONS="${RESOLUTIONS:-16 32 64 128 256 512 1024 2048 4096}" ;;
    3) RESOLUTIONS="${RESOLUTIONS:-16 32 64 128 256}" ;;
    *) echo "DATA_DIM must be 1, 2, or 3 (got '${DATA_DIM}')"; exit 1 ;;
esac
# subq_ops_fused is the fused 2D cuFFTDx kernel (subquadratic-ops-torch >= 0.2.2).
# It is 2D-only and capped at 64 per axis, so the benchmark resolves it PER POINT:
# 2D R<=64 runs fused, 2D R>=128 and all of 1D fall back to subq_ops, 3D to
# torch_fft. Set FFT_BACKEND=subq_ops for the pre-0.2.2 behaviour everywhere.
FFT_BACKEND="${FFT_BACKEND:-subq_ops_fused}"
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

# Kernel override, if staged. Offline install so no compute-node network is needed.
if [ -d "${WHEEL_MOUNT}" ]; then
    echo "[kernel] installing staged subq_ops wheels from ${WHEEL_MOUNT}"
    pip install --no-index --find-links "${WHEEL_MOUNT}" --upgrade --no-deps \
        subquadratic-ops-torch-cu13 || {
        echo "[kernel] ERROR: staged wheel install failed — aborting rather than"
        echo "[kernel]        silently benchmarking the image's older kernel."
        exit 1
    }
fi

python -c "import subquadratic_ops_torch; print('subq_ops: available')" \
    || echo "WARNING: subquadratic_ops_torch missing — hyena subq_ops points will error."
python -c "
from importlib.metadata import version
v = version('subquadratic-ops-torch-cu13')
print('subq_ops version:', v)
" || echo "WARNING: could not read subq_ops version."
# subq_ops_fused needs >= 0.2.2; fail loudly instead of silently benchmarking a fallback.
if [ "${FFT_BACKEND}" = "subq_ops_fused" ]; then
    python -c "
import sys
from subquadratic_ops_torch.fused_fft_conv2d import fused_fft_conv2d  # noqa: F401
print('subq_ops: fused_fft_conv2d available')
" || { echo "ERROR: FFT_BACKEND=subq_ops_fused but fused_fft_conv2d is missing (needs >= 0.2.2)."; exit 1; }
fi

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
