#!/bin/bash
#SBATCH --account=healthcareeng_research
#SBATCH --nodes=1
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --exclusive
#SBATCH --job-name=healthcareeng_research-nvsubq.v5hybrid

# Usage (from repo root):
#   sbatch slurm/submit_hybrid.sh examples/vit5_imagenet/vit5_hybrid/full_attention.py
#   sbatch slurm/submit_hybrid.sh examples/vit5_imagenet/vit5_hybrid/full_attention.py \
#       net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=4
#
# Via queue.sh for chaining (recommended for 800-epoch runs):
#   bash slurm/queue.sh slurm/submit_hybrid.sh 12 \
#       examples/vit5_imagenet/vit5_hybrid/full_attention.py

set -x

if [ -z "${1:-}" ]; then
    echo "Usage: sbatch slurm/submit_hybrid.sh <config.py> [overrides...]"
    exit 1
fi

# Capture start time immediately
JOB_START_TIMESTAMP=$(date +%s)
echo "Start time captured: ${JOB_START_TIMESTAMP}"

# Container mount paths
CONTAINER_DATA="/workspace/data"
CONTAINER_RESULTS="/workspace/results"

# ============================================================================
# Configuration
# ============================================================================
TIME_LIMIT_HOURS=4
EXPERIMENT_NAME="$(basename "${1%.py}")"
CONFIG_FILE="$1"; shift

CONFIG_OVERRIDES="num_nodes=${SLURM_JOB_NUM_NODES}"
CONFIG_OVERRIDES="${CONFIG_OVERRIDES} experiment_dir=${CONTAINER_RESULTS}"
CONFIG_OVERRIDES="${CONFIG_OVERRIDES} compile_mode=max-autotune-no-cudagraphs"
# Append any extra overrides passed on the command line
for arg in "$@"; do
    CONFIG_OVERRIDES="${CONFIG_OVERRIDES} ${arg}"
done

# Container image
IMAGE_NAME="${SQSH_IMAGE:-/lustre/fsw/healthcareeng_bionemo/amoradzadeh/hyena/enroot/nvsubquadratic-slurm-x86_64-04-17-2026.sqsh}"

# Host paths
WORKDIR="${PWD}"
RUNS_DIR="${WORKDIR}/runs"
DATA_DIR="${IMAGENET_HOST_DIR:-/lustre/fsw/healthcareeng_bionemo/amoradzadeh/hyena}"

# Create necessary directories
mkdir -p "${RUNS_DIR}"

# ============================================================================
# Run naming (deterministic hash -> same dir across restarts)
# ============================================================================
RUN_NAME_HASH=$(echo "${CONFIG_FILE} ${CONFIG_OVERRIDES} ${EXPERIMENT_NAME}" | md5sum | awk '{print $1}' | cut -c1-8)
RUN_NAME="run_${RUN_NAME_HASH}"

EXPERIMENT_DIR="${RUNS_DIR}/${EXPERIMENT_NAME}"
RESULTS_PATH="${EXPERIMENT_DIR}/${RUN_NAME}"

mkdir -p "${EXPERIMENT_DIR}"
mkdir -p "${RESULTS_PATH}"

# ============================================================================
# W&B run ID -- persisted across restarts so resume attaches to the same run
# ============================================================================
if [ -f "${RESULTS_PATH}/run.id" ]; then
    RUN_ID=$(<"${RESULTS_PATH}/run.id")
    echo "Resuming with existing W&B run ID: ${RUN_ID}"
else
    array=()
    for i in {a..z} {A..Z} {0..9}; do
        array[$RANDOM]=$i
    done
    RUN_ID=$(printf %s "${array[@]::8}")
    echo "${RUN_ID}" > "${RESULTS_PATH}/run.id"
    echo "Generated new W&B run ID: ${RUN_ID}"
fi

echo "$(date): Job ${SLURM_JOB_ID} started (W&B run ID: ${RUN_ID})" >> "${RESULTS_PATH}/job_chain.log"

# ============================================================================
# Container mounts
# ============================================================================
MOUNTS="${DATA_DIR}:${CONTAINER_DATA}"
MOUNTS="${MOUNTS},${RESULTS_PATH}:${CONTAINER_RESULTS}"
MOUNTS="${MOUNTS},${WORKDIR}:/workspaces/nvSubquadratic-private"
MOUNTS="${MOUNTS},$HOME/.cache:/root/.cache"

if [ -f "$HOME/.netrc" ]; then
    MOUNTS="${MOUNTS},$HOME/.netrc:/root/.netrc"
fi

# Code is baked into the container
WORK_DIR="/workspaces/nvSubquadratic-private"
CONFIG_PATH="${WORK_DIR}/${CONFIG_FILE}"

echo "================================================"
echo "Experiment:    ${EXPERIMENT_NAME}"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Run Name:      ${RUN_NAME}"
echo "Node(s):       ${SLURM_NODELIST}"
echo "Num nodes:     ${SLURM_JOB_NUM_NODES}"
echo "GPUs per node: ${SLURM_NTASKS_PER_NODE}"
echo "Total GPUs:    $((${SLURM_JOB_NUM_NODES} * ${SLURM_NTASKS_PER_NODE}))"
echo "Config:        ${CONFIG_FILE}"
echo "Overrides:     ${CONFIG_OVERRIDES}"
echo "Container:     ${IMAGE_NAME}"
echo "Results:       ${RESULTS_PATH}"
echo "Mounts:        ${MOUNTS}"
echo "================================================"

# ============================================================================
# Environment Setup
# ============================================================================
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONPATH="."
export DALI_NO_MMAP=1
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ImageNet paths (container-side) -- imagenet_folder is the ImageFolder layout
export IMAGENET_PATH="${CONTAINER_DATA}/imagenet_folder"
export IMAGENET_FOLDER_PATH="${CONTAINER_DATA}/imagenet_folder"
export LOCAL_STAGING_DIR="${CONTAINER_DATA}/imagenet_folder"

# ============================================================================
# Autoresume
# ============================================================================
if [ -f "${RESULTS_PATH}/checkpoints/last.ckpt" ]; then
    echo "Found existing checkpoint -- enabling autoresume"
    AUTORESUME_ARG="autoresume.enabled=True"
else
    echo "No existing checkpoint found, starting fresh"
    AUTORESUME_ARG="wandb.run_id=${RUN_ID}"
fi

# ============================================================================
# Training command
# ============================================================================
read -r -d '' PYTHON_CMD <<EOF || true
export HF_HOME=${CONTAINER_DATA}/.hf && \
export HF_HUB_CACHE=${CONTAINER_DATA}/.hf/hub && \
export HF_DATASETS_CACHE=${CONTAINER_DATA}/.hf/datasets && \
export TRANSFORMERS_CACHE=${CONTAINER_DATA}/.hf/transformers && \
export WANDB_DIR=${CONTAINER_RESULTS}/wandb && \
export WANDB_CACHE_DIR=${CONTAINER_RESULTS}/.cache/wandb && \
export WANDB_DATA_DIR=${CONTAINER_RESULTS}/.wandbstage && \
cd ${WORK_DIR} && \
python -m experiments.run \
    --config ${CONFIG_PATH} \
    ${CONFIG_OVERRIDES} \
    ${AUTORESUME_ARG} \
    train.run_start_time=${JOB_START_TIMESTAMP} \
    train.run_time_limit_hours=${TIME_LIMIT_HOURS}
EOF

echo "Starting training at $(date)"
echo "Command: ${PYTHON_CMD}"

# ============================================================================
# Launch
# ============================================================================
srun \
    --output "${RESULTS_PATH}/slurm-%j-%n.out" \
    --error  "${RESULTS_PATH}/error-%j-%n.out" \
    --export=ALL \
    --container-image="${IMAGE_NAME}" \
    --container-mounts="${MOUNTS}" \
    bash -c "${PYTHON_CMD}"

TRAIN_EXIT_CODE=$?

echo "Training exited with code: ${TRAIN_EXIT_CODE} at $(date)"
echo "$(date): Job ${SLURM_JOB_ID} completed with exit code ${TRAIN_EXIT_CODE}" >> "${RESULTS_PATH}/job_chain.log"

set +x
exit ${TRAIN_EXIT_CODE}
