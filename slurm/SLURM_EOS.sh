#!/bin/bash
#
# EOS Cluster Training Job with Automatic Checkpointing using Singleton Dependency
#
# Usage:
#   1. Edit the "Configuration" section below
#   2. Submit multiple copies: for i in $(seq 32); do sbatch slurm/eos_job.sh; done
#   3. SLURM will run them one at a time until training completes
#
#SBATCH --account=healthcareeng_research    # account (adjust to your account)
#SBATCH --nodes=1                           # number of nodes (adjust based on gpu needs)
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=8                 # 8 GPUs per node on EOS
#SBATCH --time=04:00:00                     # 4h (under 4h limit with buffer)
#SBATCH --mem=0                             # all mem avail
#SBATCH --mail-type=FAIL
#SBATCH --exclusive
#SBATCH --job-name=healthcareeng_research-nvsubq.imagenet64        # IMPORTANT: Keep same name for singleton to work
#SBATCH --dependency=singleton              # Only one job with this name runs at a time

set -x

# ============================================================================
# Configuration - Edit these for your job
# ============================================================================
EXPERIMENT_NAME="imagenet64_hyena_baseline"  # Give your experiment a meaningful name
CONFIG_FILE="examples/imagenet_classification/ccnn_7_512_hyena.py"
CONFIG_OVERRIDES="trainer.num_nodes=${SLURM_JOB_NUM_NODES}"  # Auto-set nodes and disable wandb with wandb.mode=disabled and export WANDB_MODE=disabled

# Container configuration
SQSH_FILE="nvsubquadratic-amd64.sqsh"  # Name of your container image
SQSH_PATH="/lustre/fsw/healthcareeng_bionemo/farhadr/enroot/${SQSH_FILE}"

# Host paths
WORKDIR="/lustre/fsw/healthcareeng_bionemo/farhadr/nvsubquadratic_workdir"
RUNS_DIR="${WORKDIR}/runs"
DATA_DIR="${WORKDIR}/data"

# Container mount paths (where host paths will be mounted inside container)
CONTAINER_DATA="/workspace/data"
CONTAINER_RESULTS="/workspace/results"

# Create necessary directories
mkdir -p ${RUNS_DIR}

# Generate a deterministic run name for this config (without timestamp for resume)
# This allows PyTorch Lightning to automatically find and resume from checkpoints
RUN_NAME_HASH=$(echo "${CONFIG_FILE} ${CONFIG_OVERRIDES}" | md5sum | awk '{print $1}' | cut -c1-8)
RUN_NAME="run_${RUN_NAME_HASH}"

# Experiment-specific directories
EXPERIMENT_DIR="${RUNS_DIR}/${EXPERIMENT_NAME}"
RESULTS_PATH="${EXPERIMENT_DIR}/${RUN_NAME}"
COMPLETION_FLAG="${RESULTS_PATH}/.training_complete"

# Create necessary directories
mkdir -p ${EXPERIMENT_DIR}
mkdir -p ${RESULTS_PATH}

# Generate (or retrieve) a unique, shared ID per run to handle restarts in W&B and Tensorboard
# =========================

if [ -f ${RESULTS_PATH}/run.id ]; then
    RUN_ID=$(<${RESULTS_PATH}/run.id)
    echo "Resuming with existing W&B run ID: ${RUN_ID}"
else
    array=()
    for i in {a..z} {A..Z} {0..9}; do
        array[$RANDOM]=$i
    done
    RUN_ID=$(printf %s ${array[@]::8})
    echo $RUN_ID > ${RESULTS_PATH}/run.id
    echo "Generated new W&B run ID: ${RUN_ID}"
fi

# Log this job in the chain
echo "$(date): Job ${SLURM_JOB_ID} started (W&B run ID: ${RUN_ID})" >> ${RESULTS_PATH}/job_chain.log

# ============================================================================
# Environment Setup
# ============================================================================
# Build mount string for container - only mount data and results
MOUNTS="${DATA_DIR}:${CONTAINER_DATA}"
MOUNTS="${MOUNTS},${RESULTS_PATH}:${CONTAINER_RESULTS}"
MOUNTS="${MOUNTS},$HOME/.cache:/root/.cache"

# Add netrc if it exists (for W&B, HF authentication)
if [ -f "$HOME/.netrc" ]; then
    MOUNTS="${MOUNTS},$HOME/.netrc:/root/.netrc"
fi

# Set paths for use inside container
# Note: Code is assumed to be in the container already
WORK_DIR="/workspaces/nvSubquadratic-private"
CONFIG_PATH="${WORK_DIR}/${CONFIG_FILE}"
CHECKPOINT_DIR="${CONTAINER_RESULTS}/checkpoints"

echo "================================================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run Name: ${RUN_NAME}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Number of nodes: ${SLURM_JOB_NUM_NODES}"
echo "GPUs per node: ${SLURM_NTASKS_PER_NODE}"
echo "Total GPUs: $((${SLURM_JOB_NUM_NODES} * ${SLURM_NTASKS_PER_NODE}))"
echo "Config: ${CONFIG_FILE}"
echo "Overrides: ${CONFIG_OVERRIDES}"
echo "Container: ${SQSH_PATH}"
echo "Checkpoint dir: ${RESULTS_PATH}/checkpoints"
echo "Mounts: ${MOUNTS}"
echo "================================================"

# Export environment variables (will be passed to container)
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
export PYTHONPATH="."
export IMAGENET_CACHE="${CONTAINER_DATA}/imagenet"

# Set NCCL and memory config for training
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Check if training is already complete
# ============================================================================
if [ -f ${COMPLETION_FLAG} ]; then
    echo "Training already complete for this run. Exiting."
    echo "$(date): Job ${SLURM_JOB_ID} exited early - training complete" >> ${RESULTS_PATH}/job_chain.log
    exit 0
fi

# ============================================================================
# Run Training
# ============================================================================
echo "Starting/resuming training at $(date)"

# Check if we should use autoresume (when checkpoints exist on host)
# Note: We check the host path since the container hasn't started yet
if [ -f "${RESULTS_PATH}/checkpoints/last.ckpt" ]; then
    echo "Found existing checkpoint: ${RESULTS_PATH}/checkpoints/last.ckpt"
    echo "Will enable autoresume mode"
    # Enable autoresume which will automatically pick up the last checkpoint
    # Pass the W&B run ID and checkpoint directory to ensure continuity across job chains
    AUTORESUME_ARG="autoresume.enabled=True autoresume.run_name=${RUN_NAME} wandb.run_id=${RUN_ID} trainer.default_root_dir=${CONTAINER_RESULTS}"
else
    echo "No existing checkpoint found, starting fresh"
    # For fresh runs, pass the W&B run ID and checkpoint directory
    AUTORESUME_ARG="wandb.run_id=${RUN_ID} trainer.default_root_dir=${CONTAINER_RESULTS}"
fi

# Build the python command
PYTHON_CMD="cd ${WORK_DIR} && python -m experiments.run --config ${CONFIG_PATH} ${CONFIG_OVERRIDES} ${AUTORESUME_ARG}"

echo "Launching training in container with srun..."
echo "Command: ${PYTHON_CMD}"

# Run training in container
srun \
    --container-image=${SQSH_PATH} \
    --container-mounts=${MOUNTS} \
    bash -c "${PYTHON_CMD}"

TRAIN_EXIT_CODE=$?

echo "Training process exited with code: ${TRAIN_EXIT_CODE} at $(date)"
echo "$(date): Job ${SLURM_JOB_ID} completed with exit code ${TRAIN_EXIT_CODE}" >> ${RESULTS_PATH}/job_chain.log

# Check if training completed and mark it
if [ -f "${RESULTS_PATH}/checkpoints/.training_complete" ]; then
    echo "Training completion marker found - marking as complete"
    touch ${COMPLETION_FLAG}
fi

set +x

exit ${TRAIN_EXIT_CODE}
