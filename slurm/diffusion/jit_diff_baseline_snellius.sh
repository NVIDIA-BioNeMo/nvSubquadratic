#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_h100
#SBATCH --chdir="/gpfs/home1/dknigge/nvsq_update_branch"
#SBATCH --output=./runs/slurm/in1k_jit_baseline_%j.out
#SBATCH --error=./runs/slurm/in1k_jit_baseline_%j.err
#SBATCH --job-name='in1k-jit_baseline'
#SBATCH --dependency=singleton

source /home/dknigge/.bashrc
source /gpfs/home1/dknigge/nvsq_update_branch/.venv/bin/activate

# HF dataset env vars (mirrors extract_imagenet_to_tar.py defaults)
HF_DATASET="${IMAGENET_HF_DATASET:-imagenet-1k}"
HF_CACHE="${IMAGENET_PATH:-$HOME/project_dir/huggingface/imagenet}"

LOCAL_SCRATCH="/scratch-local/${USER}"
LOCAL_HF_CACHE="${LOCAL_SCRATCH}/hf_cache"
SENTINEL="${LOCAL_HF_CACHE}/.hf_cache_complete"
mkdir -p "${LOCAL_HF_CACHE}"

# Capture the start time before staging so walltime budget is accurate
JOB_START_TIMESTAMP=$(date +%s)

echo "Staging HF cache to node-local storage: ${LOCAL_HF_CACHE}"
if [ -f "${SENTINEL}" ]; then
  echo "Sentinel found — local HF cache is complete; skipping staging."
else
  if [ ! -d "${HF_CACHE}" ]; then
    echo "ERROR: HF cache not found at ${HF_CACHE}."
    exit 1
  fi
  echo "Copying HF cache to local NVMe..."
  rsync -a --info=progress2 "${HF_CACHE}/" "${LOCAL_HF_CACHE}/"
  touch "${SENTINEL}"
  echo "Staging complete."
fi

export IMAGENET_HF_DATASET="${HF_DATASET}"
export IMAGENET_PATH="${LOCAL_HF_CACHE}"
echo "Start time captured: ${JOB_START_TIMESTAMP}"
WORKDIR="/gpfs/home1/dknigge/nvsq_update_branch"
TIME_LIMIT_HOURS=48
CONFIG_FILE="examples/imagenet_diffusion/jit_baseline.py"
CONFIG_OVERRIDES=(
  "wandb.job_group=in1k-jit_baseline"
  "wandb.entity=implicit-long-convs"
  # Note: Batch size is handled in config now (256). 
  # If we need to override batch size per GPU here we can, but let's stick to config file default.
)
EXPERIMENT_NAME="imagenet_diffusion_jit_baseline"

RUNS_DIR="${WORKDIR}/runs"
mkdir -p "${RUNS_DIR}/slurm"
RUN_NAME_HASH=$( (echo -n "${CONFIG_FILE} "; printf "%s " "${CONFIG_OVERRIDES[@]}") | md5sum | awk '{print $1}' | cut -c1-8)
RUN_NAME="run_${RUN_NAME_HASH}"
EXPERIMENT_DIR="${RUNS_DIR}/${EXPERIMENT_NAME}/${RUN_NAME}"

mkdir -p "${EXPERIMENT_DIR}/checkpoints"

# Ensure a stable W&B run ID so autoresume works across jobs
if [ -f "${EXPERIMENT_DIR}/run.id" ]; then
    RUN_ID=$(<"${EXPERIMENT_DIR}/run.id")
    echo "Resuming with existing W&B run ID: ${RUN_ID}"
else
    # Simple random ID generation
    RUN_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
    echo "${RUN_ID}" > "${EXPERIMENT_DIR}/run.id"
    echo "Generated new W&B run ID: ${RUN_ID}"
fi

# Start fresh so optimizer/scheduler changes always take effect cleanly
AUTORESUME_OVERRIDES=("autoresume.enabled=False" "experiment_dir=${EXPERIMENT_DIR}")
echo "Starting fresh with autoresume disabled for this launch"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"

# Redirect W&B cache to node-local scratch to avoid filling home quota
export WANDB_CACHE_DIR="${LOCAL_SCRATCH}/wandb_cache"
export WANDB_DATA_DIR="${LOCAL_SCRATCH}/wandb_data"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

echo "Running on node: $SLURM_NODELIST"

# NOTE: run.py does NOT support --experiment_dir flag in argparse. 
# We must pass it as a config override: experiment_dir=...
PYTHON_CMD=(python experiments/run.py --config "${CONFIG_FILE}")
PYTHON_CMD+=("experiment_dir=${EXPERIMENT_DIR}")
PYTHON_CMD+=("${CONFIG_OVERRIDES[@]}")
PYTHON_CMD+=("${AUTORESUME_OVERRIDES[@]}")
# Pass walltime info
PYTHON_CMD+=("train.run_start_time=${JOB_START_TIMESTAMP}" "train.run_time_limit_hours=${TIME_LIMIT_HOURS}")

printf "Command: %q " "${PYTHON_CMD[@]}"; echo
"${PYTHON_CMD[@]}"
