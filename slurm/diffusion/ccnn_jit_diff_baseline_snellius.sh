#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_h100
#SBATCH --chdir="/gpfs/home1/dknigge/nvsq_update_branch"
#SBATCH --output=./runs/slurm/in1k_ccnn_jit_baseline_%j.out
#SBATCH --error=./runs/slurm/in1k_ccnn_jit_baseline_%j.err
#SBATCH --job-name='in1k-ccnn_jit_baseline'
#SBATCH --dependency=singleton

source /home/dknigge/.bashrc
# mamba activate nvsq
source /gpfs/home1/dknigge/nvsq_update_branch/.venv/bin/activate

SHARED_IMAGENET_DIR="/scratch-shared/dknigge/imagenet_imagefolder"

if [ -d /scratch-local ]; then
  LOCAL_IMAGENET_ROOT="/scratch-local/${USER}"
else
  LOCAL_IMAGENET_ROOT="${TMPDIR:-/tmp}/${USER}"
fi
LOCAL_IMAGENET_DIR="${LOCAL_IMAGENET_ROOT}/imagenet_imagefolder"
SENTINEL="${LOCAL_IMAGENET_DIR}/.imagenet_complete"
mkdir -p "${LOCAL_IMAGENET_ROOT}"

# Capture the start time before staging so walltime budget is accurate
JOB_START_TIMESTAMP=$(date +%s)

echo "Staging ImageNet to node-local storage: ${LOCAL_IMAGENET_DIR}"
if [ -f "${SENTINEL}" ]; then
  echo "Sentinel found — local ImageNet copy is complete; skipping staging."
else
  echo "No complete local ImageNet copy found; staging from pre-packed tar."
  rm -rf "${LOCAL_IMAGENET_DIR}"
  PACKED_TAR="/scratch-shared/dknigge/imagenet_imagefolder.tar"
  if [ ! -f "${PACKED_TAR}" ]; then
    echo "ERROR: Pre-packed tar not found at ${PACKED_TAR}. Run slurm/pack_imagenet_tar.sh first."
    exit 1
  fi
  tar xf "${PACKED_TAR}" -C "${LOCAL_IMAGENET_ROOT}/"
  touch "${SENTINEL}"
  echo "Staging complete."
fi

export IMAGENET_DIR="${LOCAL_IMAGENET_DIR}"
echo "Start time captured: ${JOB_START_TIMESTAMP}"
WORKDIR="/gpfs/home1/dknigge/nvsq_update_branch"
TIME_LIMIT_HOURS=48
CONFIG_FILE="examples/imagenet_diffusion/ccnn_jit_baseline.py"
CONFIG_OVERRIDES=(
  "wandb.job_group=imagenet_diffusion_ccnn_jit_baseline"
  "wandb.entity=implicit-long-convs"
  # Note: Batch size is handled in config now (256). 
  # If we need to override batch size per GPU here we can, but let's stick to config file default.
)
EXPERIMENT_NAME="imagenet_diffusion_ccnn_jit_baseline"

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
