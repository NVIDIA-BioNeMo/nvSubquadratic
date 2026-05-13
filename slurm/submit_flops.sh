#!/bin/bash
#SBATCH --account=healthcareeng_research
#SBATCH --nodes=1
#SBATCH --partition=batch,backfill
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --job-name=healthcareeng_research-nvsubq.flops

# Usage (from repo root):
#   sbatch slurm/submit_flops.sh examples/well/v2/active_matter/hyena_gaussian_mask.py
#   sbatch slurm/submit_flops.sh examples/well/v2/active_matter/hyena_gaussian_mask.py \
#       net.in_proj_cfg.patch_size=8 dataset.batch_size=4
#
# Runs benchmarks/well/measure_flops.py inside the same container as training,
# writes flops.json into runs/<experiment>/flops_<jobid>/.

set -x

if [ -z "${1:-}" ]; then
    echo "Usage: sbatch slurm/submit_flops.sh <config.py> [overrides...]"
    exit 1
fi

CONTAINER_DATA="/workspace/data"
CONTAINER_RESULTS="/workspace/results"

EXPERIMENT_NAME="$(basename "${1%.py}")"
CONFIG_FILE="$1"; shift

CONFIG_OVERRIDES=()
PATCH_SIZE=""
for arg in "$@"; do
    CONFIG_OVERRIDES+=("${arg}")
    if [[ "${arg}" == net.in_proj_cfg.patch_size=* ]]; then
        PATCH_SIZE="${arg#net.in_proj_cfg.patch_size=}"
    fi
done

IMAGE_NAME="${SQSH_IMAGE:-/lustre/fsw/healthcareeng_research/oviessmann/nvsubquadratic/enroot/nvsubquadratic-x86_64.sqsh}"

WORKDIR="${PWD}"
RUNS_DIR="${WORKDIR}/runs"
DATA_DIR="${WELL_HOST_DIR:-/lustre/fsw/healthcareeng_research/oviessmann/nvsubquadratic/data/well_data/datasets}"

mkdir -p "${RUNS_DIR}"

EXPERIMENT_DIR="${RUNS_DIR}/${EXPERIMENT_NAME}"
RESULTS_PATH="${EXPERIMENT_DIR}/flops_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_PATH}"

REPO_DIR="/lustre/fsw/healthcareeng_research/oviessmann/nvsubquadratic"
WORK_DIR="/workspaces/nvSubquadratic-private"
CONFIG_PATH="${WORK_DIR}/${CONFIG_FILE}"
if [ -n "${PATCH_SIZE}" ]; then
    OUT_FILENAME="flops_${EXPERIMENT_NAME}_patch${PATCH_SIZE}.json"
else
    OUT_FILENAME="flops_${EXPERIMENT_NAME}.json"
fi
OUT_PATH="${CONTAINER_RESULTS}/${OUT_FILENAME}"

MOUNTS="${DATA_DIR}:${CONTAINER_DATA}"
MOUNTS="${MOUNTS},${RESULTS_PATH}:${CONTAINER_RESULTS}"
MOUNTS="${MOUNTS},${REPO_DIR}:${WORK_DIR}"
MOUNTS="${MOUNTS},$HOME/.cache:/root/.cache"

echo "================================================"
echo "Experiment:    ${EXPERIMENT_NAME}"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node(s):       ${SLURM_NODELIST}"
echo "Config:        ${CONFIG_FILE}"
echo "Overrides:     ${CONFIG_OVERRIDES[*]}"
echo "Container:     ${IMAGE_NAME}"
echo "Results:       ${RESULTS_PATH}"
echo "Output JSON:   ${RESULTS_PATH}/${OUT_FILENAME}"
echo "Mounts:        ${MOUNTS}"
echo "================================================"

export PYTHONPATH="."
export CUDA_VISIBLE_DEVICES=0
export DALI_NO_MMAP=1
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WELL_DATA_PATH="${CONTAINER_DATA}"

OVERRIDES_STR="${CONFIG_OVERRIDES[*]}"

read -r -d '' PYTHON_CMD <<EOF || true
cd ${WORK_DIR} && \
PYTHONPATH=. python benchmarks/well/measure_flops.py \
    --config ${CONFIG_PATH} \
    --out ${OUT_PATH} \
    ${OVERRIDES_STR}
EOF

echo "Starting FLOP measurement at $(date)"
echo "Command: ${PYTHON_CMD}"

srun \
    --output "${RESULTS_PATH}/slurm-%j-%n.out" \
    --error  "${RESULTS_PATH}/error-%j-%n.out" \
    --export=ALL \
    --container-image="${IMAGE_NAME}" \
    --container-mounts="${MOUNTS}" \
    --container-writable \
    bash -c "${PYTHON_CMD}"

EXIT_CODE=$?

echo "FLOP measurement exited with code: ${EXIT_CODE} at $(date)"

set +x
exit ${EXIT_CODE}
