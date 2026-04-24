#!/bin/bash
#SBATCH --account=healthcareeng_research
#SBATCH --nodes=1
#SBATCH --partition=polar,polar3,polar4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=healthcareeng_research-nvsubq.imgnet_convert

# Converts HuggingFace Arrow ImageNet-1k cache to torchvision ImageFolder layout.
# Run from the repo root:
#   sbatch slurm/convert_imagenet_to_folder.sh

set -x

DATA_DIR="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/amoradzadeh/hyena"
SQSH="${DATA_DIR}/nvsubquadratic-slurm-x86_64-04-22-2026.sqsh"
WORKDIR="${PWD}"

# Input: HF Arrow cache
HF_CACHE="${DATA_DIR}/imagenet"
# Output: ImageFolder layout (train/0000/*.jpg, val/0000/*.jpg)
OUTPUT_DIR="${DATA_DIR}/imagenet_folder"

MOUNTS="${DATA_DIR}:/workspace/data"
MOUNTS="${MOUNTS},${WORKDIR}:/workspaces/nvSubquadratic-private"
MOUNTS="${MOUNTS},$HOME/.cache:/root/.cache"

if [ -f "$HOME/.netrc" ]; then
    MOUNTS="${MOUNTS},$HOME/.netrc:/root/.netrc"
fi

WORK_DIR="/workspaces/nvSubquadratic-private"

srun \
    --output "${WORKDIR}/convert_imagenet-%j.out" \
    --error  "${WORKDIR}/convert_imagenet-%j.err" \
    --export=ALL \
    --container-image="${SQSH}" \
    --container-mounts="${MOUNTS}" \
    bash -c "
set -e
cd ${WORK_DIR}
export PYTHONPATH='.'
export IMAGENET_PATH='/workspace/data/imagenet'
export IMAGENET_FOLDER_PATH='/workspace/data/imagenet_folder'
export HF_TOKEN=${HF_TOKEN:-}

echo '=== ImageNet Arrow → ImageFolder conversion ==='
echo 'Input  (HF Arrow): '\${IMAGENET_PATH}
echo 'Output (ImageFolder): '\${IMAGENET_FOLDER_PATH}
python scripts/data/extract_imagenet_to_folder.py
echo '=== conversion complete ==='
"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Conversion succeeded. ImageFolder at: ${OUTPUT_DIR}"
else
    echo "Conversion FAILED with exit code ${EXIT_CODE}"
fi
exit ${EXIT_CODE}
