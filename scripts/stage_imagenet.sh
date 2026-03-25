#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=low
#SBATCH --container-image=/shared/images/nvsubquadratic_cuda129.sqsh
#SBATCH --container-name=nv-subq-stage
#SBATCH --container-writable
#SBATCH --container-mounts="/home/dwromero:/home/dwromero,/shared:/shared,/scratch:/scratch"
#SBATCH --container-workdir=/home/dwromero
#SBATCH --output=/home/dwromero/projects/nvSubquadratic-private/logs/stage_%j_%N.out
#SBATCH --error=/home/dwromero/projects/nvSubquadratic-private/logs/stage_%j_%N.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G

set -eo pipefail

SRC="/shared/data/image_datasets/imagenet_folder"
DST="/scratch/dwromero/imagenet_dataset"
SENTINEL="${DST}/.staging_complete"

echo "[stage] Node: $(hostname)"
echo "[stage] Source: ${SRC}"
echo "[stage] Destination: ${DST}"

if [ -f "${SENTINEL}" ]; then
    echo "[stage] Sentinel found — data already staged. Nothing to do."
    exit 0
fi

mkdir -p "${DST}"

FREE_GB=$(df --output=avail /scratch | tail -1 | awk '{printf "%.0f", $1/1048576}')
echo "[stage] Free space on /scratch: ${FREE_GB} GB"
if [ "${FREE_GB}" -lt 160 ]; then
    echo "[stage] ERROR: Not enough space (need 160 GB, have ${FREE_GB} GB)"
    exit 1
fi

echo "[stage] Copying ${SRC} → ${DST} ..."
cp -a --no-clobber -r "${SRC}/train" "${SRC}/val" "${DST}/"

echo "ok" > "${SENTINEL}"
echo "[stage] Done. Sentinel written."
