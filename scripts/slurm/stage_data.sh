#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --partition=all
#SBATCH --time=02:00:00
#SBATCH --output=/home/david.romero/projects/nvSubquadratic-private/logs/stage-data_%j_%N.out
#SBATCH --error=/home/david.romero/projects/nvSubquadratic-private/logs/stage-data_%j_%N.err

set -euo pipefail

SRC="/shared/data/image_datasets/imagenet_folder"
DST="/scratch/david.romero/imagenet_dataset"

echo "=== $(hostname) — $(date) ==="
echo "Staging $SRC -> $DST"

if [ -d "$DST/train" ] && [ -d "$DST/val" ]; then
    SRC_TRAIN_COUNT=$(ls -1 "$SRC/train" 2>/dev/null | wc -l)
    DST_TRAIN_COUNT=$(ls -1 "$DST/train" 2>/dev/null | wc -l)
    SRC_VAL_COUNT=$(ls -1 "$SRC/val" 2>/dev/null | wc -l)
    DST_VAL_COUNT=$(ls -1 "$DST/val" 2>/dev/null | wc -l)
    if [ "$SRC_TRAIN_COUNT" -eq "$DST_TRAIN_COUNT" ] && [ "$SRC_VAL_COUNT" -eq "$DST_VAL_COUNT" ]; then
        echo "Data already staged (train=$DST_TRAIN_COUNT dirs, val=$DST_VAL_COUNT dirs). Nothing to do."
        exit 0
    fi
fi

mkdir -p "$DST"
echo "Starting rsync ..."
rsync -a --info=progress2 "$SRC/" "$DST/"
echo "=== DONE on $(hostname) — $(date) ==="
