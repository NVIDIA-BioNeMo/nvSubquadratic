#!/bin/bash
#SBATCH --job-name=copy_imagenet
#SBATCH --output=slurm/copy_imagenet_%j.out
#SBATCH --account=ceesusers
#SBATCH --partition=cees
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

# Source and destination paths
SRC_DIR="/home/dknigge/code/nvSubquadratic-private/data/imagenet_folder"
DEST_DIR="/local_scratch/dknigge/imagenet_folder"

echo "Creating destination directory: $DEST_DIR"
mkdir -p "$DEST_DIR"

echo "Starting copy operation at $(date)"

# Using a tar pipe is generally much faster than cp or rsync for millions
# of tiny files (like ImageNet) because it avoids the overhead of reading/writing 
# individual file metadata back and forth across abstraction layers.
cd "$SRC_DIR" && tar -cf - . | (cd "$DEST_DIR" && tar -xf -)

echo "Copy operation finished at $(date)"

echo "Verifying sizes:"
echo "Source:"
du -sh "$SRC_DIR"
echo "Destination:"
du -sh "$DEST_DIR"
