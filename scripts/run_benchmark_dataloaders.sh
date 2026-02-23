#!/bin/bash
#SBATCH --job-name=benchmark_dl
#SBATCH --account=ceesusers
#SBATCH --partition=cees
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/%x_%j.out

# We run on 1 GPU and 16 CPUs with 14 data loader workers to match the training worker-to-GPU ratio
# (Training uses 8 GPUs and 112 cpus -> 14 cpus per GPU, 14 workers per GPU).

set -euo pipefail

# Setup env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

cd /home/dknigge/code/nvSubquadratic-private
export PYTHONPATH=.
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)

echo "Starting dataloader benchmark..."
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

python scripts/benchmark_dataloaders.py \
    --batch-size 128 \
    --num-workers 14 \
    --num-batches 200 \
    --device cuda

echo "Benchmark complete!"
