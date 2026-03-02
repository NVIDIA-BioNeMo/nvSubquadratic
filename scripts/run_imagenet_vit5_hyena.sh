#!/bin/bash
#SBATCH --job-name=vit5_hyena_dali
#SBATCH --account=ceesusers
#SBATCH --partition=cees
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:rtx_a5000:8
#SBATCH --cpus-per-task=13
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/vit5_hyena_dali_%j.out

set -eo pipefail

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================================="

# 1. Setup Environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dknigge/code/nvSubquadratic-private
export PYTHONPATH=.

# Make nvcc available from the node-local CUDA install so torch.compile
# max-autotune can use both Triton and CUDA C++ backends.
export CUDA_HOME=/usr/local/cuda-13
export PATH=$CUDA_HOME/bin:$PATH

# Run PyTorch Lightning training.
# SLURM will automatically configure DDP since we requested 8 GPUs.
CONFIG="examples/vit5_imagenet/v2/vit5_small_pretrain_multihead_hyena_cls_row_apex_fix_init.py"
srun python experiments/run.py --config "$CONFIG"

echo "Job finished at $(date)"