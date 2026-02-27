#!/bin/bash
#SBATCH --job-name=vit5_hyena_dali
#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=13
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/vit5_hyena_dali_%j.out

set -eo pipefail

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting data extraction to local scratch..."
echo "=========================================================="

# 1. Setup Environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dknigge/code/nvSubquadratic-private
export PYTHONPATH=.
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)

# 2. Extract Data to Local Scratch
export IMAGENET_FOLDER_PATH=/local_scratch/$USER/imagenet_folder
if [ -d "$IMAGENET_FOLDER_PATH/train" ] && [ -d "$IMAGENET_FOLDER_PATH/val" ]; then
    echo "ImageNet already extracted to $IMAGENET_FOLDER_PATH, skipping extraction."
else
    echo "Extracting ImageNet to $IMAGENET_FOLDER_PATH"
    python scripts/extract_imagenet_to_folder.py
fi

echo "=========================================================="
echo "Extraction finished. Starting pretraining..."
echo "=========================================================="

# 3. Define Config and Run Parameter
CONFIG="examples/vit5_imagenet/vit5_small_pretrain_hyena_gap_apex_dali.py"

# Make nvcc available from the node-local CUDA install so torch.compile
# max-autotune can use both Triton and CUDA C++ backends.
export CUDA_HOME=/usr/local/cuda-13
export PATH=$CUDA_HOME/bin:$PATH

# Run PyTorch Lightning training.
# SLURM will automatically configure DDP since we requested 8 GPUs.
srun python experiments/run.py --config "$CONFIG"

echo "Job finished at $(date)"
