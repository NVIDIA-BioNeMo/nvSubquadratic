#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:32
#SBATCH --partition=gpu_h100
#SBATCH --chdir="/home/dwessels2/code/nvSubquadratic-private"
#SBATCH --output=/home/dwessels2/code/nvSubquadratic-private/outputs/%A.%a/output.txt
#SBATCH --error=/home/dwessels2/code/nvSubquadratic-private/outputs/%A.%a/error.txt
#SBATCH --job-name='nvsubq'

# Set wd, activate conda environment.
source activate nvsubq

# print assigned node
echo "This job is running on node: $SLURM_NODELIST"
export HF_TOKEN=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

export PYTHONPATH="."
# python -m experiments.run --config examples/imagenet_classification/ccnn_7_512_hyena.py
python -m experiments.run --config examples/imagenet_classification/ccnn_7_512_hyena_circular.py
