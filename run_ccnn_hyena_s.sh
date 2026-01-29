#!/bin/bash

#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --gpus=1
#SBATCH --job-name=ccnn_hyena_s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
#SBATCH --output=slurm-out/ccnn_hyena_s_%A.out

source /home/dwessel/miniforge3/etc/profile.d/mamba.sh
mamba activate nvsubq

# Ensure we are in the repo root
cd /home/dwessel/code/nvSubquadratic-private

# Run the experiment using the direct path to the python environment
export PYTHONPATH=.
python experiments/run.py --config examples/spatial_recall_2d/emnist_regression_color_conditioning/ccnn_hyena_s.py 
