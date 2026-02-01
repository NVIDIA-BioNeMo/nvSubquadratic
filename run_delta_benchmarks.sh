#!/bin/bash

# Configuration files to run
CONFIGS=(
    "examples/spatial_recall_2d/emnist_regression_color_conditioning/ccnn_delta_hyena_xs.py"
    "examples/spatial_recall_2d/emnist_regression_color_conditioning/ccnn_delta_hyena_s.py"
    "examples/spatial_recall_2d/emnist_regression_color_conditioning/ccnn_delta_hyena_m.py"
    "examples/spatial_recall_2d/emnist_regression_color_conditioning/ccnn_delta_hyena_patchify_xs.py"
    "examples/spatial_recall_2d/emnist_regression_color_conditioning/ccnn_delta_hyena_patchify_s.py"
    "examples/spatial_recall_2d/emnist_regression_color_conditioning/ccnn_delta_hyena_patchify_m.py"
)

for CONFIG in "${CONFIGS[@]}"; do
    NAME=$(basename "$CONFIG" .py)
    echo "Submitting $NAME..."
    
    # Create a temporary slurm script for each run
    SBATCH_SCRIPT="slurm/submit_${NAME}.sh"
    
    cat <<EOT > "$SBATCH_SCRIPT"
#!/bin/bash
#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --gpus=1
#SBATCH --job-name=${NAME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
#SBATCH --output=slurm/${NAME}_%j.out

source /home/dwessel/miniforge3/etc/profile.d/mamba.sh
# mamba activate nvsubq (avoiding this as it often fails in sbatch)

export PATH=/usr/local/cuda-12.9/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:\$LD_LIBRARY_PATH
export PYTHONPATH=.

/home/dwessel/miniforge3/envs/nvsubq/bin/python experiments/run.py --config "$CONFIG" train.iterations=50000 dataset.base_datamodule_cfg.num_workers=10
EOT

    sbatch "$SBATCH_SCRIPT"
done
