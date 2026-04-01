#!/bin/bash
# Download a dataset from The Well benchmark.
#
# Usage:
#   bash scripts/download_well.sh <dataset_name> [--base-path /path] [--split train|val|test]
#
# Environment variables:
#   WELL_DATA_PATH  Base directory for downloaded data (default: ./data/the_well)
#
# Examples:
#   bash scripts/download_well.sh active_matter
#   bash scripts/download_well.sh gray_scott_reaction_diffusion --split train
#   WELL_DATA_PATH=/shared/data bash scripts/download_well.sh MHD_64

set -euo pipefail

VALID_DATASETS=(
    active_matter
    acoustic_scattering_maze
    convective_envelope_rsg
    euler_multi_quadrants_periodicBC
    gray_scott_reaction_diffusion
    helmholtz_staircase
    MHD_64
    MHD_256
    planetswe
    post_neutron_star_merger
    rayleigh_benard
    rayleigh_taylor_instability
    shear_flow
    supernova_explosion_64
    turbulence_gravity_cooling
    turbulent_radiative_layer_2D
    turbulent_radiative_layer_3D
    viscoelastic_instability
)

usage() {
    echo "Usage: $0 <dataset_name> [--base-path /path] [--split train|val|test]"
    echo ""
    echo "Available datasets:"
    for ds in "${VALID_DATASETS[@]}"; do
        echo "  - $ds"
    done
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

DATASET_NAME="$1"
shift

# Default base path from env var or fallback
BASE_PATH="${WELL_DATA_PATH:-./data/the_well}"
SPLIT=""

# Parse optional arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --base-path)
            BASE_PATH="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Validate dataset name
VALID=false
for ds in "${VALID_DATASETS[@]}"; do
    if [ "$ds" = "$DATASET_NAME" ]; then
        VALID=true
        break
    fi
done

if [ "$VALID" = false ]; then
    echo "Error: Unknown dataset '$DATASET_NAME'"
    echo ""
    echo "Available datasets:"
    for ds in "${VALID_DATASETS[@]}"; do
        echo "  - $ds"
    done
    exit 1
fi

# Build download command
CMD="the-well-download --base-path $BASE_PATH --dataset $DATASET_NAME"
if [ -n "$SPLIT" ]; then
    CMD="$CMD --split $SPLIT"
fi

echo "Downloading '$DATASET_NAME' to '$BASE_PATH'..."
echo "Command: $CMD"
$CMD
echo "Done."
