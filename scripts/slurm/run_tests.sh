#!/bin/bash
set -eo pipefail

set -a
source /home/dwromero/projects/nvSubquadratic-private/.env
set +a

source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq

cd /home/dwromero/projects/nvSubquadratic-private

PYTHONPATH=. pytest tests/ -v \
  --ignore=tests/test_nightly_validation.py \
  --ignore=tests/test_nightly_well_validation.py \
  2>&1
