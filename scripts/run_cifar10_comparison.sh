#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "=== [1/2] Isotropic Hyena — CIFAR-10 25 epochs ==="
PYTHONPATH=. python experiments/run.py \
    --config examples/patch_merging/cifar10/isotropic_hyena.py

echo "=== [2/2] Hierarchical Hyena — CIFAR-10 25 epochs ==="
PYTHONPATH=. python experiments/run.py \
    --config examples/patch_merging/cifar10/hierarchical_hyena.py

echo "=== Both runs complete ==="
