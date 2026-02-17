#!/bin/bash
# Test script for LRA benchmarks

CONFIGS=(
    "examples/lra/hyena_image.py"
    "examples/lra/attention_image.py"
    "examples/lra/hyena_text.py"
    "examples/lra/attention_text.py"
)

export PYTHONPATH=.

for cfg in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Testing config: $cfg"
    echo "========================================"
    python experiments/run.py --config "$cfg" debug=True train.iterations=5
    if [ $? -ne 0 ]; then
        echo "Error: $cfg failed verification!"
        exit 1
    fi
done

echo "All LRA configs verified successfully!"
