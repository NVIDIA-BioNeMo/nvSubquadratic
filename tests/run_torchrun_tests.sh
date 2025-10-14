#!/bin/bash
# Script to run all torchrun tests and fail on errors

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

NPROC=${1:-2}
CP_SIZE=${2:-2}

echo "Running torchrun tests with $NPROC processes, CP size=$CP_SIZE"
echo ""

# Track overall success
FAILED=0

# Test 1: Standard checkpointing
echo "Test 1/3: Standard checkpointing with CP..."
if ! torchrun --nproc_per_node=$NPROC tests/torchrun_standard_checkpointing_with_cp.py --context_parallel_size=$CP_SIZE; then
    echo "ERROR: Standard checkpointing test failed!"
    FAILED=1
fi
sleep 1

# Test 2: Sequence mixer CP
echo "Test 2/3: Sequence mixer CP equivalency..."
if ! torchrun --nproc_per_node=$NPROC tests/torchrun_sequence_mixer_cp_test.py --context_parallel_size=$CP_SIZE; then
    echo "ERROR: Sequence mixer CP test failed!"
    FAILED=1
fi
sleep 1

# Test 3: MNIST CP (2D)
echo "Test 3/3: MNIST CP integration..."
if ! torchrun --nproc_per_node=$NPROC tests/torchrun_megatron_cp_mnist.py --context_parallel_size=$CP_SIZE; then
    echo "ERROR: MNIST CP test failed!"
    FAILED=1
fi

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All distributed tests passed!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi
