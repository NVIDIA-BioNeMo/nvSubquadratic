#!/usr/bin/env bash
# Launch two arms concurrently — one per A6000 GPU.
# Usage:
#   ./examples/hyena2d_parallel_adapter/run.sh pretrain     # Phase 1
#   ./examples/hyena2d_parallel_adapter/run.sh finetune <PRETRAIN_RUN_PATH>
#   ./examples/hyena2d_parallel_adapter/run.sh arm_f        # from scratch
#
# For finetune, PRETRAIN_RUN_PATH is the W&B run path of the Phase 1 checkpoint,
# e.g. "my_entity/nvsubquadratic/abc123".
#
# Run three seeds per arm by repeating with seed=1 and seed=2.

set -euo pipefail
PYTHON=${PYTHON:-python}
RUN="PYTHONPATH=. $PYTHON experiments/run.py"

case "${1:-}" in
pretrain)
    echo "=== Phase 1: pretrain (fixed placement) ==="
    CUDA_VISIBLE_DEVICES=0 $RUN \
        config_path=examples/hyena2d_parallel_adapter/pretrain.py seed=0 &
    wait
    echo "Done. Save the W&B run path and pass it to: $0 finetune <run_path>"
    ;;

finetune)
    CKPT="${2:?Usage: $0 finetune <PRETRAIN_RUN_PATH>}"
    echo "=== Phase 2: fine-tuning arms A-E (CKPT=$CKPT) ==="

    # Arms run pairwise — GPU 0 / GPU 1 concurrently.
    for SEED in 0 1 2; do
        echo "-- Seed $SEED --"
        # Pair 1: arm A (head only) vs arm C0 (capacity ablation)
        CUDA_VISIBLE_DEVICES=0 $RUN \
            config_path=examples/hyena2d_parallel_adapter/arm_a.py \
            start_from_checkpoint.run_path="$CKPT" seed="$SEED" &
        CUDA_VISIBLE_DEVICES=1 $RUN \
            config_path=examples/hyena2d_parallel_adapter/arm_c0.py \
            start_from_checkpoint.run_path="$CKPT" seed="$SEED" &
        wait

        # Pair 2: arm C1 (local conv) vs arm C2 (Hyena — main arm)
        CUDA_VISIBLE_DEVICES=0 $RUN \
            config_path=examples/hyena2d_parallel_adapter/arm_c1.py \
            start_from_checkpoint.run_path="$CKPT" seed="$SEED" &
        CUDA_VISIBLE_DEVICES=1 $RUN \
            config_path=examples/hyena2d_parallel_adapter/arm_c2.py \
            start_from_checkpoint.run_path="$CKPT" seed="$SEED" &
        wait

        # Pair 3: arm D (full fine-tune) vs arm E (full fine-tune + adapter)
        CUDA_VISIBLE_DEVICES=0 $RUN \
            config_path=examples/hyena2d_parallel_adapter/arm_d.py \
            start_from_checkpoint.run_path="$CKPT" seed="$SEED" &
        CUDA_VISIBLE_DEVICES=1 $RUN \
            config_path=examples/hyena2d_parallel_adapter/arm_e.py \
            start_from_checkpoint.run_path="$CKPT" seed="$SEED" &
        wait
    done
    ;;

arm_f)
    echo "=== Arm F: from-scratch interleaved hybrid ==="
    for SEED in 0 1 2; do
        CUDA_VISIBLE_DEVICES=0 $RUN \
            config_path=examples/hyena2d_parallel_adapter/arm_f.py seed="$SEED" &
        wait
    done
    ;;

*)
    echo "Usage: $0 {pretrain | finetune <PRETRAIN_RUN_PATH> | arm_f}"
    exit 1
    ;;
esac
