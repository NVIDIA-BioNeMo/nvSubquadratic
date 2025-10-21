#!/usr/bin/env python3
# TODO: Add license header here

"""End-to-end CP gradient equivalence test with real data loader.

This test implements John's recommendation from the October 17, 2025 discussion:
- Use actual training pipeline (run.py)
- Use real MNIST DataLoader with DistributedSampler
- Run one training step
- Compare gradients between CP=1 and CP=2

The test ensures both runs get the same batch by using DP=1 for both,
which makes DistributedSampler use (num_replicas=1, rank=0) in both cases.

Usage:
    python tests/test_e2e_gradient_with_dataloader.py
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_training(cp_size, save_dir, config, seed, batch_size):
    """Run training for one step and log gradients.

    Key insight: By limiting the dataset to exactly batch_size samples
    and disabling shuffle, both CP=1 and CP=2 will process the exact
    same samples in the exact same order!
    """
    port = 29500 if cp_size == 1 else 29501
    test_experiment_dir = f"/tmp/test_cp{cp_size}_experiment"

    cmd = [
        "torchrun",
        f"--nproc_per_node={cp_size}",
        f"--master_port={port}",
        "examples/run.py",
        "--config",
        config,
        "--log_gradients",
        str(save_dir),
        "--gradient_log_steps",
        "1",
        "--experiment_dir",
        test_experiment_dir,
        "distributed.enabled=True",
        f"distributed.context_parallel_size={cp_size}",
        "dataset.enable_cp=True",
        f"dataset.batch_size={batch_size}",
        f"dataset.seed={seed}",
        f"seed={seed}",
        "deterministic=True",
        "benchmark=False",
        "train.iterations=1",
        # CRITICAL: Disable dropout for deterministic gradients
        "net.block_cfg.dropout_cfg.p=0.0",
        "net.dropout_in_cfg.p=0.0",
        # CRITICAL: Limit dataset to exactly batch_size samples (only 1 batch total!)
        # This ensures both CP=1 and CP=2 process the exact same samples in same order
    ]

    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "disabled",
            "PYTHONHASHSEED": str(seed),
            "CUDA_VISIBLE_DEVICES": "0" if cp_size == 1 else "0,1",
        }
    )

    logger.info(f"  Running with CP={cp_size}...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)

    if result.returncode != 0:
        logger.error(f"FAILED (exit code {result.returncode})")
        return False

    logger.info("Completed successfully")
    return True


def load_gradients(save_dir, rank, step):
    """Load gradients from file."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from nvsubquadratic.testing import load_gradient_stats

    return load_gradient_stats(Path(save_dir), rank=rank, step=step)


def compare(cp1_grads, cp2_grads, tolerance):
    """Compare gradients."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("GRADIENT COMPARISON")
    logger.info("=" * 80)

    all_match = True
    for name in sorted(cp1_grads.keys()):
        if name not in cp2_grads:
            logger.error(f"MISSING: {name} in CP=2")
            all_match = False
            continue

        v1, v2 = cp1_grads[name]["norm"], cp2_grads[name]["norm"]
        diff = abs(v1 - v2) / abs(v1) if abs(v1) > 1e-10 else abs(v1 - v2)
        match = diff < tolerance

        if not match:
            all_match = False

        status = "PASS" if match else "FAIL"
        short_name = name.replace("network.", "")
        logger.info(f"{status:4s} {short_name:50s}  {v1:10.4e} vs {v2:10.4e}  (diff: {diff:7.2%})")

    logger.info("=" * 80)
    logger.info("RESULT: SUCCESS - All gradients match" if all_match else "RESULT: FAILURE - Some gradients differ")
    logger.info("=" * 80)
    return all_match


def main():
    """Main function to run the end-to-end CP gradient equivalence test."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/mnist_classification/experiments/mnist_classification_ccnn_testing_minimal_dataset_hyena_rope_qknorm.py",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=5e-3)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    logger.info("")
    logger.info("=" * 80)
    logger.info("END-TO-END CP GRADIENT TEST (with Minimal Dataset)")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {args.seed}, Batch size: {args.batch_size}, Tolerance: {args.tolerance:.1%}")
    logger.info("=" * 80)

    save_dir = Path(args.save_dir) if args.save_dir else Path(tempfile.mkdtemp(prefix="cp_grad_"))
    cleanup = not args.keep

    try:
        logger.info("\nSTEP 1: CP=1 (1 GPU, DP=1, CP=1)")
        logger.info("-" * 80)
        if not run_training(1, save_dir / "cp1", args.config, args.seed, args.batch_size):
            return 1

        logger.info("\nSTEP 2: CP=2 (2 GPUs, DP=1, CP=2)")
        logger.info("-" * 80)
        if not run_training(2, save_dir / "cp2", args.config, args.seed, args.batch_size):
            return 1

        logger.info("\nSTEP 3: Loading gradients")
        logger.info("-" * 80)
        cp1_grads = load_gradients(save_dir / "cp1", rank=0, step=1)
        logger.info(f"Loaded {len(cp1_grads)} parameters from CP=1")

        cp2_grads = load_gradients(save_dir / "cp2", rank=0, step=1)
        logger.info(f"Loaded {len(cp2_grads)} parameters from CP=2")

        logger.info("\nSTEP 4: Comparing")
        logger.info("-" * 80)
        success = compare(cp1_grads, cp2_grads, args.tolerance)

        return 0 if success else 1
    finally:
        if cleanup:
            shutil.rmtree(save_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
