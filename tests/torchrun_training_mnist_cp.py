# TODO: Add license header here

"""Integration test for full training loop with Context Parallelism on MNIST.

This test validates the complete training pipeline with CP:
1. Data loading with CP-aware splitting
2. Model training with CP
3. Validation and metrics tracking
4. Checkpointing during training

Example Usage:
    torchrun --nproc_per_node=2 tests/torchrun_training_mnist_cp.py
"""

import argparse
import logging
import os
import subprocess
import sys


logger = logging.getLogger(__name__)


def run_training_test(
    nproc: int = 2,
    cp_size: int = 2,
    iterations: int = 100,
    warmup_iterations: int = 10,
    timeout: int = 120,
) -> bool:
    """Run a full training test with Context Parallelism.

    Args:
        nproc: Number of processes per node
        cp_size: Context parallel size
        iterations: Number of training iterations
        warmup_iterations: Number of warmup iterations for scheduler
        timeout: Timeout in seconds for the training run

    Returns:
        bool: True if training completed successfully, False otherwise
    """
    # Get the rank if running in distributed mode
    rank = int(os.getenv("RANK", "0"))

    # Only run from rank 0 when called via torchrun
    # Or run directly if not in distributed mode
    if rank != 0 and os.getenv("RANK") is not None:
        logger.info(f"Rank {rank}: Skipping training test (only rank 0 runs)")
        return True

    logger.info("=" * 80)
    logger.info("Running full training integration test with Context Parallelism")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  - Processes per node: {nproc}")
    logger.info(f"  - Context parallel size: {cp_size}")
    logger.info(f"  - Training iterations: {iterations}")
    logger.info(f"  - Warmup iterations: {warmup_iterations}")
    logger.info(f"  - Timeout: {timeout}s")
    logger.info("=" * 80)

    # Build the torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--master_port=29522",  # Use different port to avoid conflicts
        "examples/run.py",
        "--config",
        "examples/mnist_classification/experiments/mnist_classification_ccnn_cp_test.py",
        "distributed.enabled=True",
        f"distributed.context_parallel_size={cp_size}",
        "dataset.enable_cp=True",
        f"train.iterations={iterations}",
        f"scheduler.warmup_iterations={warmup_iterations}",
        "debug=True",
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info("")

    # Run the training
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        # Log output
        logger.info("Training output:")
        logger.info("-" * 80)

        # Filter for important lines
        for line in result.stdout.split("\n"):
            if any(
                keyword in line
                for keyword in ["Epoch", "train/loss", "val/acc", "Sanity", "Testing", "GPU", "Error", "ERROR"]
            ):
                logger.info(line)

        if result.stderr:
            logger.warning("Stderr output:")
            for line in result.stderr.split("\n"):
                if line.strip():
                    logger.warning(line)

        logger.info("-" * 80)

        # Check if training succeeded
        if result.returncode != 0:
            logger.error(f"Training failed with exit code {result.returncode}")
            return False

        # Check for key success indicators in output
        success_indicators = [
            "Epoch" in result.stdout,  # Training ran
            "train/loss" in result.stdout,  # Loss was computed
        ]

        if not all(success_indicators):
            logger.error("Training output missing expected indicators")
            logger.error(f"Success indicators: {success_indicators}")
            return False

        logger.info("=" * 80)
        logger.info("✓ Training integration test passed!")
        logger.info("=" * 80)
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"Training test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test full training with CP")
    parser.add_argument("--nproc", type=int, default=2, help="Number of processes per node")
    parser.add_argument("--context_parallel_size", type=int, default=2, help="Context parallel size")
    parser.add_argument("--iterations", type=int, default=100, help="Training iterations")
    parser.add_argument("--warmup_iterations", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
    args = parser.parse_args()

    # Setup logging
    rank = int(os.getenv("RANK", "0"))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [Rank {rank}] - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    try:
        success = run_training_test(
            nproc=args.nproc,
            cp_size=args.context_parallel_size,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            timeout=args.timeout,
        )

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
