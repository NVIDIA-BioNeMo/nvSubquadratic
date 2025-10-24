# TODO: Add license header here

"""End-to-end CP gradient test using optimizer state (John's approach).

This implements the approach from the October 17, 2025 discussion:
1. Set Adam betas=(0.0, 0.0) so optimizer stores raw gradients in exp_avg
2. Run one training step
3. Save checkpoint with optimizer state
4. Extract gradients from optimizer.state['exp_avg']
5. Compare CP=1 vs CP=2

Usage:
    python tests/e2e_gradient_from_optimizer_state.py
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_and_save_checkpoint(cp_size, checkpoint_dir, config, seed, batch_size):
    """Run training for one step and save checkpoint with optimizer state."""
    port = 29500 if cp_size == 1 else 29501

    cmd = [
        "torchrun",
        f"--nproc_per_node={cp_size}",
        f"--master_port={port}",
        "examples/run.py",
        "--config",
        config,
        "--experiment_dir",
        str(checkpoint_dir),
        "distributed.enabled=True",
        f"distributed.context_parallel_size={cp_size}",
        "dataset.enable_cp=True",
        f"dataset.batch_size={batch_size}",
        f"dataset.seed={seed}",  # Dataset seed
        f"seed={seed}",  # Global seed
        "deterministic=True",
        "benchmark=False",
        "train.iterations=1",
        # Skip validation and testing to save checkpoint immediately after training step
        "validate=False",
        "test=False",
        # Disable dropout
        "net.block_cfg.dropout_cfg.p=0.0",
        "net.dropout_in_cfg.p=0.0",
        # CRITICAL: Adam with betas=(0,0) stores raw gradients in exp_avg
        "optimizer.__target__=torch.optim.Adam",
        "optimizer.betas=(0.0,0.0)",
        "optimizer.weight_decay=0.0",
    ]

    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "disabled",
            "PYTHONHASHSEED": str(seed),
            "PL_GLOBAL_SEED": str(seed),  # PyTorch Lightning seed
            "PL_SEED_WORKERS": "1",  # Seed DataLoader workers
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # Deterministic CUDA ops
            "CUDA_VISIBLE_DEVICES": "0" if cp_size == 1 else "0,1",
            # "NCCL_P2P_DISABLE": "1", # NOTE some users need to disable P2P to avoid hanging
        }
    )

    logger.info(f"  Running CP={cp_size} and saving checkpoint...")
    logger.info(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)

    if result.returncode != 0:
        logger.error(f"  Training failed (exit code {result.returncode})")
        return False

    logger.info("  Training completed, looking for checkpoint...")

    # Find the checkpoint Lightning saved
    ckpt_files = list(Path(checkpoint_dir).glob("**/last.ckpt"))
    if not ckpt_files:
        ckpt_files = list(Path(checkpoint_dir).glob("**/epoch*.ckpt"))

    if ckpt_files:
        logger.info(f"  Found checkpoint: {ckpt_files[0]}")
        return True
    else:
        logger.error(f"  No checkpoint found in {checkpoint_dir}")
        return False


def extract_gradients_from_checkpoint(checkpoint_path):
    """Extract gradients from optimizer state.

    With Adam betas=(0,0), optimizer.state[param_id]['exp_avg'] = gradient
    """
    logger.info(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "optimizer_states" not in ckpt:
        raise ValueError("No optimizer_states in checkpoint")

    opt_state = ckpt["optimizer_states"][0]
    state_dict = ckpt["state_dict"]

    # Build mapping: param_id -> gradient (exp_avg)
    param_id_to_grad = {}
    for param_id, param_state in opt_state["state"].items():
        if "exp_avg" in param_state:
            param_id_to_grad[param_id] = param_state["exp_avg"]

    # Build mapping: param_id -> param_name (by order)
    param_ids = []
    for group in opt_state["param_groups"]:
        param_ids.extend(group["params"])

    param_names = []
    for name in state_dict.keys():
        # Skip non-trainable or DDP wrapper prefixes
        if not any(skip in name for skip in ["_forward_module", "module.module"]):
            param_names.append(name)

    # Map by order
    param_id_to_name = {}
    for param_id, name in zip(param_ids, param_names):
        param_id_to_name[param_id] = name

    # Extract gradient statistics
    gradients = {}
    for param_id, grad in param_id_to_grad.items():
        name = param_id_to_name.get(param_id, f"unknown_{param_id}")
        # Clean up DDP wrapper prefixes
        name = name.replace("_forward_module.", "").replace("module.", "")

        gradients[name] = {
            "grad": grad,
            "shape": list(grad.shape),
            "norm": grad.norm().item(),
            "mean": grad.mean().item(),
            "std": grad.std().item(),
            "min": grad.min().item(),
            "max": grad.max().item(),
        }

    logger.info(f"  Extracted {len(gradients)} parameter gradients from optimizer state")
    return gradients


def compare_gradients(cp1_grads, cp2_grads, tolerance):
    """Compare gradients from optimizer state."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("GRADIENT COMPARISON (from optimizer.state['exp_avg'])")
    logger.info("=" * 80)

    all_match = True
    for name in sorted(cp1_grads.keys()):
        if name not in cp2_grads:
            logger.error(f"MISSING: {name} in CP=2")
            all_match = False
            continue

        g1 = cp1_grads[name]["grad"]
        g2 = cp2_grads[name]["grad"]
        v1 = cp1_grads[name]["norm"]
        v2 = cp2_grads[name]["norm"]
        metric = ((g1 - g2).norm() / g1.norm()).item()
        match = metric < tolerance

        if not match:
            all_match = False

        status = "PASS" if match else "FAIL"
        short_name = name.replace("network.", "")
        logger.info(f"{status:4s} {short_name:50s}  {v1:10.4e} vs {v2:10.4e}  (metric: {metric:7.2%})")

    logger.info("=" * 80)
    logger.info("RESULT: SUCCESS - All gradients match" if all_match else "RESULT: FAILURE - Some gradients differ")
    logger.info("=" * 80)
    return all_match


def create_initial_checkpoint(checkpoint_path, config, seed, batch_size):
    """Create initial checkpoint by running 1 training step.

    This checkpoint will be used as the starting point for both CP=1 and CP=2 tests,
    ensuring identical model initialization.
    """
    logger.info("  Creating initial checkpoint (1 training step, single GPU)...")

    init_dir = checkpoint_path.parent / "init_run"

    cmd = [
        "python",
        "examples/run.py",
        "--config",
        config,
        "--experiment_dir",
        str(init_dir),
        f"seed={seed}",
        f"dataset.seed={seed}",
        f"dataset.batch_size={batch_size}",
        "train.iterations=1",
        "deterministic=True",
        "benchmark=False",
    ]

    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "disabled",
            "PL_GLOBAL_SEED": str(seed),
            "PL_SEED_WORKERS": "1",
            "PYTHONHASHSEED": str(seed),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        logger.error("  Failed to create initial checkpoint")
        if result.stderr:
            for line in result.stderr.split("\n")[-10:]:
                if line.strip():
                    logger.error(f"    {line}")
        return False

    # Find the saved checkpoint
    ckpt_files = list(init_dir.glob("**/last.ckpt"))
    if ckpt_files:
        shutil.copy(ckpt_files[0], checkpoint_path)
        logger.info(f"  Saved initial checkpoint: {checkpoint_path}")
        return True
    else:
        logger.error(f"  No checkpoint found in {init_dir}")
        return False


def run_from_checkpoint(cp_size, checkpoint_dir, init_checkpoint, config, seed, batch_size):
    """Run training from initial checkpoint for 1 more step.

    Initial checkpoint is at step 1, so we train to step 2.
    """
    port = 29500 if cp_size == 1 else 29501

    cmd = [
        "torchrun",
        f"--nproc_per_node={cp_size}",
        f"--master_port={port}",
        "examples/run.py",
        "--config",
        config,
        "--experiment_dir",
        str(checkpoint_dir),
        "--ckpt_path",
        str(init_checkpoint),  # Resume from initial checkpoint (step 1)
        "distributed.enabled=True",
        f"distributed.context_parallel_size={cp_size}",
        "dataset.enable_cp=True",
        f"dataset.batch_size={batch_size}",
        f"dataset.seed={seed}",
        f"seed={seed}",
        "deterministic=True",
        "benchmark=False",
        "train.iterations=1",  # Train to step 1 (one more step from checkpoint at step 1)
        # Skip validation/testing
        "validate=False",
        "test=False",
        # Disable dropout
        "net.block_cfg.dropout_cfg.p=0.0",
        "net.dropout_in_cfg.p=0.0",
        # CRITICAL: Adam with betas=(0,0) stores raw gradients in exp_avg
        "optimizer.__target__=torch.optim.Adam",
        "optimizer.betas=(0.0,0.0)",
        "optimizer.weight_decay=0.0",
    ]

    env = os.environ.copy()
    custom_env = {
        "WANDB_MODE": "disabled",
        "PYTHONHASHSEED": str(seed),
        "PL_GLOBAL_SEED": str(seed),
        "PL_SEED_WORKERS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "0" if cp_size == 1 else "0,1",
        # "NCCL_P2P_DISABLE": "1", # NOTE some users need to disable P2P to avoid hanging
    }
    env.update(custom_env)

    logger.info(f"  Running CP={cp_size} from initial checkpoint...")
    logger.info(f"  Environment: {custom_env}")
    logger.info(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)

    if result.returncode != 0:
        logger.error(f"  Training failed (exit code {result.returncode})")
        return False

    logger.info("  Training completed")
    return True


def main():
    """Main function to run E2E gradient test extracting from optimizer state."""
    parser = argparse.ArgumentParser(description="E2E gradient test from optimizer state")
    parser.add_argument(
        "--config",
        default="examples/mnist_classification/experiments/mnist_classification_ccnn_testing_minimal_dataset_hyena_rope_qknorm.py",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.01)  # 1% - should be very close
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    logger.info("")
    logger.info("=" * 80)
    logger.info("E2E GRADIENT TEST (from optimizer state)")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info("Method: Resume from same initial checkpoint → guarantees same initialization")
    logger.info("Optimizer: Adam with betas=(0.0, 0.0) → exp_avg stores raw gradients")
    logger.info("=" * 80)

    save_dir = Path(args.save_dir) if args.save_dir else Path(tempfile.mkdtemp(prefix="cp_opt_"))
    cleanup = not args.keep

    try:
        logger.info("\nSTEP 0: Create initial checkpoint (same starting point for both runs)")
        logger.info("-" * 80)
        init_checkpoint = save_dir / "init.ckpt"
        if not create_initial_checkpoint(init_checkpoint, args.config, args.seed, args.batch_size):
            logger.error("Failed to create initial checkpoint")
            return 1

        logger.info("\nSTEP 1: Run CP=1 from initial checkpoint")
        logger.info("-" * 80)
        cp1_dir = save_dir / "cp1"
        if not run_from_checkpoint(1, cp1_dir, init_checkpoint, args.config, args.seed, args.batch_size):
            return 1

        logger.info("\nSTEP 2: Run CP=2 from initial checkpoint")
        logger.info("-" * 80)
        cp2_dir = save_dir / "cp2"
        if not run_from_checkpoint(2, cp2_dir, init_checkpoint, args.config, args.seed, args.batch_size):
            return 1

        logger.info("\nSTEP 3: Extract gradients from final checkpoints")
        logger.info("-" * 80)

        # Find final checkpoints (after 1 training step)
        cp1_ckpt = next(iter(cp1_dir.glob("**/last.ckpt")))
        cp2_ckpt = next(iter(cp2_dir.glob("**/last.ckpt")))

        cp1_grads = extract_gradients_from_checkpoint(cp1_ckpt)
        cp2_grads = extract_gradients_from_checkpoint(cp2_ckpt)

        logger.info("\nSTEP 4: Compare gradients")
        logger.info("-" * 80)
        success = compare_gradients(cp1_grads, cp2_grads, args.tolerance)

        return 0 if success else 1

    finally:
        if cleanup:
            shutil.rmtree(save_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
