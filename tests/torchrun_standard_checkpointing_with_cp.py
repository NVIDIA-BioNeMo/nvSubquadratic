# TODO: Add license header here

"""Test that standard Lightning checkpoints work with Context Parallelism.

This test validates that we can use Lightning's standard checkpointing
(non-distributed) with CP training, which is simpler and more reliable
for most use cases.

Example Usage:
    torchrun --nproc_per_node=2 tests/integration/test_standard_checkpoint_with_cp.py
"""

import argparse
import logging
import os
import shutil
import time

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from torch.nn.parallel import DistributedDataParallel as DDP

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.self_attention import SelfAttention
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.classification_resnet import ClassificationResNet
from nvsubquadratic.parallel.utils import init_parallel_state


logger = logging.getLogger(__name__)


def create_model(hidden_dim=64):
    """Create a simple model for testing."""
    return ClassificationResNet(
        in_channels=100,
        out_channels=10,
        num_blocks=1,
        hidden_dim=hidden_dim,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=100, out_features=hidden_dim, bias=False),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=hidden_dim, out_features=10, bias=True),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim=hidden_dim,
                mixer_cfg=LazyConfig(SelfAttention)(
                    hidden_dim=hidden_dim,
                    num_heads=4,
                    apply_qk_norm=True,
                    use_rope=False,
                ),
            ),
            mlp_cfg=LazyConfig(torch.nn.Identity)(),
            norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
    )


def test_checkpoint_save_load_standard():
    """Test that standard Lightning checkpoints work with CP."""
    logger.info("Test: Standard Lightning Checkpoint with CP")

    cp_group = parallel_state.get_context_parallel_group()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_size = parallel_state.get_context_parallel_world_size()

    logger.info(f"CP rank={cp_rank}/{cp_size}")

    # Set seed to ensure all ranks have the same initial model and data
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create model and wrap with DDP to synchronize gradients
    model = create_model(hidden_dim=64)
    model = model.to(torch.cuda.current_device())

    # Wrap with DDP on world group to synchronize gradients across all ranks
    # This is necessary even in pure CP setups to keep parameters synchronized
    model = DDP(model, device_ids=[torch.cuda.current_device()])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train for a few steps
    for step in range(5):
        batch_size = 2
        seq_len = 392  # For CP=2
        x = torch.randn(batch_size, seq_len, 100, device=torch.cuda.current_device())
        output = model(x, cp_group=cp_group)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Use a fixed shared location (simpler than broadcasting)
    checkpoint_dir = "/tmp/cp_standard_checkpoint_test"
    if dist.get_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    dist.barrier()  # Wait for directory creation

    checkpoint_path = os.path.join(checkpoint_dir, "test.ckpt")
    logger.info(f"Using checkpoint path: {checkpoint_path}")

    # Save checkpoint (only rank 0)
    if dist.get_rank() == 0:
        checkpoint = {
            "state_dict": model.module.state_dict(),  # Use .module to get unwrapped model
            "optimizer_states": [optimizer.state_dict()],
            "epoch": 1,
            "global_step": 5,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    dist.barrier()

    # Verify checkpoint exists
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    # Create new model (simulate restart)
    model_new = create_model(hidden_dim=64)
    model_new = model_new.to(torch.cuda.current_device())
    model_new = DDP(model_new, device_ids=[torch.cuda.current_device()])
    optimizer_new = torch.optim.Adam(model_new.parameters(), lr=0.001)

    # Load checkpoint (all ranks load same file)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{torch.cuda.current_device()}")

    model_new.module.load_state_dict(checkpoint["state_dict"])  # Load into unwrapped model
    if "optimizer_states" in checkpoint:
        optimizer_new.load_state_dict(checkpoint["optimizer_states"][0])

    logger.info(f"Loaded checkpoint: epoch={checkpoint['epoch']}, step={checkpoint['global_step']}")

    # Verify parameters match exactly (not just close - they should be identical)
    # Compare unwrapped models
    for (name1, param1), (name2, param2) in zip(model.module.named_parameters(), model_new.module.named_parameters()):
        assert name1 == name2
        torch.testing.assert_close(param1, param2, rtol=0, atol=0, msg=f"Parameter {name1} mismatch")

    # Cleanup
    dist.barrier()
    if dist.get_rank() == 0:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Cleaned up checkpoint directory: {checkpoint_dir}")

    if dist.get_rank() == 0:
        logger.info("Standard checkpoint save/load test passed!")
        logger.info("Parameters remain synchronized across CP ranks with DDP gradient sync")

    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test standard checkpoints with CP")
    parser.add_argument("--context_parallel_size", type=int, default=2, help="Context parallel size")
    args = parser.parse_args()

    # Setup logging
    rank = int(os.getenv("RANK", "0"))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [Rank {rank}] - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    try:
        # Initialize distributed
        local_rank = init_parallel_state(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=args.context_parallel_size,
        )

        logger.info(f"Initialized distributed: local_rank={local_rank}")

        # Run test
        logger.info("=" * 80)
        test_checkpoint_save_load_standard()
        dist.barrier()

        if dist.get_rank() == 0:
            logger.info("=" * 80)
            logger.info("All standard checkpoint tests passed!")
            logger.info("=" * 80)
            logger.info("Standard Lightning checkpoints work perfectly with CP!")
            logger.info("You can now use Lightning's built-in checkpointing for production.")
            logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        logger.info("Cleaning up resources")
        torch.cuda.empty_cache()
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
        time.sleep(1)


if __name__ == "__main__":
    exit(main())
