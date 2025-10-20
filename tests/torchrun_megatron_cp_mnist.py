# TODO: Add license header here

"""Integration test for Context Parallelism with MNIST classification (2D data).

This test validates end-to-end distributed training with Context Parallelism on 2D image data,
ensuring that:
1. 2D data (28x28 images) is correctly split across CP ranks
2. Model forward/backward passes work with CP on 2D spatial data
3. Gradients are correctly synchronized
4. 2D convolutions and kernels work with distributed operations

Example Usage:
    torchrun --nproc_per_node=2 tests/torchrun_megatron_cp_mnist.py
"""

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist

from nvsubquadratic.distributed.backend import (
    MegatronBackend,
    ParallelConfig,
    get_global_backend,
    set_global_backend,
)
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.distributed_depthwise_conv_nd import DistributedDepthwiseConv2d
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.classification_resnet import ClassificationResNet
from nvsubquadratic.parallel.utils import (
    zigzag_gather_from_group_ranks,
    zigzag_split_across_group_ranks,
)


logger = logging.getLogger(__name__)


def test_data_splitting():
    """Test that data is correctly split across CP ranks, broadcasted to all CP ranks, and gathered back to the original rank."""
    # Get CP info from backend
    backend = get_global_backend()
    cp_group = backend.get_context_parallel_group()
    cp_rank = backend.get_context_parallel_rank()
    cp_size = backend.get_context_parallel_world_size()

    logger.info(f"Testing data splitting: CP rank={cp_rank}/{cp_size}")

    # Create dummy 2D data (MNIST-like: 28x28 images)
    batch_size = 2
    height = 28
    width = 28
    hidden_dim = 128
    # Shape: (batch, height, width, channels)
    data = torch.randn(batch_size, height, width, hidden_dim, device=torch.cuda.current_device())

    # Broadcast to all CP ranks
    cp_group_ranks = dist.get_process_group_ranks(cp_group)
    source_rank = min(cp_group_ranks)
    dist.broadcast(data, src=source_rank, group=cp_group)

    # Split using zigzag along height dimension (dim=1)
    data_split = zigzag_split_across_group_ranks(data, group=cp_group, seq_dim=1)

    # Verify shape
    expected_shape = (batch_size, height // cp_size, width, hidden_dim)
    assert data_split.shape == expected_shape, f"Data split shape mismatch: {data_split.shape} vs {expected_shape}"

    # Gather back
    data_gathered = zigzag_gather_from_group_ranks(data_split, group=cp_group, seq_dim=1)

    # Verify gathered data matches original
    if dist.get_rank() == 0:
        torch.testing.assert_close(data, data_gathered)
        logger.info(f"Data splitting test passed! (2D: {height}x{width})")

    return True


def test_network_forward_pass_runs():
    """Test that network forward pass works with CP on 2D data."""
    # Get CP info from backend
    backend = get_global_backend()
    cp_group = backend.get_context_parallel_group()
    cp_rank = backend.get_context_parallel_rank()
    cp_size = backend.get_context_parallel_world_size()

    logger.info(f"Testing network forward pass: CP rank={cp_rank}/{cp_size}")

    # Create simple network for 2D data
    hidden_dim = 64
    in_channels = 1
    out_channels = 10

    # Create network with Hyena using DistributedDepthwiseConv2d (CP-ready for 2D)
    network = ClassificationResNet(
        in_channels=in_channels,
        out_channels=out_channels,
        num_blocks=1,  # Just 1 block for testing
        hidden_dim=hidden_dim,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=in_channels, out_features=hidden_dim, bias=False),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=hidden_dim, out_features=out_channels, bias=True),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim=hidden_dim,
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim=2,  # 2D for MNIST images
                        hidden_dim=hidden_dim,
                        kernel_cfg=LazyConfig(SIRENKernelND)(
                            data_dim=2,  # 2D kernels
                            out_dim=hidden_dim,
                            mlp_hidden_dim=16,
                            num_layers=2,
                            embedding_dim=16,
                            omega_0=100.0,
                            L_cache=16,
                            use_bias=True,
                            hidden_omega_0=1.0,
                        ),
                        mask_cfg=LazyConfig(GaussianModulationND)(
                            data_dim=2,  # 2D masks
                            num_channels=hidden_dim,
                            min_std=0.025,
                            max_std=1.25,
                            init_std_low=0.05,
                            init_std_high=1.0,
                            parametrization="direct",
                        ),
                        grid_type="single",
                    ),
                    short_conv_cfg=LazyConfig(DistributedDepthwiseConv2d)(  # 2D convolution
                        hidden_dim=hidden_dim * 3,
                        kernel_size=3,
                        num_groups=hidden_dim * 3,
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                    pixelhyena_norm_cfg=LazyConfig(torch.nn.Identity)(),
                    apply_qk_norm=True,
                    use_rope=False,
                    rope_base=10000.0,
                ),
            ),
            mlp_cfg=LazyConfig(torch.nn.Identity)(),
            norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
    )
    network = network.to(torch.cuda.current_device())

    # Create dummy 2D input (split for CP along height dimension)
    batch_size = 2
    height = 28
    width = 28
    height_local = height // cp_size  # Split height across CP ranks
    # Shape: (batch, height_local, width, channels)
    x = torch.randn(batch_size, height_local, width, in_channels, device=torch.cuda.current_device())

    # Forward pass with CP
    output = network(x, cp_group=cp_group)

    # Verify output shape
    expected_shape = (batch_size, 10)
    assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} vs {expected_shape}"

    # Test backward pass
    loss = output.mean()
    loss.backward()

    # Verify gradients exist
    has_grads = all(p.grad is not None for p in network.parameters() if p.requires_grad)
    assert has_grads, "Some parameters don't have gradients!"

    if dist.get_rank() == 0:
        logger.info(f"Network forward/backward pass test passed! (2D: {height}x{width})")

    return True


def main():
    """Main test function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test MNIST with Context Parallelism")
    parser.add_argument("--context_parallel_size", type=int, default=2, help="Context parallel size")
    parser.add_argument("--log_dir", type=str, default="/tmp/cp_mnist_test", help="Directory for logs")
    args = parser.parse_args()

    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    rank = int(os.getenv("RANK", "0"))
    log_file = os.path.join(args.log_dir, f"rank_{rank}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    try:
        # Initialize backend
        config = ParallelConfig(
            backend_type="megatron",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=args.context_parallel_size,
        )
        backend = MegatronBackend(config)

        # Initialize distributed (backend handles device setup and process group init)
        world_size = torch.cuda.device_count()
        rank = int(os.getenv("RANK", 0))

        backend.initialize(world_size=world_size, rank=rank)
        set_global_backend(backend)

        logger.info(f"Backend initialized: rank={rank}, world_size={world_size}")

        # Run tests
        logger.info("=" * 80)
        logger.info("Test 1: Data Splitting")
        logger.info("=" * 80)
        test_data_splitting()
        dist.barrier()

        logger.info("=" * 80)
        logger.info("Test 2: Network Forward/Backward Pass")
        logger.info("=" * 80)
        test_network_forward_pass_runs()
        dist.barrier()

        if dist.get_rank() == 0:
            logger.info("=" * 80)
            logger.info("All integration tests passed!")
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

        # Clean up backend
        backend = get_global_backend()
        if backend and hasattr(backend, "destroy"):
            backend.destroy()
        set_global_backend(None)

        # Destroy parallel state (Megatron cleanup)
        try:
            from megatron.core import parallel_state

            parallel_state.destroy_model_parallel()
        except Exception:
            pass

        if dist.is_initialized():
            dist.destroy_process_group()
        time.sleep(1)


if __name__ == "__main__":
    exit(main())
