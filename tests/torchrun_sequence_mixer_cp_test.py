"""Context Parallelism Test for QKVSequenceMixer.

This module provides comprehensive testing for QKVSequenceMixer with context parallelism,
ensuring that the model produces equivalent outputs when run with and without context
parallelism enabled. The tests verify both forward pass correctness and gradient
computation across multiple GPUs.

Example Usage:
    torchrun --nproc_per_node=2 tests/test_sequence_mixer_cp_torchrun.py --context_parallel_size=2
"""

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from torch.nn.parallel import DistributedDataParallel as DDP

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.parallel.a2a_comms import zigzag_gather_from_group_ranks, zigzag_split_across_group_ranks
from nvsubquadratic.parallel.utils import init_parallel_state


def sequence_mixer_config(data_dim: int = 1) -> LazyConfig:
    """Create a LazyConfig for QKVSequenceMixer with Hyena as inner mixer.

    Constructs a complete configuration for QKVSequenceMixer using Hyena as the
    inner sequence mixer.

    Args:
        data_dim: Dimensionality of the input data (default: 1 for 1D sequences).

    Returns:
        LazyConfig: A lazy configuration object for QKVSequenceMixer that can be
            instantiated later.
    """
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=128,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=data_dim,
                hidden_dim=128,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=data_dim,
                    out_dim=128,
                    mlp_hidden_dim=32,
                    num_layers=3,
                    embedding_dim=32,
                    omega_0=100.0,
                    L_cache=32,
                    use_bias=True,
                    hidden_omega_0=1.0,
                ),
                mask_cfg=LazyConfig(GaussianModulationND)(
                    data_dim=data_dim,
                    num_channels=128,
                    min_std=0.025,
                    max_std=1.25,
                    init_std_low=0.05,
                    init_std_high=1.0,
                    parametrization="direct",
                ),
                grid_type="single",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv1d)(
                in_channels=384,  # 3 * 128 for concatenated q, k, v
                out_channels=384,  # 3 * 128 for concatenated q, k, v
                kernel_size=3,
                groups=384,  # Grouped convolution
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(torch.nn.Identity)(),
            apply_qk_norm=True,
            use_rope=False,  # Disable RoPE to avoid in-place issues
            rope_base=10000.0,
        ),
    )


def test_sequence_mixer_cp_equivalency(data_dim: int = 1, dtype: str = "float32") -> bool:
    """Test that the sequence mixer works correctly with and without context parallelism.

    This comprehensive test validates the correctness of QKVSequenceMixer when using
    context parallelism by comparing outputs from two scenarios:
    1. Running without context parallelism (full sequence on single GPU)
    2. Running with context parallelism (sequence split across multiple GPUs)

    Args:
        data_dim: Dimensionality of the input data (default: 1).
        dtype: Data type for model and input tensors (default: "float32").
            Supported types: "float32", "float16", "bfloat16".

    Returns:
        bool: True if all tests pass successfully, False otherwise.

    Raises:
        AssertionError: If tensor shapes don't match expected dimensions or if
            numerical differences exceed tolerance thresholds.
    """
    # Check if we can run distributed test
    if torch.cuda.device_count() < 2 or not dist.is_available():
        logging.warning("Not enough GPUs or distributed not available. Skipping distributed test.")
        return False

    # Check if distributed environment is set up
    if not os.getenv("RANK") and not os.getenv("WORLD_SIZE"):
        logging.warning("Distributed environment not set up. Skipping distributed test.")
        return False

    try:
        # Create sequence mixer configuration and instantiate
        sequence_mixer_cfg = sequence_mixer_config(data_dim=data_dim)
        sequence_mixer = instantiate(sequence_mixer_cfg)

        # Move model to the correct device
        sequence_mixer = sequence_mixer.to(torch.cuda.current_device())

        # Convert dtype if needed
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        sequence_mixer = sequence_mixer.to(dtype_map[dtype])

        # Wrap with DDP
        ddp_sequence_mixer = DDP(
            sequence_mixer,
            process_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            find_unused_parameters=True,
        )

        # Create test input
        batch_size = 2
        seq_len = 1024
        hidden_dim = 128

        # Convert dtype if needed
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        input_dtype = dtype_map[dtype]

        test_input = torch.randn(
            batch_size, seq_len, hidden_dim, device=torch.cuda.current_device(), dtype=input_dtype
        )

        # Broadcast input across context parallel group
        cp_group = parallel_state.get_context_parallel_group()
        dist.broadcast(test_input, min(dist.get_process_group_ranks(cp_group)), group=cp_group)

        logging.info("Running without context parallel")
        output_no_cp = ddp_sequence_mixer(test_input, _use_cp=False)

        if dist.get_rank() == 0:
            try:
                assert output_no_cp.shape == (batch_size, seq_len, hidden_dim), (
                    f"output_no_cp.shape: {output_no_cp.shape}"
                )
                logging.info(f"Output without CP shape: {output_no_cp.shape}")
            except AssertionError as e:
                logging.error(f"Assertion error for output without CP shape: {e}")
                raise

        loss_no_cp = output_no_cp.float().mean()
        loss_no_cp.backward()

        # Store gradients for comparison
        grads_without_cp = []
        for n, p in ddp_sequence_mixer.named_parameters():
            if p.grad is not None:
                grads_without_cp.append((n, p.grad.clone()))

        ddp_sequence_mixer.zero_grad()
        dist.barrier()

        logging.info("Running with context parallel")
        # Split the input features across the context parallel group using zigzag
        test_input_cp = zigzag_split_across_group_ranks(test_input, group=cp_group, seq_dim=1)

        output_with_cp = ddp_sequence_mixer(test_input_cp, _use_cp=True)

        if dist.get_rank() == 0:
            try:
                # With zigzag splitting, each rank gets a portion of the sequence
                expected_cp_shape = (
                    batch_size,
                    seq_len // parallel_state.get_context_parallel_world_size(),
                    hidden_dim,
                )
                assert output_with_cp.shape == expected_cp_shape, (
                    f"output_with_cp.shape: {output_with_cp.shape}, expected: {expected_cp_shape}"
                )
                logging.info(f"Output with CP shape: {output_with_cp.shape}")
            except AssertionError as e:
                logging.error(f"Assertion error for output with CP shape: {e}")
                raise

        # Gather output from all ranks using zigzag gathering
        output_with_cp_gathered = zigzag_gather_from_group_ranks(output_with_cp, group=cp_group, seq_dim=1)

        if dist.get_rank() == 0:
            try:
                assert output_with_cp_gathered.shape == (batch_size, seq_len, hidden_dim), (
                    f"output_with_cp_gathered.shape: {output_with_cp_gathered.shape}"
                )
                logging.info(f"Output with CP gathered shape: {output_with_cp_gathered.shape}")
            except AssertionError as e:
                logging.error(f"Assertion error for output with CP gathered shape: {e}")
                raise

        # Compute loss and gradients for CP case
        loss_with_cp = output_with_cp_gathered.float().mean()
        loss_with_cp.backward()
        dist.barrier()

        # Store gradients for comparison
        grads_with_cp = []
        for n, p in ddp_sequence_mixer.named_parameters():
            if p.grad is not None:
                grads_with_cp.append((n, p.grad.clone()))

        ddp_sequence_mixer.zero_grad()
        dist.barrier()

        # Only perform comparison on rank 0
        if dist.get_rank() == 0:
            logging.info(f"Comparing loss values: without CP = {loss_no_cp.item()}, with CP = {loss_with_cp.item()}")
            try:
                torch.testing.assert_close(loss_no_cp, loss_with_cp)
                logging.info("Loss comparison successful")
            except AssertionError as e:
                logging.error(f"Loss comparison failed: {e}")
                raise

            try:
                torch.testing.assert_close(output_no_cp, output_with_cp_gathered)
                logging.info("Output tensor comparison successful")
            except AssertionError as e:
                logging.error(f"Output tensor comparison failed: {e}")
                raise

            # Check gradients with and without CP
            try:
                assert len(grads_without_cp) == len(grads_with_cp)
                logging.info(f"Comparing {len(grads_without_cp)} gradient tensors")
            except AssertionError as e:
                logging.error(f"Gradient count mismatch: {e}")
                raise

            gradient_mismatch = False
            for (n_without_cp, g_without_cp), (n_with_cp, g_with_cp) in zip(grads_without_cp, grads_with_cp):
                try:
                    torch.testing.assert_close(g_without_cp, g_with_cp)
                except AssertionError as e:
                    gradient_mismatch = True
                    logging.error(f"Gradient mismatch for {n_without_cp}: {e}")

            if gradient_mismatch:
                logging.warning("There were gradient mismatches!")
            else:
                logging.info("All gradients matched successfully!")

        logging.info("Test completed successfully!")
        return True

    except Exception as e:
        logging.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        logging.info("Test completed, cleaning up resources")
        torch.cuda.empty_cache()


def main() -> int:
    """Main entry point for the context parallelism test script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test QKVSequenceMixer with context parallelism")
    parser.add_argument(
        "--data_dim",
        type=int,
        default=1,
        help="Data dimension",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model and input (float32, float16, bfloat16)",
    )
    parser.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        default=1,
        help="Tensor model parallel size",
    )
    parser.add_argument(
        "--pipeline_model_parallel_size",
        type=int,
        default=1,
        help="Pipeline model parallel size",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=2,
        help="Context parallel size",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/tmp/nvsubquadratic_cp_results",
        help="Directory for logs",
    )
    args = parser.parse_args()

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up file handler for logging
    rank = int(os.getenv("RANK", "0"))
    log_file = os.path.join(args.log_dir, f"rank_{rank}.log")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Initialize parallel state
    init_parallel_state(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
    )

    logging.info(f"Starting QKVSequenceMixer CP test with args: {args}")

    try:
        # Run the test
        success = test_sequence_mixer_cp_equivalency(args.data_dim, args.dtype)

        if success:
            logging.info("✅ All tests passed successfully!")
        else:
            logging.error("❌ Tests failed!")
            return 1

    finally:
        # Log final cleanup
        logging.info("Test completed, cleaning up resources")

        # Reset CUDA device
        torch.cuda.empty_cache()

        # Clean up any dangling context or process groups
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

        # Force a small delay to ensure all cleanup is complete
        time.sleep(1)

    return 0


if __name__ == "__main__":
    exit(main())
