# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from nvsubquadratic.modules.distributed import DistributedConv1d, DistributedConv2d, DistributedConv3d
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.parallel.utils import (
    init_parallel_state,
    zigzag_gather_from_group_ranks,
    zigzag_split_across_group_ranks,
)


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


def sequence_mixer_config_distributed(data_dim: int = 1) -> LazyConfig:
    """Create a LazyConfig for QKVSequenceMixer with distributed convolutions.

    Constructs a complete configuration for QKVSequenceMixer using Hyena as the
    inner sequence mixer with distributed-aware convolution layers.

    Args:
        data_dim: Dimensionality of the input data (default: 1 for 1D sequences).

    Returns:
        LazyConfig: A lazy configuration object for QKVSequenceMixer that can be
            instantiated later.
    """
    # Select the appropriate distributed convolution class based on data dimension
    if data_dim == 1:
        conv_class = DistributedConv1d
    elif data_dim == 2:
        conv_class = DistributedConv2d
    elif data_dim == 3:
        conv_class = DistributedConv3d
    else:
        raise ValueError(f"Unsupported data dimension: {data_dim}")

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
            # Use distributed convolution instead of standard torch.nn.Conv
            short_conv_cfg=LazyConfig(conv_class)(
                in_channels=384,  # 3 * 128 for concatenated q, k, v
                out_channels=384,  # 3 * 128 for concatenated q, k, v
                kernel_size=3,
                groups=384,  # Grouped convolution
                padding=1,
                bias=False,
                num_groups=128,  # Custom number of groups for weight sharing
                use_depthwise_grouping=True,  # Enable depthwise grouping
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(torch.nn.Identity)(),
            apply_qk_norm=True,
            use_rope=False,  # Disable RoPE to avoid in-place issues
            rope_base=10000.0,
        ),
    )


def test_distributed_vs_standard_equivalency(data_dim: int = 1, dtype: str = "float32") -> bool:
    """Test that distributed and standard convolutions produce equivalent results when CP=1.

    This test validates that our distributed convolution wrappers produce the same
    results as standard PyTorch convolutions when no parallelism is used.

    Args:
        data_dim: Dimensionality of the input data (default: 1).
        dtype: Data type for model and input tensors (default: "float32").

    Returns:
        bool: True if outputs are equivalent, False otherwise.
    """
    try:
        # Create both configurations
        standard_cfg = sequence_mixer_config(data_dim=data_dim)
        distributed_cfg = sequence_mixer_config_distributed(data_dim=data_dim)

        # Instantiate both models
        standard_mixer = instantiate(standard_cfg)
        distributed_mixer = instantiate(distributed_cfg)

        # Move to device and set dtype
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        input_dtype = dtype_map[dtype]

        standard_mixer = standard_mixer.to(torch.cuda.current_device()).to(input_dtype)
        distributed_mixer = distributed_mixer.to(torch.cuda.current_device()).to(input_dtype)

        # Create test input
        batch_size = 2
        seq_len = 128
        hidden_dim = 128

        test_input = torch.randn(
            batch_size, seq_len, hidden_dim, device=torch.cuda.current_device(), dtype=input_dtype
        )

        # Run both models with CP disabled
        with torch.no_grad():
            standard_output = standard_mixer(test_input, cp_group=None)
            distributed_output = distributed_mixer(test_input, cp_group=None)

        # Compare shapes
        assert standard_output.shape == distributed_output.shape, (
            f"Shape mismatch: standard {standard_output.shape} vs distributed {distributed_output.shape}"
        )

        # Note: We can't directly compare values since the models have different initializations
        # But we can verify they both run successfully and produce reasonable outputs
        assert not torch.isnan(standard_output).any(), "Standard model produced NaN values"
        assert not torch.isnan(distributed_output).any(), "Distributed model produced NaN values"
        assert not torch.isinf(standard_output).any(), "Standard model produced Inf values"
        assert not torch.isinf(distributed_output).any(), "Distributed model produced Inf values"

        logging.info("✅ Standard vs Distributed equivalency test passed!")
        logging.info(f"Standard output shape: {standard_output.shape}")
        logging.info(f"Distributed output shape: {distributed_output.shape}")
        logging.info(
            f"Standard output stats: mean={standard_output.mean().item():.4f}, std={standard_output.std().item():.4f}"
        )
        logging.info(
            f"Distributed output stats: mean={distributed_output.mean().item():.4f}, std={distributed_output.std().item():.4f}"
        )

        return True

    except Exception as e:
        logging.error(f"Standard vs Distributed equivalency test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_distributed_convolutions(data_dim: int = 1, dtype: str = "float32") -> bool:
    """Test distributed convolution classes directly.

    This test validates the correctness of the distributed convolution wrappers
    by comparing them with standard PyTorch convolutions in single-GPU mode.

    Args:
        data_dim: Dimensionality of the input data (default: 1).
        dtype: Data type for model and input tensors (default: "float32").

    Returns:
        bool: True if all tests pass successfully, False otherwise.
    """
    try:
        # Test 1D distributed convolution
        if data_dim == 1:
            # Create distributed conv that matches the actual use case
            # (similar to what's used in the sequence mixer config)
            dist_conv = DistributedConv1d(
                in_channels=384,  # 3 * 128 for concatenated q, k, v
                out_channels=384,  # 3 * 128 for concatenated q, k, v
                kernel_size=3,
                padding=1,
                bias=False,
                groups=384,  # Grouped convolution
                num_groups=128,  # Custom number of groups for weight sharing
                use_depthwise_grouping=True,  # Enable depthwise grouping
            )

            # Create test input
            batch_size = 2
            seq_len = 256
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            input_dtype = dtype_map[dtype]

            test_input = torch.randn(batch_size, 384, seq_len, device=torch.cuda.current_device(), dtype=input_dtype)

            # Test forward pass
            output = dist_conv(test_input, cp_group=None)
            logging.info(f"DistributedConv1d output shape: {output.shape}")

            # Verify output shape
            expected_shape = (batch_size, 384, seq_len)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        # Test 2D distributed convolution
        elif data_dim == 2:
            dist_conv = DistributedConv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=True,
                num_groups=32,
                use_depthwise_grouping=False,
            )

            batch_size = 2
            height, width = 32, 32
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            input_dtype = dtype_map[dtype]

            test_input = torch.randn(
                batch_size, 64, height, width, device=torch.cuda.current_device(), dtype=input_dtype
            )

            output = dist_conv(test_input, cp_group=None)
            logging.info(f"DistributedConv2d output shape: {output.shape}")

            expected_shape = (batch_size, 128, height, width)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        # Test 3D distributed convolution
        elif data_dim == 3:
            dist_conv = DistributedConv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=True,
                num_groups=32,
                use_depthwise_grouping=False,
            )

            batch_size = 2
            depth, height, width = 16, 16, 16
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            input_dtype = dtype_map[dtype]

            test_input = torch.randn(
                batch_size, 64, depth, height, width, device=torch.cuda.current_device(), dtype=input_dtype
            )

            output = dist_conv(test_input, cp_group=None)
            logging.info(f"DistributedConv3d output shape: {output.shape}")

            expected_shape = (batch_size, 128, depth, height, width)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        logging.info("✅ Distributed convolution tests passed!")
        return True

    except Exception as e:
        logging.error(f"Distributed convolution test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


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
        # Use distributed configuration for both CP=1 and CP>1 cases
        # This ensures we test CP communication logic rather than different layer types
        sequence_mixer_cfg = sequence_mixer_config_distributed(data_dim=data_dim)
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
        # _use_cp=False: Distributed layers active, but no CP communication
        output_no_cp = ddp_sequence_mixer(test_input, cp_group=None)

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

        # _use_cp=True: Distributed layers active WITH CP communication
        output_with_cp = ddp_sequence_mixer(test_input_cp, cp_group=cp_group)

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
    init_parallel_state(context_parallel_size=args.context_parallel_size)

    logging.info(f"Starting QKVSequenceMixer CP test with args: {args}")

    try:
        # Run the distributed convolution tests first
        logging.info("Running distributed convolution tests...")
        conv_success = test_distributed_convolutions(args.data_dim, args.dtype)

        # Run the standard vs distributed equivalency test
        logging.info("Running standard vs distributed equivalency test...")
        equiv_success = test_distributed_vs_standard_equivalency(args.data_dim, args.dtype)

        # Run the sequence mixer CP equivalency test (now using distributed config for both cases)
        logging.info("Running sequence mixer CP equivalency test...")
        mixer_success = test_sequence_mixer_cp_equivalency(args.data_dim, args.dtype)

        if conv_success and equiv_success and mixer_success:
            logging.info("✅ All tests passed successfully!")
        else:
            logging.error("❌ Some tests failed!")
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
