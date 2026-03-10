# TODO: Add license header here

"""
Test module for all-to-all communication functions with zigzag splitting.
"""

import torch

from nvsubquadratic.parallel.a2a_comms import (
    _get_inverse_zigzag_indices,
    _get_zigzag_indices,
)


class TestZigzagSplitting:
    """Test cases for zigzag splitting functionality."""

    def test_zigzag_indices_generation(self):
        """Test that zigzag indices are generated correctly."""
        cp_world_size = 2  # Context parallel world size
        num_chunks = 2 * cp_world_size  # Total number of chunks for zigzag splitting
        zigzag_idx = _get_zigzag_indices(num_chunks)
        inverse_zigzag_idx = _get_inverse_zigzag_indices(num_chunks)

        # Check that indices are within bounds
        assert torch.all(zigzag_idx >= 0)
        assert torch.all(zigzag_idx < num_chunks)
        assert torch.all(inverse_zigzag_idx >= 0)
        assert torch.all(inverse_zigzag_idx < num_chunks)

        # Check that inverse operation works
        assert torch.allclose(zigzag_idx[inverse_zigzag_idx], torch.arange(num_chunks))
        assert torch.allclose(inverse_zigzag_idx[zigzag_idx], torch.arange(num_chunks))

    def test_zigzag_splitting_1d(self):
        """Test zigzag splitting for 1D tensors."""
        # Parameters
        B = 2  # Batch size
        d = 4  # Hidden size
        L = 16  # Sequence length
        cp_world_size = 2  # Context parallel world size
        num_chunks = 2 * cp_world_size  # Total number of chunks for zigzag splitting

        # Create an input tensor with sequential values
        input_tensor = torch.arange(B * d * L).reshape(B, d, L).float()

        # Generate zigzag indices
        zigzag_idx = _get_zigzag_indices(num_chunks)

        # Apply zigzag splitting
        chunk_length = L // num_chunks
        input_reshaped = input_tensor.reshape(B, d, num_chunks, chunk_length)
        zigzag_tensor = input_reshaped.index_select(dim=2, index=zigzag_idx).reshape(B, d, L)

        # Generate inverse zigzag indices
        inverse_zigzag_idx = _get_inverse_zigzag_indices(num_chunks)

        # Apply inverse zigzag rearrangement
        zigzag_reshaped = zigzag_tensor.reshape(B, d, num_chunks, chunk_length)
        recovered_tensor = zigzag_reshaped.index_select(dim=2, index=inverse_zigzag_idx).reshape(B, d, L)

        # Verify that the recovered tensor matches the original tensor
        assert torch.allclose(input_tensor, recovered_tensor), "Zigzag splitting round-trip failed"

    def test_zigzag_splitting_2d(self):
        """Test zigzag splitting for 2D tensors."""
        # Parameters
        B = 2  # Batch size
        d = 4  # Number of features/channels
        H = 8  # Height
        W = 8  # Width
        cp_world_size = 2  # Context parallel world size
        num_chunks = 2 * cp_world_size  # Total number of chunks for zigzag splitting

        # Create an input tensor with sequential values
        input_tensor = torch.arange(B * d * H * W).reshape(B, d, H, W).float()

        # Generate zigzag indices
        zigzag_idx = _get_zigzag_indices(num_chunks)

        # Apply zigzag splitting to the height dimension
        chunk_length = H // num_chunks
        input_reshaped = input_tensor.reshape(B, d, num_chunks, chunk_length, -1)
        zigzag_tensor = input_reshaped.index_select(dim=2, index=zigzag_idx).reshape(B, d, H, W)

        # Generate inverse zigzag indices
        inverse_zigzag_idx = _get_inverse_zigzag_indices(num_chunks)

        # Apply inverse zigzag rearrangement
        zigzag_reshaped = zigzag_tensor.reshape(B, d, num_chunks, chunk_length, -1)
        recovered_tensor = zigzag_reshaped.index_select(dim=2, index=inverse_zigzag_idx).reshape(B, d, H, W)

        # Verify that the recovered tensor matches the original tensor
        assert torch.allclose(input_tensor, recovered_tensor), "Zigzag splitting round-trip failed for 2D"

    def test_zigzag_splitting_3d(self):
        """Test zigzag splitting for 3D tensors."""
        # Parameters
        B = 2  # Batch size
        d = 4  # Number of features/channels
        T = 8  # Temporal dimension
        H = 4  # Height
        W = 4  # Width
        cp_world_size = 2  # Context parallel world size
        num_chunks = 2 * cp_world_size  # Total number of chunks for zigzag splitting

        # Create an input tensor with sequential values
        input_tensor = torch.arange(B * d * T * H * W).reshape(B, d, T, H, W).float()

        # Generate zigzag indices
        zigzag_idx = _get_zigzag_indices(num_chunks)

        # Apply zigzag splitting to the temporal dimension
        chunk_length = T // num_chunks
        input_reshaped = input_tensor.reshape(B, d, num_chunks, chunk_length, -1)
        zigzag_tensor = input_reshaped.index_select(dim=2, index=zigzag_idx).reshape(B, d, T, H, W)

        # Generate inverse zigzag indices
        inverse_zigzag_idx = _get_inverse_zigzag_indices(num_chunks)

        # Apply inverse zigzag rearrangement
        zigzag_reshaped = zigzag_tensor.reshape(B, d, num_chunks, chunk_length, -1)
        recovered_tensor = zigzag_reshaped.index_select(dim=2, index=inverse_zigzag_idx).reshape(B, d, T, H, W)

        # Verify that the recovered tensor matches the original tensor
        assert torch.allclose(input_tensor, recovered_tensor), "Zigzag splitting round-trip failed for 3D"


if __name__ == "__main__":
    # Run the tests
    import pytest

    pytest.main([__file__, "-v"])
