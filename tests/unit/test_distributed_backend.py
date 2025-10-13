# TODO: Add license header here

"""Unit tests for distributed backend abstraction.

These tests verify the backend abstraction interface and basic functionality
without requiring actual distributed training setup.
"""

import pytest

from nvsubquadratic.distributed.backend import (
    MegatronBackend,
    ParallelConfig,
    create_backend,
    get_context_parallel_group,
    get_global_backend,
    set_global_backend,
)


class TestParallelConfig:
    """Test ParallelConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ParallelConfig()
        assert config.backend_type == "megatron"
        assert config.context_parallel_size == 1
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert config.data_parallel_size == -1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ParallelConfig(
            backend_type="megatron",
            context_parallel_size=2,
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
        )
        assert config.backend_type == "megatron"
        assert config.context_parallel_size == 2
        assert config.tensor_parallel_size == 4
        assert config.pipeline_parallel_size == 2


class TestBackendFactory:
    """Test backend factory function."""

    def test_create_megatron_backend(self):
        """Test creation of Megatron backend."""
        config = ParallelConfig(backend_type="megatron")
        backend = create_backend(config)
        assert isinstance(backend, MegatronBackend)
        assert backend.config == config
        assert not backend.is_initialized

    def test_create_device_mesh_backend_raises(self):
        """Test that DeviceMesh backend raises NotImplementedError."""
        config = ParallelConfig(backend_type="device_mesh")
        backend = create_backend(config)
        # Should create but not be usable yet
        with pytest.raises(NotImplementedError):
            backend.initialize(world_size=2, rank=0)

    def test_create_unknown_backend_raises(self):
        """Test that unknown backend raises ValueError."""
        config = ParallelConfig()
        config.backend_type = "unknown"  # type: ignore
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend(config)


class TestGlobalBackendContext:
    """Test global backend context management."""

    def test_default_global_backend_is_none(self):
        """Test that global backend is None by default."""
        set_global_backend(None)  # Reset
        assert get_global_backend() is None

    def test_set_and_get_global_backend(self):
        """Test setting and getting global backend."""
        config = ParallelConfig(backend_type="megatron")
        backend = create_backend(config)

        set_global_backend(backend)
        assert get_global_backend() is backend

        # Cleanup
        set_global_backend(None)

    def test_get_context_parallel_group_without_backend(self):
        """Test getting CP group when no backend is set."""
        set_global_backend(None)
        assert get_context_parallel_group() is None


class TestMegatronBackendInterface:
    """Test MegatronBackend interface (without actual initialization)."""

    def test_backend_creation(self):
        """Test creating Megatron backend."""
        config = ParallelConfig(
            backend_type="megatron",
            context_parallel_size=2,
        )
        backend = MegatronBackend(config)

        assert backend.config == config
        assert not backend.is_initialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
