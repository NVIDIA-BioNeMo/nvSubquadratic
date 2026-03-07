"""Tests for the AutoregressiveWrapper.

Tests cover:
1. Batch preparation (input/target shifting)
2. Discrete mode (cross-entropy loss, accuracy)
3. Continuous mode (MSE/MAE loss)
4. Generation utilities (sampling, greedy)
5. Forward pass shape correctness
"""

import pytest
import torch
import torch.nn as nn

from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.autoregressive_wrapper import AutoregressiveWrapper
from nvsubq_paper.lazy_config import PLACEHOLDER, LazyConfig


# =============================================================================
# Mock Network for Testing
# =============================================================================


class MockAutoregressiveNetwork(nn.Module):
    """Simple mock network that outputs logits for testing.

    Args:
        hidden_dim: Hidden dimension.
        output_dim: Output dimension (vocab_size for discrete, channels for continuous).
    """

    def __init__(self, hidden_dim: int = 64, output_dim: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embed = nn.Linear(1, hidden_dim)  # Simple embedding
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_and_condition: dict) -> dict:
        x = input_and_condition["input"]  # [B, L] or [B, L, C]

        # Handle discrete (integer) input
        if x.dtype in (torch.long, torch.int):
            x = x.float().unsqueeze(-1)  # [B, L] -> [B, L, 1]

        # Simple forward pass
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # [B, L] -> [B, L, 1]

        h = self.embed(x)  # [B, L, hidden_dim]
        logits = self.proj(h)  # [B, L, output_dim]

        return {"logits": logits}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a minimal experiment config for testing."""
    config = ExperimentConfig()
    config.optimizer = LazyConfig(torch.optim.AdamW)(lr=1e-4, weight_decay=0.0)
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.0,
        total_iterations=100,
        mode="min",
    )
    config.train = TrainConfig(batch_size=4, iterations=100, grad_clip=1.0)
    config.wandb = WandbConfig(job_group="test", entity="test", project="test")
    config.dataset = PLACEHOLDER
    config.net = PLACEHOLDER
    config.callbacks = []
    return config


@pytest.fixture
def discrete_network():
    """Create a mock network for discrete token prediction."""
    return MockAutoregressiveNetwork(hidden_dim=64, output_dim=100)  # vocab_size=100


@pytest.fixture
def continuous_network():
    """Create a mock network for continuous value prediction."""
    return MockAutoregressiveNetwork(hidden_dim=64, output_dim=1)  # 1 channel output


# =============================================================================
# Tests: Batch Preparation
# =============================================================================


class TestBatchPreparation:
    """Tests for input/target shifting in _prepare_batch."""

    def test_shift_discrete_2d(self, discrete_network, mock_config):
        """Test shifting for discrete tokens [B, L]."""
        wrapper = AutoregressiveWrapper(
            network=discrete_network,
            cfg=mock_config,
            mode="discrete",
            vocab_size=100,
        )

        # Create batch with sequence length 10
        batch = {"input": torch.randint(0, 100, (4, 10))}  # [B, L]

        input_seq, target_seq = wrapper._prepare_batch(batch)

        assert input_seq.shape == (4, 9), f"Expected (4, 9), got {input_seq.shape}"
        assert target_seq.shape == (4, 9), f"Expected (4, 9), got {target_seq.shape}"
        # Check shifting is correct
        assert torch.equal(input_seq, batch["input"][:, :-1])
        assert torch.equal(target_seq, batch["input"][:, 1:])

    def test_shift_continuous_3d(self, continuous_network, mock_config):
        """Test shifting for continuous values [B, L, C]."""
        wrapper = AutoregressiveWrapper(
            network=continuous_network,
            cfg=mock_config,
            mode="continuous",
            loss_type="mse",
        )

        # Create batch with sequence length 10, 3 channels
        batch = {"input": torch.randn(4, 10, 3)}  # [B, L, C]

        input_seq, target_seq = wrapper._prepare_batch(batch)

        assert input_seq.shape == (4, 9, 3), f"Expected (4, 9, 3), got {input_seq.shape}"
        assert target_seq.shape == (4, 9, 3), f"Expected (4, 9, 3), got {target_seq.shape}"
        # Check shifting is correct
        assert torch.equal(input_seq, batch["input"][:, :-1, :])
        assert torch.equal(target_seq, batch["input"][:, 1:, :])


# =============================================================================
# Tests: Discrete Mode
# =============================================================================


class TestDiscreteMode:
    """Tests for discrete token prediction mode."""

    def test_discrete_forward_shape(self, discrete_network, mock_config):
        """Test output shape in discrete mode."""
        wrapper = AutoregressiveWrapper(
            network=discrete_network,
            cfg=mock_config,
            mode="discrete",
            vocab_size=100,
        )

        batch = {"input": torch.randint(0, 100, (4, 10))}
        input_seq, target_seq = wrapper._prepare_batch(batch)

        output = wrapper({"input": input_seq, "condition": None})
        logits = output["logits"]

        # Should be [B, L-1, vocab_size]
        assert logits.shape == (4, 9, 100), f"Expected (4, 9, 100), got {logits.shape}"

    def test_discrete_loss_computes(self, discrete_network, mock_config):
        """Test that cross-entropy loss computes without error."""
        wrapper = AutoregressiveWrapper(
            network=discrete_network,
            cfg=mock_config,
            mode="discrete",
            vocab_size=100,
        )

        batch = {"input": torch.randint(0, 100, (4, 10))}
        input_seq, target_seq = wrapper._prepare_batch(batch)

        output = wrapper({"input": input_seq, "condition": None})
        logits = output["logits"]

        loss, predictions = wrapper._compute_loss(logits, target_seq)

        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert predictions.shape == (4, 9), f"Predictions shape: {predictions.shape}"

    def test_discrete_training_step(self, discrete_network, mock_config):
        """Test full training step in discrete mode."""
        wrapper = AutoregressiveWrapper(
            network=discrete_network,
            cfg=mock_config,
            mode="discrete",
            vocab_size=100,
        )

        input_tensor = torch.randint(0, 100, (4, 10))
        batch = {"input": input_tensor, "label": input_tensor, "condition": None}
        loss = wrapper.training_step(batch, batch_idx=0)

        assert loss.ndim == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"


# =============================================================================
# Tests: Continuous Mode
# =============================================================================


class TestContinuousMode:
    """Tests for continuous value prediction mode."""

    def test_continuous_forward_shape(self, continuous_network, mock_config):
        """Test output shape in continuous mode."""
        wrapper = AutoregressiveWrapper(
            network=continuous_network,
            cfg=mock_config,
            mode="continuous",
            loss_type="mse",
        )

        batch = {"input": torch.randn(4, 10, 1)}
        input_seq, target_seq = wrapper._prepare_batch(batch)

        output = wrapper({"input": input_seq, "condition": None})
        logits = output["logits"]

        # Should be [B, L-1, 1]
        assert logits.shape == (4, 9, 1), f"Expected (4, 9, 1), got {logits.shape}"

    def test_continuous_mse_loss(self, continuous_network, mock_config):
        """Test MSE loss in continuous mode."""
        wrapper = AutoregressiveWrapper(
            network=continuous_network,
            cfg=mock_config,
            mode="continuous",
            loss_type="mse",
        )

        batch = {"input": torch.randn(4, 10, 1)}
        input_seq, target_seq = wrapper._prepare_batch(batch)

        output = wrapper({"input": input_seq, "condition": None})
        logits = output["logits"]

        loss, predictions = wrapper._compute_loss(logits, target_seq)

        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0, "MSE should be non-negative"

    def test_continuous_mae_loss(self, mock_config):
        """Test MAE loss in continuous mode."""
        network = MockAutoregressiveNetwork(hidden_dim=64, output_dim=1)
        wrapper = AutoregressiveWrapper(
            network=network,
            cfg=mock_config,
            mode="continuous",
            loss_type="mae",
        )

        input_tensor = torch.randn(4, 10, 1)
        batch = {"input": input_tensor, "label": input_tensor, "condition": None}
        loss = wrapper.training_step(batch, batch_idx=0)

        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0, "MAE should be non-negative"


# =============================================================================
# Tests: Generation
# =============================================================================


class TestGeneration:
    """Tests for autoregressive generation utilities."""

    def test_generate_discrete_shape(self, discrete_network, mock_config):
        """Test generation output shape for discrete tokens."""
        wrapper = AutoregressiveWrapper(
            network=discrete_network,
            cfg=mock_config,
            mode="discrete",
            vocab_size=100,
        )

        prompt = torch.randint(0, 100, (2, 5))  # [B, L] = [2, 5]
        max_new_tokens = 10

        generated = wrapper.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
        )

        expected_length = 5 + 10  # prompt + new tokens
        assert generated.shape == (2, expected_length), f"Expected (2, {expected_length}), got {generated.shape}"

    def test_generate_greedy(self, discrete_network, mock_config):
        """Test greedy generation."""
        wrapper = AutoregressiveWrapper(
            network=discrete_network,
            cfg=mock_config,
            mode="discrete",
            vocab_size=100,
        )

        prompt = torch.randint(0, 100, (2, 5))
        max_new_tokens = 5

        generated = wrapper.generate_greedy(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        expected_length = 5 + 5
        assert generated.shape == (2, expected_length)

    def test_generate_with_top_k(self, discrete_network, mock_config):
        """Test generation with top-k sampling."""
        wrapper = AutoregressiveWrapper(
            network=discrete_network,
            cfg=mock_config,
            mode="discrete",
            vocab_size=100,
        )

        prompt = torch.randint(0, 100, (2, 5))

        generated = wrapper.generate(
            prompt=prompt,
            max_new_tokens=5,
            temperature=0.8,
            top_k=10,
        )

        assert generated.shape == (2, 10)

    def test_generate_with_top_p(self, discrete_network, mock_config):
        """Test generation with nucleus (top-p) sampling."""
        wrapper = AutoregressiveWrapper(
            network=discrete_network,
            cfg=mock_config,
            mode="discrete",
            vocab_size=100,
        )

        prompt = torch.randint(0, 100, (2, 5))

        generated = wrapper.generate(
            prompt=prompt,
            max_new_tokens=5,
            temperature=0.9,
            top_p=0.95,
        )

        assert generated.shape == (2, 10)

    def test_generate_continuous(self, continuous_network, mock_config):
        """Test generation for continuous values."""
        wrapper = AutoregressiveWrapper(
            network=continuous_network,
            cfg=mock_config,
            mode="continuous",
            loss_type="mse",
        )

        prompt = torch.randn(2, 5, 1)  # [B, L, C]
        max_new_tokens = 5

        generated = wrapper.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        expected_length = 5 + 5
        assert generated.shape == (2, expected_length, 1), f"Expected (2, {expected_length}, 1), got {generated.shape}"


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_discrete_requires_vocab_size(self, discrete_network, mock_config):
        """Test that discrete mode requires vocab_size."""
        with pytest.raises(ValueError, match="vocab_size must be provided"):
            AutoregressiveWrapper(
                network=discrete_network,
                cfg=mock_config,
                mode="discrete",
                vocab_size=None,  # Should raise error
            )

    def test_ignore_index(self, discrete_network, mock_config):
        """Test that ignore_index is passed to loss function."""
        wrapper = AutoregressiveWrapper(
            network=discrete_network,
            cfg=mock_config,
            mode="discrete",
            vocab_size=100,
            ignore_index=-100,
        )

        assert wrapper.loss_fn.ignore_index == -100

    def test_batch_size_1(self, discrete_network, mock_config):
        """Test with batch size 1."""
        wrapper = AutoregressiveWrapper(
            network=discrete_network,
            cfg=mock_config,
            mode="discrete",
            vocab_size=100,
        )

        input_tensor = torch.randint(0, 100, (1, 10))  # Batch size 1
        batch = {"input": input_tensor, "label": input_tensor, "condition": None}
        loss = wrapper.training_step(batch, batch_idx=0)

        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_short_sequence(self, discrete_network, mock_config):
        """Test with minimum sequence length (2 tokens)."""
        wrapper = AutoregressiveWrapper(
            network=discrete_network,
            cfg=mock_config,
            mode="discrete",
            vocab_size=100,
        )

        input_tensor = torch.randint(0, 100, (4, 2))  # Length 2 -> 1 after shift
        batch = {"input": input_tensor, "label": input_tensor, "condition": None}
        loss = wrapper.training_step(batch, batch_idx=0)

        assert loss.ndim == 0
        assert not torch.isnan(loss)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
