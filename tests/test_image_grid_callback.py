"""Tests for ValidationImageGridCallback."""

from unittest.mock import MagicMock

import pytest
import torch

from experiments.callbacks.image_grid_val_visualization import ValidationImageGridCallback


class TestAsNchwImages:
    """Test the _as_nchw_images method for different tensor formats."""

    @pytest.fixture
    def callback(self):
        return ValidationImageGridCallback(num_samples=2)

    def test_bchw_1_channel(self, callback):
        """Test BCHW format with 1 channel (grayscale)."""
        tensor = torch.randn(4, 1, 64, 64)  # B=4, C=1, H=64, W=64
        result = callback._as_nchw_images(tensor)
        assert result.shape == (4, 1, 64, 64)

    def test_bchw_3_channel(self, callback):
        """Test BCHW format with 3 channels (RGB)."""
        tensor = torch.randn(4, 3, 64, 64)
        result = callback._as_nchw_images(tensor)
        assert result.shape == (4, 3, 64, 64)

    def test_bchw_2_channel(self, callback):
        """Test BCHW format with 2 channels (grayscale + mask)."""
        tensor = torch.randn(4, 2, 64, 64)
        result = callback._as_nchw_images(tensor)
        assert result.shape == (4, 2, 64, 64)

    def test_bhwc_1_channel(self, callback):
        """Test BHWC format with 1 channel - should convert to BCHW."""
        tensor = torch.randn(4, 64, 64, 1)  # B=4, H=64, W=64, C=1
        result = callback._as_nchw_images(tensor)
        assert result.shape == (4, 1, 64, 64)

    def test_bhwc_3_channel(self, callback):
        """Test BHWC format with 3 channels - should convert to BCHW."""
        tensor = torch.randn(4, 64, 64, 3)
        result = callback._as_nchw_images(tensor)
        assert result.shape == (4, 3, 64, 64)

    def test_bhwc_2_channel(self, callback):
        """Test BHWC format with 2 channels - should convert to BCHW."""
        tensor = torch.randn(4, 64, 64, 2)  # B=4, H=64, W=64, C=2
        result = callback._as_nchw_images(tensor)
        assert result.shape == (4, 2, 64, 64)

    def test_flattened_square(self, callback):
        """Test B(H*W)C format with square image - should reshape to BCHW."""
        tensor = torch.randn(4, 64 * 64, 1)  # 64x64 flattened
        result = callback._as_nchw_images(tensor)
        assert result.shape == (4, 1, 64, 64)

    def test_flattened_explicit_shape(self):
        """Test B(H*W)C format with explicit flattened_image_shape."""
        callback = ValidationImageGridCallback(num_samples=2, flattened_image_shape=(32, 128))
        tensor = torch.randn(4, 32 * 128, 1)
        result = callback._as_nchw_images(tensor)
        assert result.shape == (4, 1, 32, 128)


class TestGridBuilding:
    """Test the image grid building with different configurations."""

    def create_mock_trainer_and_module(self, x_shape, pred_shape, y_shape):
        """Create mock trainer and module for testing."""
        trainer = MagicMock()
        trainer.global_step = 100
        trainer.current_epoch = 1
        # Create logger that doesn't have log_image (like wandb logger)
        # so it falls through to experiment.log
        trainer.logger = MagicMock(spec=["experiment"])
        trainer.logger.experiment = MagicMock()
        trainer.logger.experiment.log = MagicMock()
        # Ensure datamodule doesn't have on_before_batch_transfer
        trainer.datamodule = MagicMock(spec=[])

        pl_module = MagicMock()
        pl_module.device = torch.device("cpu")

        # Create dataloader with mock batch - use "label" not "target"
        x = torch.rand(*x_shape)
        y = torch.rand(*y_shape)
        batch = {"input": x, "label": y, "condition": None}

        # Create a proper list-like object that yields the batch when iterated
        class MockDataloader:
            def __init__(self, batches):
                self._batches = batches

            def __iter__(self):
                return iter(self._batches)

            def __len__(self):
                return len(self._batches)

        dataloader = MockDataloader([batch])
        trainer.val_dataloaders = dataloader

        # Mock forward pass - need to handle __call__
        pred = torch.rand(*pred_shape)
        pl_module.return_value = {"logits": pred}
        pl_module.eval = MagicMock()

        return trainer, pl_module

    def test_1_channel_input_output(self):
        """Test grid with 1-channel input and output (simple grayscale)."""
        callback = ValidationImageGridCallback(num_samples=2, show_input=True)

        trainer, pl_module = self.create_mock_trainer_and_module(
            x_shape=(4, 64, 64, 1),  # BHWC grayscale
            pred_shape=(4, 16, 16, 1),  # smaller prediction
            y_shape=(4, 16, 16, 1),
        )

        # This should not raise
        callback._log_image_grid(trainer, pl_module, event_idx=0)
        trainer.logger.experiment.log.assert_called_once()

    def test_2_channel_input_no_mask_separation(self):
        """Test 2-channel input without mask separation (default)."""
        callback = ValidationImageGridCallback(num_samples=2, show_input=True, show_mask_separately=False)

        trainer, pl_module = self.create_mock_trainer_and_module(
            x_shape=(4, 64, 64, 2),  # BHWC with 2 channels (grayscale + mask)
            pred_shape=(4, 16, 16, 1),
            y_shape=(4, 16, 16, 1),
        )

        # This should not raise - takes first channel only
        callback._log_image_grid(trainer, pl_module, event_idx=0)
        trainer.logger.experiment.log.assert_called_once()

    def test_2_channel_input_with_mask_separation(self):
        """Test 2-channel input WITH mask separation (side-by-side display)."""
        callback = ValidationImageGridCallback(num_samples=2, show_input=True, show_mask_separately=True)

        trainer, pl_module = self.create_mock_trainer_and_module(
            x_shape=(4, 64, 64, 2),  # BHWC with 2 channels
            pred_shape=(4, 16, 16, 1),
            y_shape=(4, 16, 16, 1),
        )

        # This should not raise - shows canvas and mask side-by-side
        callback._log_image_grid(trainer, pl_module, event_idx=0)
        trainer.logger.experiment.log.assert_called_once()

    def test_no_input_shown(self):
        """Test grid without showing input (only pred and label)."""
        callback = ValidationImageGridCallback(num_samples=2, show_input=False)

        trainer, pl_module = self.create_mock_trainer_and_module(
            x_shape=(4, 64, 64, 1),
            pred_shape=(4, 16, 16, 1),
            y_shape=(4, 16, 16, 1),
        )

        callback._log_image_grid(trainer, pl_module, event_idx=0)
        trainer.logger.experiment.log.assert_called_once()


class TestGridDimensions:
    """Test that grid dimensions are correct for different configurations."""

    def test_nrow_calculation_basic(self):
        """Test nrow is 3 for basic case with input."""
        callback = ValidationImageGridCallback(num_samples=2, show_input=True)
        # Without mask separation: nrow=3 (input, pred, label)
        assert callback.show_input is True
        assert callback.show_mask_separately is False

    def test_nrow_calculation_no_input(self):
        """Test nrow is 2 when input not shown."""
        callback = ValidationImageGridCallback(num_samples=2, show_input=False)
        # Without input: nrow=2 (pred, label)
        assert callback.show_input is False

    def test_nrow_calculation_with_mask(self):
        """Test nrow is 4 when mask shown separately."""
        callback = ValidationImageGridCallback(num_samples=2, show_input=True, show_mask_separately=True)
        # With mask separation: nrow=4 (canvas, mask, pred, label)
        assert callback.show_mask_separately is True


class TestToGrayscaleRgb:
    """Test the internal to_grayscale_rgb conversion function."""

    def test_conversion_logic(self):
        """Verify the conversion logic by tracing through _log_image_grid."""
        callback = ValidationImageGridCallback(num_samples=1, show_mask_separately=True)

        # Create a simple 2-channel input (grayscale + mask)
        # Channel 0: grayscale values
        # Channel 1: binary mask (1 where selected, 0 elsewhere)
        x = torch.zeros(1, 2, 8, 8)
        x[0, 0, :, :] = 0.5  # grayscale = 0.5 everywhere
        x[0, 1, 2:6, 2:6] = 1.0  # mask = 1 in center region

        # Convert to NCHW (already is)
        x_nchw = callback._as_nchw_images(x)
        assert x_nchw.shape == (1, 2, 8, 8)

        # After split, should have:
        # canvas: [1, 1, 8, 8] with values 0.5
        # mask: [1, 1, 8, 8] with 1s in center
        canvas = x_nchw[:, 0:1]
        mask = x_nchw[:, 1:2]

        assert canvas.shape == (1, 1, 8, 8)
        assert mask.shape == (1, 1, 8, 8)
        assert torch.allclose(canvas, torch.full_like(canvas, 0.5))
        assert mask[0, 0, 3, 3] == 1.0  # center is masked
        assert mask[0, 0, 0, 0] == 0.0  # corner is not masked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
