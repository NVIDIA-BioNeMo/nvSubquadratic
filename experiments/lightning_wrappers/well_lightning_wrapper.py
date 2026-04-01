"""Lightning wrapper for WELL benchmark tasks."""

from typing import Literal

import torch
from einops import rearrange
from the_well.benchmark.metrics import validation_metric_suite
from the_well.data.data_formatter import DefaultChannelsLastFormatter

from experiments.default_cfg import ExperimentConfig
from experiments.lightning_wrappers.regression_wrapper import RegressionWrapper


class WELLRegressionWrapper(RegressionWrapper):
    """Lightning wrapper for WELL benchmark regression tasks.

    This wrapper adapts the WELL benchmark's data format and metrics to work
    with the nvSubquadratic training infrastructure.

    Training is done with 1-step prediction. Validation uses both short and long
    autoregressive rollouts with WELL benchmark metrics.

    WELL data format:
    - Batch format: dict with 'input_fields', 'constant_fields' (optional), etc.
    - Fields are in channels-last format: [B, T, H, W, C]

    Model expects:
    - Input: [B, H, W, C_in] where C_in = n_steps_input * n_fields + n_constant_fields
    - Output: [B, H, W, C_out] where C_out = n_fields

    Args:
        network: Network to wrap
        cfg: Experiment configuration
        metadata: WELL dataset metadata
        n_steps_input: Number of input timesteps
        n_steps_output: Number of output timesteps (for training, usually 1)
        max_rollout_steps: Maximum rollout steps for validation
        metric: Training metric ('MSE' or 'MAE')
    """

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
        metadata,
        n_steps_input: int = 4,
        n_steps_output: int = 1,
        max_rollout_steps: int = 32,
        metric: Literal["MAE", "MSE"] = "MSE",
        normalization=None,
    ):
        """Initialize the WELL regression wrapper with dataset metadata and rollout settings."""
        super().__init__(network=network, cfg=cfg, metric=metric)

        self.metadata = metadata
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.max_rollout_steps = max_rollout_steps

        # Setup WELL benchmark metrics
        self.validation_metric_suite = validation_metric_suite

        # Data formatter for WELL benchmark
        self.formatter = DefaultChannelsLastFormatter(metadata)

        # Track best validation loss
        self.best_val_loss = float("inf")

        # Normalization object for denormalizing metric inputs back to physical scale
        self.normalization = normalization

    def _denormalize_for_metrics(self, predictions, targets):
        """Denormalize predictions and targets for metric computation.

        This matches the WELL benchmark behavior where metrics are computed
        on physical scale (denormalized) data, even though the model works
        with normalized data.

        Args:
            predictions: [B, T, H, W, C] normalized predictions
            targets: [B, T, H, W, C] normalized targets

        Returns:
            denormalized_predictions, denormalized_targets
        """
        if self.normalization is None:
            # No normalization, return as is
            return predictions, targets

        # Denormalize using the normalization object
        # The data is in [B, T, H, W, C] format, we need to denormalize per channel
        pred_denorm = self.normalization.denormalize_flattened(predictions, "variable")
        target_denorm = self.normalization.denormalize_flattened(targets, "variable")

        return pred_denorm, target_denorm

    def _process_batch_input(self, batch):
        """Process WELL batch format into model input.

        Args:
            batch: Dict with 'input_fields' [B, T, *spatial, C] and optional 'constant_fields' [B, *spatial, C]
                   2D: spatial = (H, W), 3D: spatial = (D, H, W)

        Returns:
            model_input: [B, *spatial, C_in] where C_in = T*C + C_const
        """
        input_fields = batch["input_fields"]

        # Flatten timesteps into channels: [B, T, *spatial, C] -> [B, *spatial, T*C]
        ndim = input_fields.ndim
        if ndim == 5:  # 2D: [B, T, H, W, C]
            model_input = rearrange(input_fields, "b t h w c -> b h w (t c)")
        elif ndim == 6:  # 3D: [B, T, D, H, W, C]
            model_input = rearrange(input_fields, "b t d h w c -> b d h w (t c)")
        else:
            raise ValueError(f"Unexpected input_fields ndim={ndim}, expected 5 (2D) or 6 (3D)")

        # Concatenate constant fields if present
        if "constant_fields" in batch:
            constant_fields = batch["constant_fields"]
            model_input = torch.cat([model_input, constant_fields], dim=-1)

        return model_input

    def training_step(self, batch, batch_idx):
        """Training uses 1-step prediction.

        Args:
            batch: Dict with 'input_fields' and target in subsequent timestep
            batch_idx: Index of the current batch
        """
        # Process input
        model_input = self._process_batch_input(batch)  # [B, H, W, C_in]

        # Get target directly from batch (avoids redundant rearrange + nan_to_num
        # that formatter.process_input() would do on input_fields)
        y_ref = batch["output_fields"]  # [B, n_steps_output, *spatial, C]

        # For single-step prediction, squeeze time dimension
        if self.n_steps_output == 1:
            target = y_ref[:, 0]  # [B, *spatial, C]
        else:
            target = y_ref  # Keep as is for multi-step

        # Forward pass
        pred = self({"input": model_input, "condition": None})["logits"]  # [B, H, W, C]

        # Compute loss
        loss = self.loss_metric(pred, target)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed)

        return {"loss": loss}

    def _autoregressive_rollout(self, batch, num_steps):
        """Perform autoregressive rollout prediction.

        Args:
            batch: Initial batch dict
            num_steps: Number of steps to rollout

        Returns:
            predictions: [B, num_steps, H, W, C]
            targets: [B, num_steps, H, W, C]
        """
        # Get initial input and target
        _, y_ref = self.formatter.process_input(batch)  # [B, T_total, H, W, C]
        num_steps = min(num_steps, y_ref.shape[1])
        y_ref = y_ref[:, :num_steps]

        # Initialize with input fields
        current_input = batch["input_fields"].clone()  # [B, T_input, H, W, C]
        predictions = []

        for step in range(num_steps):
            # Create batch dict for current step
            current_batch = {"input_fields": current_input}
            if "constant_fields" in batch:
                current_batch["constant_fields"] = batch["constant_fields"]

            # Process input and predict
            model_input = self._process_batch_input(current_batch)
            pred = self({"input": model_input, "condition": None})["logits"]  # [B, H, W, C]

            # Store prediction
            predictions.append(pred.unsqueeze(1))  # [B, 1, H, W, C]

            # Update input for next step: drop oldest, append prediction
            if step < num_steps - 1:
                current_input = torch.cat(
                    [
                        current_input[:, 1:],  # Drop first timestep
                        pred.unsqueeze(1),  # Add prediction as new timestep
                    ],
                    dim=1,
                )

        predictions = torch.cat(predictions, dim=1)  # [B, num_steps, H, W, C]
        return predictions, y_ref

    def validation_step(self, batch, batch_idx):
        """Validation uses autoregressive rollout and WELL benchmark metrics.

        Args:
            batch: Batch dict from validation dataloader
            batch_idx: Index of the current batch
        """
        # Perform rollout (WELL validation typically uses shorter rollouts)
        predictions, targets = self._autoregressive_rollout(batch, self.n_steps_output)

        # Denormalize predictions and targets for metric computation
        # This matches WELL benchmark behavior: model works with normalized data,
        # but metrics are computed on physical scale
        predictions_denorm, targets_denorm = self._denormalize_for_metrics(predictions, targets)

        # Compute WELL benchmark metrics on denormalized data
        metric_results = {}
        for metric_fn in self.validation_metric_suite:
            # WELL metrics expect [B, T, H, W, C] format and metadata
            metric_value = metric_fn(predictions_denorm, targets_denorm, self.metadata)

            # Handle both scalar and dict returns
            if isinstance(metric_value, dict):
                for k, v in metric_value.items():
                    if isinstance(v, torch.Tensor):
                        v = v.mean()
                    metric_results[f"val/{k}"] = v
            else:
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.mean()
                metric_results[f"val/{metric_fn.__class__.__name__}"] = metric_value

        # Log all metrics
        for name, value in metric_results.items():
            self.log(name, value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=self.distributed)

        # Use MSE as primary validation loss for consistency (computed on denormalized data)
        val_loss = metric_results.get("val/MSELoss", self.loss_metric(predictions_denorm, targets_denorm))
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)

        return {}

    def test_step(self, batch, batch_idx):
        """Test uses autoregressive rollout and WELL benchmark metrics.

        Args:
            batch: Batch dict from test dataloader
            batch_idx: Index of the current batch
        """
        # Perform rollout
        predictions, targets = self._autoregressive_rollout(batch, self.n_steps_output)

        # Denormalize predictions and targets for metric computation
        predictions_denorm, targets_denorm = self._denormalize_for_metrics(predictions, targets)

        # Compute WELL benchmark metrics on denormalized data
        metric_results = {}
        for metric_fn in self.validation_metric_suite:
            metric_value = metric_fn(predictions_denorm, targets_denorm, self.metadata)

            if isinstance(metric_value, dict):
                for k, v in metric_value.items():
                    if isinstance(v, torch.Tensor):
                        v = v.mean()
                    metric_results[f"test/{k}"] = v
            else:
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.mean()
                metric_results[f"test/{metric_fn.__class__.__name__}"] = metric_value

        # Log all metrics
        for name, value in metric_results.items():
            self.log(name, value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=self.distributed)

        # Use MSE as primary test loss (computed on denormalized data)
        test_loss = metric_results.get("test/MSELoss", self.loss_metric(predictions_denorm, targets_denorm))
        self.log("test/loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)

        return {}

    def on_validation_epoch_end(self):
        """Log best validation loss (rank 0 only to avoid duplicate wandb logs)."""
        if self.trainer.sanity_checking:
            return
        if not self.trainer.is_global_zero:
            return
        if "val/loss" in self.trainer.callback_metrics:
            val_loss = self.trainer.callback_metrics["val/loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss.item()
                if hasattr(self, "logger") and hasattr(self.logger, "experiment"):
                    self.logger.experiment.log(
                        {
                            "val/best_loss": self.best_val_loss,
                            "global_step": self.global_step,
                        }
                    )
