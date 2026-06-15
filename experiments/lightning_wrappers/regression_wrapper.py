# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Lightning wrapper for regression tasks (MAE / MSE loss).

Provides :class:`RegressionWrapper`, which supports both Mean Absolute Error
(L1) and Mean Squared Error (L2) regression objectives.  It is also the base
class for :class:`~experiments.lightning_wrappers.well_lightning_wrapper.WELLRegressionWrapper`.
"""

from typing import Literal

import torch
import torchmetrics
import wandb

from experiments.default_cfg import ExperimentConfig
from experiments.lightning_wrappers.base_lightning_wrapper import LightningWrapperBase


class RegressionWrapper(LightningWrapperBase):
    """Lightning wrapper for regression tasks.

    .. TODO(@dwromero/dwessels): Resume support (see ClassificationWrapper for reference)
        - Add ``on_save_checkpoint`` / ``on_load_checkpoint`` to persist
          ``best_train_loss`` and ``best_val_loss`` across job resumes.
          Without this, best-metric tracking silently resets to initial
          values (1e9) after every SLURM preemption or manual resume.
        - Add corresponding tests in ``tests/test_checkpoint_resume.py``
          (see ``TestBestMetricsPersistence`` for the classification pattern).
    """

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
        metric: Literal["MAE", "MSE"],
    ):
        """Initialize the RegressionWrapper.

        Args:
            network: Network to wrap.
            cfg: Configuration.
            metric: Metric to use. Must be either 'MAE' or 'MSE'.
        """
        super().__init__(
            network=network,
            cfg=cfg,
        )
        if metric == "MAE":
            MetricClass = torchmetrics.MeanAbsoluteError
            LossMetricClass = torch.nn.L1Loss
        elif metric == "MSE":
            MetricClass = torchmetrics.MeanSquaredError
            LossMetricClass = torch.nn.MSELoss
        else:
            raise ValueError(f"Metric must be either 'MAE' or 'MSE'. Got {metric}.")

        # Other metrics
        self.train_metric = MetricClass()
        self.val_metric = MetricClass()
        self.test_metric = MetricClass()
        # Loss metric
        self.loss_metric = LossMetricClass()
        # Placeholders for logging of best train & validation values
        self.best_train_loss = 1e9
        self.best_val_loss = 1e9

    def _step(self, batch: dict[str, torch.Tensor], metric_calculator: torchmetrics.Metric):
        """Perform a step (either training, validation or test) and calculate the loss."""
        # Validate the structure of the batch
        assert isinstance(batch, dict), "Batch must be a dictionary"
        assert len(batch) == 3, "Batch must contain exactly 3 keys: 'input', 'label' and 'condition'"
        assert "input" in batch, "Batch must contain 'input' key"
        assert "label" in batch, "Batch must contain 'label' key"
        assert "condition" in batch, "Batch must contain 'condition' key"

        # Extract the label from the batch
        labels = batch.pop("label")

        output = self(input_and_condition=batch)  # Pass {input: x, condition: condition}

        assert isinstance(output, dict), "Output must be a dictionary"
        assert "logits" in output, "Output must contain 'logits' key"

        logits = output["logits"].contiguous()
        prediction = logits  # In regression, predictions are the logits

        # Calculate metric (use reshape instead of view for non-contiguous tensors)
        metric_calculator(prediction.reshape(-1), labels.reshape(-1))

        # Other outputs
        other_outputs = {}  # Not adding anything here for now, but we could add things to track per epoch, etc.

        # Calculate loss
        loss = self.loss_metric(prediction.reshape(-1), labels.reshape(-1))

        # Return predictions, loss and other outputs (contains logits and possibly other outputs such as token stats)
        return prediction, loss, other_outputs

    def training_step(self, batch, batch_idx):
        """Perform training step and log the training loss."""
        # Start timing (CUDA events)
        self._start_timing()
        # Perform step
        _predictions, loss, other_outputs = self._step(batch, self.train_metric)
        # Log loss
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        # Add other outputs to the list of other outputs. This is used for end of epoch logging.
        self.other_outputs_train.append(other_outputs)
        # Return loss
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step and log the validation loss."""
        # Perform step
        _predictions, loss, other_outputs = self._step(batch, self.val_metric)
        # Log loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        # Add other outputs to the list of other outputs. This is used for end of epoch logging.
        self.other_outputs_validation.append(other_outputs)
        # Return loss
        return loss

    def test_step(self, batch, batch_idx):
        """Perform a test step and log the test loss."""
        # Perform step
        _predictions, loss, _ = self._step(batch, self.test_metric)
        # Log loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)

    def on_train_epoch_end(self):
        """Log best train loss and logits over the training set."""
        train_step_outputs = self.other_outputs_train
        if len(train_step_outputs) == 0:
            # When autoresuming, the first epoch step outputs is empty, which would otherwise raise an error.
            # We add this here to avoid that error.
            return
        train_step_outputs_keys = train_step_outputs[0].keys()

        # Log the logits histogram
        if "logits" in train_step_outputs_keys:
            flattened_logits = torch.flatten(torch.cat([step_output["logits"] for step_output in train_step_outputs]))
            self.logger.experiment.log(
                {
                    "train/logits": wandb.Histogram(flattened_logits.to("cpu")),
                    "global_step": self.global_step,
                }
            )

        # Log other things if desired
        # .....

        # Clear the cache of other outputs for the next epoch
        self.other_outputs_train.clear()

        # Log best training loss
        train_loss = self.trainer.callback_metrics["train/loss_epoch"]
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss.item()
            self.logger.experiment.log(
                {
                    "train/best_loss": self.best_train_loss,
                    "global_step": self.global_step,
                }
            )

    def on_validation_epoch_end(self):
        """Log best validation loss and logits over the validation set."""
        if self.trainer.sanity_checking:
            self.other_outputs_validation.clear()
            return

        validation_step_outputs = self.other_outputs_validation
        validation_step_outputs_keys = validation_step_outputs[0].keys()

        # Log the logits histogram
        if "logits" in validation_step_outputs_keys:
            flattened_logits = torch.flatten(
                torch.cat([step_output["logits"] for step_output in validation_step_outputs])
            )
            self.logger.experiment.log(
                {
                    "val/logits": wandb.Histogram(flattened_logits.to("cpu")),
                    "val/logit_max_abs_value": flattened_logits.abs().max().item(),
                    "global_step": self.global_step,
                }
            )

        # Log other things if desired
        # .....

        # Clear the cache of other outputs for the next epoch
        self.other_outputs_validation.clear()

        # Log best validation loss
        val_loss = self.trainer.callback_metrics["val/loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.logger.experiment.log(
                {
                    "val/best_loss": self.best_val_loss,
                    "global_step": self.global_step,
                }
            )
