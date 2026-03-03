# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Lightning wrappers for the Classification and Regression experiments."""

import torch
import torch.nn.functional as F
import torchmetrics
import wandb

from experiments.default_cfg import ExperimentConfig
from experiments.lightning_wrappers.base_lightning_wrapper import LightningWrapperBase


class SoftTargetCrossEntropy(torch.nn.Module):
    """Cross-entropy loss with soft targets (from DeiT III / timm).

    Works with Mixup/CutMix soft labels by computing -sum(target * log_softmax(logits))
    per sample, then averaging over the batch. Avoids the multi-class gradient dilution
    issue that occurs with BCEWithLogitsLoss(reduction='mean').
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-targets * F.log_softmax(logits, dim=-1), dim=-1)
        return loss.mean()


class ClassificationWrapper(LightningWrapperBase):
    """Lightning wrapper for classification tasks."""

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
        use_bce_loss: bool = False,
    ):
        """Initialize the ClassificationWrapper.

        Args:
            network: Network to wrap.
            cfg: Configuration.
            use_bce_loss: Whether to use BCEWithLogitsLoss for classification.
        """
        super().__init__(
            network=network,
            cfg=cfg,
        )
        # Other metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=network.out_proj.out_features)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=network.out_proj.out_features)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=network.out_proj.out_features)

        # Binary problem?
        self.multiclass = network.out_proj.out_features != 1

        # Loss metric
        self.use_bce_loss = use_bce_loss
        if self.multiclass and not self.use_bce_loss:
            self.loss_metric = torch.nn.CrossEntropyLoss()
        elif self.multiclass and self.use_bce_loss:
            self.loss_metric = SoftTargetCrossEntropy()
        else:
            self.loss_metric = torch.nn.BCEWithLogitsLoss()

        # Function to get predictions:
        if self.multiclass:
            self.get_predictions = self.multiclass_prediction
        else:
            self.get_predictions = self.binary_prediction

        # Placeholders for logging of best train & validation values
        self.best_train_acc = 0.0
        self.best_val_acc = 0.0
        self.best_train_loss = 1e9
        self.best_val_loss = 1e9

        # Suffix appended to validation metric names (set by EMA callbacks to "_ema")
        self._val_metric_suffix = ""

    def _step(
        self, batch: dict[str, torch.Tensor], accuracy_calculator: torchmetrics.Metric
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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

        logits = output["logits"].contiguous()  # [B, T, C]
        logits = logits.reshape(-1, logits.shape[-1])  # [B * seq_len, out_channels]

        # Handle labels based on their shape (hard indices vs soft probabilities)
        if labels.ndim > 1 and labels.shape[-1] == logits.shape[-1]:
            # Soft labels: [B, C] or [B, T, C] -> [B * T, C]
            labels = labels.reshape(-1, labels.shape[-1])
        else:
            # Hard labels: [B] or [B, T] -> [B * T]
            labels = labels.reshape(-1)

        # Predictions
        predictions = self.get_predictions(logits)

        # For multi-class classification, if the labels are float, we need to convert them to long for the accuracy calculator.
        # This is a workaround used during training to have accuracy calculations for training steps / epochs as well.
        if self.multiclass:
            if labels.dtype == torch.float and labels.ndim > 1:
                accuracy_calculator(predictions, torch.argmax(labels, dim=1))
            else:
                accuracy_calculator(predictions, labels)
        else:  # Binary classification
            accuracy_calculator(predictions, labels)
            labels = labels.float()

        # SoftTargetCrossEntropy requires soft probability targets [B, C].
        # During training, Mixup/CutMix already provides soft labels.
        # During validation/test, labels are integer indices and need one-hot conversion.
        if self.use_bce_loss and labels.ndim == 1:
            labels = torch.nn.functional.one_hot(labels.long(), num_classes=logits.shape[-1]).float()

        # Calculate the loss
        loss = self.loss_metric(logits, labels)

        # Not adding anything here for now, but we could add things to track per epoch, etc.
        other_outputs = {}

        # Return predictions and loss
        return predictions, loss, other_outputs

    def training_step(self, batch, batch_idx):
        """Perform a training step and log the training loss & accuracy."""
        # Perform step
        predictions, loss, other_outputs = self._step(batch, self.train_acc)
        # Log and return loss (Required in training step)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log(
            "train/acc",
            self.train_acc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        # Add other outputs to the list of other outputs. This is used for end of epoch logging.
        self.other_outputs_train.append(other_outputs)
        # Return loss
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step and log the validation loss & accuracy."""
        # Perform step
        predictions, loss, other_outputs = self._step(batch, self.val_acc)
        s = self._val_metric_suffix
        self.log(
            f"val/loss{s}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            f"val/acc{s}",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        # Add other outputs to the list of other outputs. This is used for end of epoch logging.
        self.other_outputs_validation.append(other_outputs)
        # Return loss
        return loss

    def test_step(self, batch, batch_idx):
        """Perform a test step and log the test loss & accuracy."""
        # Perform step
        predictions, loss, _ = self._step(batch, self.test_acc)
        # Log and return loss (Required in training step)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "test/acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )

    def on_train_epoch_end(self):
        """Log best training accuracy and loss and logits over the training set."""
        train_step_outputs = self.other_outputs_train
        if len(train_step_outputs) == 0:
            # When autoresuming, the first epoch step outputs is empty, which would otherwise raise an error.
            # We add this here to avoid that error.
            return
        train_step_outputs_keys = train_step_outputs[0].keys()

        # Log the logits histogram
        if "logits" in train_step_outputs_keys and self.logger is not None:
            flattened_logits = torch.flatten(torch.cat([step_output["logits"] for step_output in train_step_outputs]))
            self.logger.experiment.log(
                {
                    "train/logits": wandb.Histogram(flattened_logits),
                    "global_step": self.global_step,
                }
            )

        # Log other things if desired
        # .....

        # Clear the cache of other outputs for the next epoch
        self.other_outputs_train.clear()

        # Log best accuracy
        train_acc = self.trainer.callback_metrics["train/acc_epoch"]
        if train_acc > self.best_train_acc:
            self.best_train_acc = train_acc.item()
            if self.logger is not None:
                self.logger.experiment.log(
                    {
                        "train/best_acc": self.best_train_acc,
                        "global_step": self.global_step,
                    }
                )
        # Log best training loss
        train_loss = self.trainer.callback_metrics["train/loss_epoch"]
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss.item()
            if self.logger is not None:
                self.logger.experiment.log(
                    {
                        "train/best_loss": self.best_train_loss,
                        "global_step": self.global_step,
                    }
                )

    def on_validation_epoch_end(self):
        """Log best validation accuracy and loss and logits over the validation set."""
        # Sanity check runs only a few batches, so metrics are not
        # representative.  Skip best-metric tracking to avoid inflated values.
        if self.trainer.sanity_checking:
            self.other_outputs_validation.clear()
            return

        # Gather logits from validation set and construct a histogram of them.
        validation_step_outputs = self.other_outputs_validation
        validation_step_outputs_keys = validation_step_outputs[0].keys()

        if "logits" in validation_step_outputs_keys:
            flattened_logits = torch.flatten(
                torch.cat([step_output["logits"] for step_output in validation_step_outputs])
            )
            if self.logger is not None:
                self.logger.experiment.log(
                    {
                        "val/logits": wandb.Histogram(flattened_logits),
                        "global_step": self.global_step,
                    }
                )

        # Log other things if desired
        # .....

        # Clear the cache of other outputs for the next epoch
        self.other_outputs_validation.clear()

        # Log best accuracy
        s = self._val_metric_suffix
        val_acc = self.trainer.callback_metrics[f"val/acc{s}"]
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc.item()
            if self.logger is not None:
                self.logger.experiment.log(
                    {
                        f"val/best_acc{s}": self.best_val_acc,
                        "global_step": self.global_step,
                    }
                )
        # Log best validation loss
        val_loss = self.trainer.callback_metrics[f"val/loss{s}"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            if self.logger is not None:
                self.logger.experiment.log(
                    {
                        f"val/best_loss{s}": self.best_val_loss,
                        "global_step": self.global_step,
                    }
                )

    @staticmethod
    def multiclass_prediction(logits):
        """Predict the class with the highest logit for multi-class classification."""
        return torch.argmax(logits, 1)

    @staticmethod
    def binary_prediction(logits):
        """Predict the class with the highest logit for binary classification."""
        return (logits > 0.0).squeeze().long()
