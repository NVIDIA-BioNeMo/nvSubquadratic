# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Lightning wrappers for the Classification and Regression experiments."""

from typing import Literal

import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import OmegaConf
from pytorch_lightning.utilities import grad_norm

import wandb
from experiments.default_cfg import PLACEHOLDER, ExperimentConfig, SchedulerConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules import schedulers


def construct_optimizer(
    model,
    optimizer_cfg: LazyConfig,
):
    """Constructs an optimizer for a given model given a configuration.

    Args:
        model: a list of parameters to be trained
        optimizer_cfg (LazyConfig): The optimizer configuration.

    Returns:
        torch.optim.Optimizer: The constructed optimizer.
    """
    # Create parameter groups based on weight decay flag
    # IMPORTANT: Avoid duplicates by iterating parameters ONCE at the top level
    # and tracking by object identity (id(param)).
    wd_params: list[torch.nn.Parameter] = []
    no_wd_params: list[torch.nn.Parameter] = []
    seen_param_ids: set[int] = set()

    for name, param in model.named_parameters(recurse=True):
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_param_ids:
            continue
        seen_param_ids.add(pid)
        if getattr(param, "_no_weight_decay", False):
            no_wd_params.append(param)
        else:
            wd_params.append(param)

    # Safety: ensure no overlaps and no duplicates
    assert len(seen_param_ids) == len(set(map(id, wd_params))) + len(set(map(id, no_wd_params))), (
        "Optimizer param group mismatch: duplicate parameters across groups or some trainable "
        "parameters were not assigned. Every requires_grad=True parameter must appear in exactly one group."
    )

    # Create parameter groups with appropriate weight decay
    parameters = [
        {"params": wd_params, "weight_decay": optimizer_cfg.weight_decay},
        {"params": no_wd_params, "weight_decay": 0.0},
    ]

    # OmegaConf has problems with non-serializable objects. To instantiate the optimizer, we need to do the following:
    # 1. Convert the optimizer config to a dictionary
    # 2. Import the optimizer class
    # 3. Instantiate the optimizer

    # 1. Convert the optimizer config to a dictionary
    _optim_cfg = OmegaConf.to_container(optimizer_cfg, resolve=True)

    # 2. Import the optimizer class
    _optimizer_cls = _optim_cfg.pop("__target__")
    module_path, class_name = _optimizer_cls.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    _optimizer_cls = getattr(module, class_name)

    # 3. Instantiate the optimizer with wd=0. Weight decay is calculated over the generated kernels.
    _optim_cfg["params"] = parameters
    optimizer = _optimizer_cls(**_optim_cfg)

    return optimizer


def construct_scheduler(
    optimizer,
    scheduler_cfg: SchedulerConfig,
):
    """Creates a learning rate scheduler for a given optimizer given a configuration.

    Args:
        optimizer: the optimizer to be used
        scheduler_cfg (SchedulerConfig): The scheduler configuration.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The constructed scheduler.
    """
    assert scheduler_cfg.name in [PLACEHOLDER, "cosine"], (
        f"scheduler_cfg.name must be either {PLACEHOLDER} or 'cosine'. Got {scheduler_cfg.name}"
    )
    if scheduler_cfg.name != PLACEHOLDER:
        assert scheduler_cfg.total_iterations != PLACEHOLDER, (
            f"scheduler_cfg.total_iterations must be set when scheduler_cfg.name is not {PLACEHOLDER}"
        )

    # Unpack values from scheduler_cfg
    scheduler_type = scheduler_cfg.name
    warmup_iterations_percentage = scheduler_cfg.warmup_iterations_percentage
    total_iterations = scheduler_cfg.total_iterations

    # Interpret fractional warmup as a percentage of total iterations
    assert warmup_iterations_percentage >= 0.0 and warmup_iterations_percentage < 1.0, (
        f"scheduler_cfg.warmup_iterations_percentage must be in [0.0, 1.0). Got {warmup_iterations_percentage}"
    )
    warmup_iterations = int(total_iterations * warmup_iterations_percentage)

    # Create warm_up scheduler
    if warmup_iterations != 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_iterations,
        )
    else:
        warmup_scheduler = None

    # Create main scheduler
    if scheduler_type == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_iterations - warmup_iterations,
            last_epoch=-warmup_iterations,
        )
    else:
        lr_scheduler = None
        print(f"WARNING! No scheduler will be used. cfg.train.scheduler = {scheduler_type}")

    # Concatenate schedulers if required
    if warmup_scheduler is not None:
        # If both schedulers are defined, concatenate them
        if lr_scheduler is not None:
            lr_scheduler = schedulers.ChainedScheduler(
                [
                    warmup_scheduler,
                    lr_scheduler,
                ]
            )
        # Otherwise, return only the warmup scheduler
        else:
            lr_scheduler = warmup_scheduler

    return lr_scheduler


class LightningWrapperBase(pl.LightningModule):
    """Base Lightning wrapper class."""

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
    ):
        """Initialize the LightningWrapperBase.

        Args:
            network: Network to wrap.
            cfg: Configuration.
        """
        super().__init__()
        # Define network
        self.network = network

        # Save optimizer & scheduler parameters
        self.optimizer_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler

        # Explicitly define whether we are in distributed mode.
        self.distributed = torch.cuda.device_count() > 1

        # Calculate the number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        self.num_params = num_params

        # Gradient tracking configuration
        self.should_track_grad_norm = cfg.train.track_grad_norm > 0
        self.grad_norm_interval = cfg.train.track_grad_norm

        # Placeholder for other outputs from the training and validation steps.
        self.other_outputs_train = []
        self.other_outputs_validation = []

    def forward(self, input_and_condition: dict[str, torch.Tensor]):
        """Forward pass of the network.

        Args:
            input_and_condition: A dictionary containing the input and condition.
                Keys: "input" and "condition".

        Returns:
            The output of the network.
        """
        return self.network(input_and_condition)

    def configure_optimizers(self):
        """Configure the optimizer and scheduler for training."""
        # Construct optimizer & scheduler
        optimizer = construct_optimizer(
            model=self,
            optimizer_cfg=self.optimizer_cfg,
        )
        scheduler = construct_scheduler(
            optimizer=optimizer,
            scheduler_cfg=self.scheduler_cfg,
        )
        # Construct output dictionary
        optim_dict = {"optimizer": optimizer}
        if scheduler is not None:
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
            }
        # Return
        return optim_dict

    def on_before_optimizer_step(self, optimizer):
        """Log the gradient norm before the optimizer step every `grad_norm_interval` steps."""
        if self.should_track_grad_norm and self.global_step % self.grad_norm_interval == 0:
            self.log_dict(grad_norm(self, norm_type=2))

    def on_fit_start(self):
        """Log the model architecture to Weights & Biases once training starts."""
        if self.logger is not None:
            model_repr = str(self.network)
            # Log as HTML wrapped in <pre> to preserve formatting in the UI.
            self.logger.experiment.log(
                {
                    "model/architecture": wandb.Html(f"<pre>{model_repr}</pre>"),
                    "global_step": self.global_step,
                }
            )
            # Also send to raw logs (stdout captured by W&B) and W&B terminal log
            self.print(f"Model architecture:\n{model_repr}")
            wandb.termlog(f"Model architecture:\n{model_repr}")
            # # Optionally watch the model to track gradients/parameters.
            # if hasattr(self.logger, "watch"):
            #     self.logger.watch(self.network, log="gradients", log_freq=100)


class ClassificationWrapper(LightningWrapperBase):
    """Lightning wrapper for classification tasks."""

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
    ):
        """Initialize the ClassificationWrapper.

        Args:
            network: Network to wrap.
            cfg: Configuration.
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
        if self.multiclass:
            self.loss_metric = torch.nn.CrossEntropyLoss()
        else:
            self.loss_metric = torch.nn.BCEWithLogitsLoss()  # TODO: Required?
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

        # Validate the structure of the batch and pass to the model
        assert len(batch) == 2, "Batch must contain exactly 2 keys: 'input' and 'condition'"
        output = self(input_and_condition=batch)  # Pass {input: x, condition: condition}

        assert isinstance(output, dict), "Output must be a dictionary"
        assert "logits" in output, "Output must contain 'logits' key"

        logits = output["logits"].contiguous()  # [B, T, C]
        logits = logits.reshape(-1, logits.shape[-1])  # [B * seq_len, out_channels]
        labels = labels.reshape(-1)  # [B * seq_len]

        # Predictions
        predictions = self.get_predictions(logits)
        # For multi-class classification, if the labels are float, we need to convert them to long for the accuracy calculator.
        # This is a workaround used during training to have accuracy calculations for training steps / epochs as well.
        if self.multiclass:
            if labels.dtype == torch.float:
                accuracy_calculator(predictions, torch.argmax(labels, dim=1))
            else:
                accuracy_calculator(predictions, labels)
        else:  # Binary classification
            accuracy_calculator(predictions, labels)
            labels = labels.float()

        # Calculate the loss
        loss = self.loss_metric(logits, labels)

        other_outputs = {}  # Not adding anything here for now, but we could add things to track per epoch, etc.

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
        # Log and return loss (Required in training step)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "val/acc",
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
        if "logits" in train_step_outputs_keys:
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
            self.logger.experiment.log(
                {
                    "train/best_loss": self.best_train_loss,
                    "global_step": self.global_step,
                }
            )

    def on_validation_epoch_end(self):
        """Log best validation accuracy and loss and logits over the validation set."""
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
        val_acc = self.trainer.callback_metrics["val/acc"]
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc.item()
            if self.logger is not None:
                self.logger.experiment.log(
                    {
                        "val/best_acc": self.best_val_acc,
                        "global_step": self.global_step,
                    }
                )
        # Log best validation loss
        val_loss = self.trainer.callback_metrics["val/loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            if self.logger is not None:
                self.logger.experiment.log(
                    {
                        "val/best_loss": self.best_val_loss,
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


class RegressionWrapper(LightningWrapperBase):
    """Lightning wrapper for regression tasks."""

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

        # Validate the structure of the batch and pass to the model
        assert len(batch) == 2, "Batch must contain exactly 2 keys: 'input' and 'condition'"
        output = self(input_and_condition=batch)  # Pass {input: x, condition: condition}

        assert isinstance(output, dict), "Output must be a dictionary"
        assert "logits" in output, "Output must contain 'logits' key"

        logits = output["logits"].contiguous()
        prediction = logits  # In regression, predictions are the logits

        # Calculate metric
        metric_calculator(prediction.view(-1), labels.view(-1))

        # Other outputs
        other_outputs = {}  # Not adding anything here for now, but we could add things to track per epoch, etc.

        # Calculate loss
        loss = self.loss_metric(prediction.view(-1), labels.view(-1))

        # Return predictions, loss and other outputs (contains logits and possibly other outputs such as token stats)
        return prediction, loss, other_outputs

    def training_step(self, batch, batch_idx):
        """Perform training step and log the training loss."""
        # Perform step
        predictions, loss, other_outputs = self._step(batch, self.train_metric)
        # Log loss
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        # Add other outputs to the list of other outputs. This is used for end of epoch logging.
        self.other_outputs_train.append(other_outputs)
        # Return loss
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step and log the validation loss."""
        # Perform step
        predictions, loss, other_outputs = self._step(batch, self.val_metric)
        # Log loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        # Add other outputs to the list of other outputs. This is used for end of epoch logging.
        self.other_outputs_validation.append(other_outputs)
        # Return loss
        return loss

    def test_step(self, batch, batch_idx):
        """Perform a test step and log the test loss."""
        # Perform step
        predictions, loss, _ = self._step(batch, self.test_metric)
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
