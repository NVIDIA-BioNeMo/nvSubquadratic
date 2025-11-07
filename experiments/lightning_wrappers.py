# TODO: Add license header here


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

    def forward(self, x):
        """Forward pass of the network."""
        return self.network(x)

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

    def _step(self, batch, accuracy_calculator):
        """Perform a step (either training, validation or test) and calculate the loss."""
        x, labels = batch
        logits = self(x)
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
            logits = logits.view(-1)

        loss = self.loss_metric(logits, labels)
        # Return predictions and loss
        return predictions, logits, loss

    def training_step(self, batch, batch_idx):
        """Perform a training step and log the training loss & accuracy."""
        # Perform step
        predictions, logits, loss = self._step(batch, self.train_acc)
        # Log and return loss (Required in training step)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log(
            "train/acc",
            self.train_acc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        return {"loss": loss, "logits": logits.detach()}

    def validation_step(self, batch, batch_idx):
        """Perform a validation step and log the validation loss & accuracy."""
        # Perform step
        predictions, logits, loss = self._step(batch, self.val_acc)
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
        return logits  # used to log histograms in validation_epoch_step

    def test_step(self, batch, batch_idx):
        """Perform a test step and log the test loss & accuracy."""
        # Perform step
        predictions, _, loss = self._step(batch, self.test_acc)
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

    def on_train_epoch_end(self, train_step_outputs=None):
        """Log best training accuracy and loss and logits over the training set."""
        if train_step_outputs is not None:
            flattened_logits = torch.flatten(torch.cat([step_output["logits"] for step_output in train_step_outputs]))
            self.logger.experiment.log(
                {
                    "train/logits": wandb.Histogram(flattened_logits.to("cpu")),
                    "global_step": self.global_step,
                }
            )
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

    def on_validation_epoch_end(self, validation_step_outputs=None):
        """Log best validation accuracy and loss and logits over the validation set."""
        # Gather logits from validation set and construct a histogram of them.
        if validation_step_outputs is not None:
            flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
            self.logger.experiment.log(
                {
                    "val/logits": wandb.Histogram(flattened_logits.to("cpu")),
                    "val/logit_max_abs_value": flattened_logits.abs().max().item(),
                    "global_step": self.global_step,
                }
            )
        # Log best accuracy
        val_acc = self.trainer.callback_metrics["val/acc"]
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc.item()
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

    def _step(self, batch, metric_calculator):
        """Perform a step (either training, validation or test) and calculate the loss."""
        x, labels = batch
        prediction = self(x).contiguous()
        # Calculate loss
        metric_calculator(prediction, labels)
        loss = self.loss_metric(prediction, labels)
        # Return predictions and loss
        return prediction, loss

    def training_step(self, batch, batch_idx):
        """Perform training step and log the training loss."""
        # Perform step
        _, loss = self._step(batch, self.train_metric)
        # Log loss
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Perform a validation step and log the validation loss."""
        # Perform step
        predictions, loss = self._step(batch, self.val_metric)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        """Perform a test step and log the test loss."""
        # Perform step
        predictions, loss = self._step(batch, self.test_metric)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        return {"loss": loss}

    def on_train_epoch_end(self, train_step_outputs=None):
        """Log best train loss and logits over the training set."""
        if train_step_outputs is not None:
            flattened_logits = torch.flatten(torch.cat([step_output["logits"] for step_output in train_step_outputs]))
            self.logger.experiment.log(
                {
                    "train/logits": wandb.Histogram(flattened_logits.to("cpu")),
                    "global_step": self.global_step,
                }
            )
        # Log best accuracy
        train_loss = self.trainer.callback_metrics["train/loss_epoch"]
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss.item()
            self.logger.experiment.log(
                {
                    "train/best_loss": self.best_train_loss,
                    "global_step": self.global_step,
                }
            )

    def on_validation_epoch_end(self, validation_step_outputs=None):
        """Log best validation loss and logits over the validation set."""
        # Gather logits from validation set and construct a histogram of them.
        if validation_step_outputs is not None:
            flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
            self.logger.experiment.log(
                {
                    "val/logits": wandb.Histogram(flattened_logits.to("cpu")),
                    "val/logit_max_abs_value": flattened_logits.abs().max().item(),
                    "global_step": self.global_step,
                }
            )
        # Log best accuracy
        val_loss = self.trainer.callback_metrics["val/loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.logger.experiment.log(
                {
                    "val/best_loss": self.best_val_loss,
                    "global_step": self.global_step,
                }
            )


class L2RelativeError(torchmetrics.Metric):
    """Computes L2 relative error: ||pred - target|| / ||target||.

    This metric is commonly used in PDE tasks.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_relative_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric state with predictions and targets.

        Args:
            preds: Predictions of shape [B, ...]
            target: Targets of shape [B, ...]
        """
        # Flatten spatial dimensions: [B, ...] -> [B, N]
        batch_size = preds.shape[0]
        preds_flat = preds.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        # Compute L2 norms along spatial dimension
        diff_norm = torch.linalg.norm(preds_flat - target_flat, dim=1)  # [B]
        target_norm = torch.linalg.norm(target_flat, dim=1)  # [B]

        # Compute relative error
        relative_error = diff_norm / (target_norm + 1e-10)

        self.sum_relative_error += relative_error.sum()
        self.total += batch_size

    def compute(self):
        """Compute the mean L2 relative error."""
        return self.sum_relative_error / self.total


class PDERegressionWrapper(RegressionWrapper):
    """Lightning wrapper for PDE regression tasks with autoregressive rollout.

    Training uses 1-step prediction and MSE loss. 
    Validation/test use N-step autoregressive rollout and compute L2 relative error.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
        prev_steps: int,
        rollout_steps: int,
        metric: Literal["MAE", "MSE"] = "MSE",
    ):
        """Initialize the PDERegressionWrapper.

        Args:
            network: Network to wrap.
            cfg: Configuration.
            metric: Metric to use for loss (only used for training). Default is 'MSE'.
        """
        super().__init__(network=network, cfg=cfg, metric=metric)
        
        self.prev_steps = prev_steps
        self.rollout_steps = rollout_steps

        # L2 relative error metrics for val/test
        self.val_l2_error = L2RelativeError()
        self.test_l2_error = L2RelativeError()

    @torch.inference_mode()
    def autoregressive_rollout(self, initial_input, num_steps):
        """
        Perform autoregressive rollout prediction.
        It is done in inference mode to avoid tracking gradients.

        Args:
            initial_input: Initial input of shape [B, prev_steps, H, W, C]
            num_steps: Number of steps to roll out

        Returns:
            Predictions of shape [B, num_steps, H, W, C]
        """            
        b, t, h, w, c = initial_input.shape
        predictions = torch.empty((b, num_steps, h, w, c), device=initial_input.device)  # [B, num_steps, H, W, C]
        
        current_input = initial_input  # [B, prev_steps, H, W, C]

        for step_id in range(num_steps):
            # Flatten time into channels for model input
            b, t, h, w, c = current_input.shape
            model_input = current_input.permute(0, 2, 3, 1, 4).reshape(b, h, w, t * c)  # [B, H, W, prev_steps*C]

            # Predict next step
            pred = self(model_input).unsqueeze(1)  # [B, 1, H, W, C]
            predictions[:, step_id:step_id+1] = pred  # Store prediction

            # Update input: drop oldest timestep, append new prediction
            current_input = torch.cat([current_input[:, 1:], pred], dim=1)  # [B, prev_steps, H, W, C]

        return predictions

    def training_step(self, batch, batch_idx):
        """Training uses 1-step prediction with MSE loss.

        Args:
            batch: Tensor of shape [B, prev_steps + 1, H, W, C]
        """        
        b, t, h, w, c = batch.shape
        
        # Split: first prev_steps for input, next 1 for target
        x = batch[:, :-1]           # [B, prev_steps, H, W, C]
        target = batch[:, -1]       # [B, H, W, C]

        # Flatten time into channels for model input
        x = x.permute(0, 2, 3, 1, 4).reshape(b, h, w, t * c)  # [B, H, W, prev_steps*C]

        # Predict next timestep
        prediction = self(x)

        loss = self.loss_metric(prediction, target)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        
        return {"loss": loss}

    def eval_step(self, batch, batch_idx, *, stage: Literal["val", "test"]):
        """Evaluation uses N-step autoregressive rollout and L2 relative error.

        Args:
            batch: Tensor of shape [B, prev_steps + rollout_steps, H, W, C]
        """
        # Split: first prev_steps for input, remaining for ground truth
        initial_input = batch[:, :self.prev_steps]  # [B, prev_steps, H, W, C]
        ground_truth = batch[:, self.prev_steps:]   # [B, rollout_steps, H, W, C]
        
        # Perform autoregressive rollout
        predictions = self.autoregressive_rollout(initial_input, self.rollout_steps)

        # Compute L2 relative error
        self.val_l2_error(predictions, ground_truth)

        # Log L2 error
        self.log("val/loss", self.val_l2_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        
        return {}
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self.eval_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self.eval_step(batch, batch_idx, stage="test")

    def on_validation_epoch_end(self, validation_step_outputs=None):
        """Log best validation loss."""
        val_loss = self.trainer.callback_metrics["val/loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.logger.experiment.log(
                {
                    "val/best_loss": self.best_val_loss,
                    "global_step": self.global_step,
                }
            )
