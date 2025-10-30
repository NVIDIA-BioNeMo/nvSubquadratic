
# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Lightning wrappers for the Classification and Regression experiments."""

import math
from typing import Any, Literal, Optional

import torch.nn.functional as F
from torchvision.utils import make_grid

import copy

import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import OmegaConf
from pytorch_lightning.utilities import grad_norm

import wandb
from diffusers import DDIMScheduler
from experiments.default_cfg import DiffusionExperimentConfig, PLACEHOLDER, ExperimentConfig, SchedulerConfig
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

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._log_sanity_samples()

    def _log_sanity_samples(self, max_samples: int = 8) -> None:
        logger = getattr(self, "logger", None)
        if logger is None or not hasattr(logger, "experiment"):
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return

        loader = None
        for loader_name in ("sanity_dataloader", "val_dataloader", "train_dataloader"):
            if hasattr(datamodule, loader_name):
                try:
                    candidate = getattr(datamodule, loader_name)()
                except TypeError:
                    continue
                if candidate is not None:
                    loader = candidate
                    break
        if loader is None:
            return

        try:
            batch = next(iter(loader))
        except Exception:
            return

        if isinstance(batch, dict):
            inputs: Any = next((batch[key] for key in ("input", "image", "images") if key in batch), None)
        elif isinstance(batch, (list, tuple)):
            inputs = batch[0] if len(batch) > 0 else None
        else:
            inputs = batch

        if inputs is None:
            return

        with torch.no_grad():
            tensor = torch.as_tensor(inputs).detach().cpu()
            if tensor.ndim == 4:
                if tensor.shape[1] in (1, 3):
                    pass
                elif tensor.shape[-1] in (1, 3):
                    tensor = torch.moveaxis(tensor, -1, 1)
                else:
                    return
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            else:
                return

            tensor = tensor[: max_samples]
            if tensor.numel() == 0:
                return

            grid = make_grid(
                tensor,
                nrow=min(4, tensor.shape[0]),
                normalize=True,
                value_range=(-1.0, 1.0),
            )

        try:
            logger.experiment.log({"sanity/samples": wandb.Image(grid), "global_step": self.global_step})
        except Exception:
            return

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


class DiffusionWrapper(LightningWrapperBase):
    """Lightning module for DDPM/DDIM-style training with CKConv backbones."""

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: DiffusionExperimentConfig,
    ) -> None:
        super().__init__(network=network, cfg=cfg)

        if not isinstance(cfg, DiffusionExperimentConfig):
            raise TypeError("DiffusionWrapper requires cfg to be a DiffusionExperimentConfig instance.")
        diffusion_cfg = cfg.diffusion
        if diffusion_cfg is None:
            raise ValueError("DiffusionWrapper requires cfg.diffusion to be provided.")

        # Store diffusion hyper-parameters from the configuration so that we can reuse them
        # across training and sampling without constantly reaching outside the module.
        self.num_train_timesteps = int(diffusion_cfg.num_train_timesteps)
        self.beta_schedule = diffusion_cfg.beta_schedule
        self.beta_start = float(diffusion_cfg.beta_start)
        self.beta_end = float(diffusion_cfg.beta_end)

        # Set the prediction type and validate it.
        assert diffusion_cfg.prediction_type in ['epsilon', 'sample', 'v_prediction']
        self.prediction_type = diffusion_cfg.prediction_type

        # Instantiate the diffusers scheduler, delegating all diffusion math (alphas, betas, posteriors, etc.)
        # to a single well-tested implementation rather than maintaining our own copy here.
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            prediction_type=self.prediction_type,
            clip_sample=False,
            set_alpha_to_one=False,
        )

        hidden_dim = getattr(network, "hidden_dim", None)
        if hidden_dim is None:
            raise AttributeError("DiffusionWrapper requires the network to expose a 'hidden_dim' attribute.")

        schedule_time_embed = diffusion_cfg.time_embed_dim
        timestep_dim = int(schedule_time_embed) if schedule_time_embed is not None else hidden_dim * 2
        if timestep_dim % 2 != 0:
            timestep_dim += 1
        self.timestep_dim = timestep_dim
        self.max_period = float(diffusion_cfg.max_period)

        # Time conditioning pipeline: sinusoidal embedding followed by an MLP so the backbone always
        # receives conditioning in its native hidden dimension.
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.timestep_dim, hidden_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # We keep the objective expressed as a simple mean-squared error; the scheduler decides which
        # target we compare against (noise, clean sample, or velocity) and we simply follow along.
        self.loss_fn = torch.nn.MSELoss()

        # Book-keeping for sampling and logging.
        self.example_input_shape: Optional[torch.Size] = None
        self.default_inference_steps = int(diffusion_cfg.num_inference_steps)
        self.log_samples = bool(diffusion_cfg.log_samples)
        self.num_generated_samples = int(diffusion_cfg.num_samples)
        self.ddim_eta = float(diffusion_cfg.ddim_eta)

        # Exponential moving average (EMA) tracking mirrors the previous implementation; we only
        # modernise the diffusion math, not the stabilisation tricks that already work well.
        self.ema_enabled = bool(diffusion_cfg.ema_enabled)
        self.ema_decay = float(diffusion_cfg.ema_decay)
        self.ema_update_every = int(diffusion_cfg.ema_update_every)
        self.ema_warmup_steps = int(diffusion_cfg.ema_warmup_steps)
        self._ema_model: Optional[torch.nn.Module] = None
        self._ema_has_been_updated = False
        if self.ema_enabled:
            # Create an EMA shadow copy that never receives gradients so we can use it for evaluation
            # time sampling without polluting the main optimiser state.
            self._ema_model = copy.deepcopy(self.network)
            for p in self._ema_model.parameters():
                p.detach_()
                p.requires_grad_(False)

    @staticmethod
    def _channels_last_to_first(tensor: torch.Tensor) -> torch.Tensor:
        """Diffusers schedulers operate on channels-first tensors, so we convert on the fly."""
        return torch.moveaxis(tensor, -1, 1).contiguous()

    @staticmethod
    def _channels_first_to_last(tensor: torch.Tensor) -> torch.Tensor:
        """Convert back to channels-last so the backbone can keep using its preferred convention."""
        return torch.moveaxis(tensor, 1, -1).contiguous()

    def _compute_training_target(
        self,
        clean_images_bchw: torch.Tensor,
        noise_bchw: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Let the scheduler tell us which training target corresponds to the configured objective."""
        prediction_type = self.scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise_bchw
        elif prediction_type == "sample":
            target = clean_images_bchw
        elif prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(clean_images_bchw, noise_bchw, timesteps)
        else:  # pragma: no cover - guarded by configuration validation
            raise ValueError(f"Unsupported prediction type: {prediction_type}")
        return target

    def _timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Standard sinusoidal embedding with configurable dimensionality, identical to the previous
        # implementation so we preserve conditioning behaviour.
        device = timesteps.device
        half_dim = self.timestep_dim // 2
        exponent = torch.arange(half_dim, device=device, dtype=torch.float32)
        exponent = -math.log(self.max_period) * exponent / max(half_dim - 1, 1)
        freqs = torch.exp(exponent)
        args = timesteps.float().view(-1, 1) * freqs.view(1, -1)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if embedding.shape[-1] < self.timestep_dim:
            embedding = F.pad(embedding, (0, self.timestep_dim - embedding.shape[-1]))
        return embedding

    def _condition_from_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Feed the sinusoidal embedding through the learnable MLP so the denoiser can ingest it.
        emb = self._timestep_embedding(timesteps)
        return self.time_mlp(emb)

    def _shared_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Inputs arrive in channels-last format from the datamodule; we keep that convention for the
        # network but convert to channels-first whenever diffusers expects it.
        images = batch["input"].to(self.device)
        if self.example_input_shape is None:
            # Cache the tensor shape so we can initialise random noise for sampling later on.
            self.example_input_shape = images.shape[1:]

        images_bchw = self._channels_last_to_first(images)
        batch_size = images_bchw.shape[0]

        # Sample an independent diffusion timestep for each element in the batch.
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=images_bchw.device,
            dtype=torch.long,
        )

        # Draw standard Gaussian noise and let diffusers corrupt the clean image for us.
        noise_bchw = torch.randn_like(images_bchw)
        noisy_images_bchw = self.scheduler.add_noise(images_bchw, noise_bchw, timesteps)
        noisy_images = self._channels_first_to_last(noisy_images_bchw)

        # Time-conditional forward pass through the backbone.
        condition = self._condition_from_timesteps(timesteps)
        
        # The denoiser returns all of its outputs in a dict; training only needs the raw logits tensor.
        prediction = self.network({"input": noisy_images, "condition": condition})["logits"]

        # Convert prediction to channels-first for loss computation.
        prediction_bchw = self._channels_last_to_first(prediction)

        # Compute the appropriate training target and return the MSE loss.
        target = self._compute_training_target(images_bchw, noise_bchw, timesteps)
        loss = self.loss_fn(prediction_bchw, target)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        return loss

    def test_step(self, batch, batch_idx):
        # Generation experiments do not have a dedicated test metric.
        pass

    @torch.no_grad()
    def sample(self, num_samples: int, num_inference_steps: Optional[int] = None) -> torch.Tensor:
        if self.example_input_shape is None:
            raise RuntimeError("Cannot sample before observing at least one training batch.")

        num_inference_steps = num_inference_steps or self.default_inference_steps
        device = self.device
        height, width, channels = self.example_input_shape

        # Start from pure Gaussian noise in channels-first format because that's what the scheduler expects.
        sample_bchw = torch.randn((num_samples, channels, height, width), device=device)

        # Prepare the scheduler timesteps on the current device – this mirrors the standard diffusers pipeline.
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        use_ema = self.ema_enabled and self._ema_model is not None and self._ema_has_been_updated
        inference_model = self._ema_model if use_ema else self.network
        was_training = inference_model.training
        inference_model.eval()
        inference_model = inference_model.to(device)

        for timestep in self.scheduler.timesteps:
            # Broadcast the scalar timestep to a batch so we can embed it and feed the denoiser.
            t_batch = torch.full((num_samples,), timestep.item(), device=device, dtype=torch.long)
            condition = self._condition_from_timesteps(t_batch)

            # Convert the working sample back to channels-last before asking the network for a prediction.
            model_input = self._channels_first_to_last(sample_bchw)
            
            # As during training, we only consume the logits prediction emitted by the denoiser.
            outputs = inference_model({"input": model_input, "condition": condition})["logits"]
            model_output_bchw = self._channels_last_to_first(outputs)

            # One DDIM step brings us closer to the clean sample; eta tunes deterministic vs. stochastic paths.
            scheduler_output = self.scheduler.step(
                model_output_bchw,
                timestep,
                sample_bchw,
                eta=self.ddim_eta,
                return_dict=True,
            )
            sample_bchw = scheduler_output.prev_sample

        if was_training:
            inference_model.train()

        sample_hwc = self._channels_first_to_last(sample_bchw)
        return torch.clamp(sample_hwc, -1.0, 1.0)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.ema_enabled and self._ema_model is not None:
            self._ema_model.to(self.device)
            self._ema_model.eval()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)

        if (
            self.ema_enabled
            and self._ema_model is not None
            and self.global_step >= self.ema_warmup_steps
            and (self.global_step % self.ema_update_every == 0)
        ):
            with torch.no_grad():
                decay = self.ema_decay
                for ema_param, param in zip(self._ema_model.parameters(), self.network.parameters()):
                    ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
                for ema_buffer, buffer in zip(self._ema_model.buffers(), self.network.buffers()):
                    if ema_buffer.shape != buffer.shape:
                        ema_buffer.resize_as_(buffer)
                    ema_buffer.copy_(buffer)
                self._ema_has_been_updated = True

    def on_validation_epoch_end(self, outputs=None):
        if not self.log_samples:
            return
        if self.example_input_shape is None:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        num_samples = int(self.num_generated_samples)
        samples = self.sample(num_samples=num_samples)

        # Unnormalize the obtained samples.
        samples = self.trainer.datamodule.unnormalize(samples)

        samples_bchw = self._channels_last_to_first(samples)
        grid = make_grid(
            samples_bchw.detach().cpu(),
            nrow=max(1, int(math.sqrt(num_samples)))
        )
        self.logger.experiment.log(
            {
                "val/samples": wandb.Image(grid.cpu()),
                "global_step": self.global_step,
            }
        )
