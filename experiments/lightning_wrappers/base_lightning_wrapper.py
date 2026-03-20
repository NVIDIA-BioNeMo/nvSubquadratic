# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Lightning wrappers for the Classification and Regression experiments."""

import warnings

import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.utilities import grad_norm

from experiments.default_cfg import (
    PLACEHOLDER,
    ExperimentConfig,
    SchedulerConfig,
)
from experiments.utils.checkpointing import align_compiled_keys
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.schedulers import ResumableSequentialLR


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
    valid_names = [PLACEHOLDER, "cosine", "constant", "wsd"]
    assert scheduler_cfg.name in valid_names, (
        f"scheduler_cfg.name must be one of {valid_names}. Got {scheduler_cfg.name}"
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

    # Validate eta_min < base lr to prevent nonsensical cosine schedules
    base_lr = optimizer.param_groups[0]["lr"]
    if scheduler_type == "cosine" and scheduler_cfg.eta_min >= base_lr:
        raise ValueError(
            f"eta_min ({scheduler_cfg.eta_min}) must be strictly less than the optimizer learning rate ({base_lr})."
        )

    # Build the warmup scheduler (shared across schedule types)
    warmup_scheduler = None
    if warmup_iterations != 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_iterations,
        )

    # Build the post-warmup scheduler
    if scheduler_type == "wsd":
        # Warmup-Stable-Decay: constant LR for stable phase, then linear decay
        stable_pct = getattr(scheduler_cfg, "stable_iterations_percentage", 0.0)
        assert 0.0 <= stable_pct < 1.0, f"stable_iterations_percentage must be in [0.0, 1.0). Got {stable_pct}"
        warmup_pct = warmup_iterations_percentage
        decay_pct = 1.0 - warmup_pct - stable_pct
        assert decay_pct > 0, (
            f"warmup ({warmup_pct}) + stable ({stable_pct}) must be < 1.0, leaving room for decay (got {decay_pct})"
        )
        stable_iterations = int(total_iterations * stable_pct)
        decay_iterations = total_iterations - warmup_iterations - stable_iterations

        schedulers = []
        milestones = []

        if warmup_scheduler is not None:
            schedulers.append(warmup_scheduler)
            milestones.append(warmup_iterations)

        # Stable phase: constant LR
        stable_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=stable_iterations,
        )
        schedulers.append(stable_scheduler)
        milestones.append(milestones[-1] + stable_iterations if milestones else stable_iterations)

        # Decay phase: linear ramp from base_lr to eta_min
        eta_min = scheduler_cfg.eta_min
        end_factor = max(eta_min / base_lr, 1e-8) if base_lr > 0 else 1e-8
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=end_factor,
            total_iters=decay_iterations,
        )
        schedulers.append(decay_scheduler)

        lr_scheduler = ResumableSequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
        )
        return lr_scheduler

    elif scheduler_type == "cosine":
        post_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_iterations - warmup_iterations,
            eta_min=scheduler_cfg.eta_min,
        )
    elif scheduler_type == "constant":
        post_warmup = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_iterations,
        )
    else:
        post_warmup = None
        warnings.warn(
            f"No scheduler will be used. cfg.train.scheduler = {scheduler_type}",
            stacklevel=2,
        )

    # Combine warmup + post-warmup via SequentialLR
    if warmup_scheduler is not None and post_warmup is not None:
        lr_scheduler = ResumableSequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, post_warmup],
            milestones=[warmup_iterations],
        )
    elif warmup_scheduler is not None:
        lr_scheduler = warmup_scheduler
    else:
        lr_scheduler = post_warmup

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

        # Timing tracking - uses CUDA events for accurate GPU timing
        self.timing_log_interval = 100  # Log every N steps
        self._timing_forward_accum = 0.0
        self._timing_backward_accum = 0.0
        self._timing_step_count = 0
        self._cuda_start_event = None
        self._cuda_forward_end_event = None
        self._cuda_backward_end_event = None

    @property
    def logger(self):
        """Return the first logger (Lightning 2.x uses a loggers list)."""
        loggers = getattr(self, "loggers", None)
        if loggers and len(loggers) > 0:
            return loggers[0]
        return getattr(super(), "logger", None)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Patch checkpoint for cross-optimizer and compiled/non-compiled resume.

        Handles three mismatch scenarios:

        1. **state_dict key prefixes** — ``torch.compile`` wraps modules under
           ``_orig_mod``, so checkpoint keys may differ from the live model.

        2. **current_model_state key prefixes** — when EMA is active, Lightning
           saves the raw training weights under ``current_model_state``.  The
           EMA callback later calls ``pl_module.load_state_dict(...)`` with
           these keys, so they must also be remapped.

        3. **optimizer param-group keys** — resuming with a different optimizer
           (e.g. Apex FusedLAMB vs torch_optimizer.Lamb) may require injecting
           default values for keys the new optimizer expects but the old
           checkpoint lacks (like ``bias_correction``, ``adam_w_mode``, etc.).
        """
        model_keys = set(self.state_dict().keys())

        # --- 1. state_dict key remapping ----------------------------------
        state_dict = checkpoint.get("state_dict")
        if state_dict is not None:
            checkpoint["state_dict"] = align_compiled_keys(state_dict, model_keys)

        # --- 2. current_model_state key remapping (EMA) -------------------
        current_model_state = checkpoint.get("current_model_state")
        if current_model_state is not None:
            checkpoint["current_model_state"] = align_compiled_keys(current_model_state, model_keys)

        # --- 3. optimizer param-group patching ----------------------------
        #
        # When resuming with a different optimizer (e.g. Apex FusedLAMB from
        # a torch_optimizer.Lamb checkpoint), the checkpoint's param groups
        # may be missing keys the new optimizer expects in its step().
        #
        # Strategy: construct a throwaway optimizer with the current config to
        # obtain its param groups with all keys correctly set, then inject any
        # missing keys into the checkpoint's groups.  This uses the *configured*
        # values (not just constructor defaults), so keys like max_grad_norm
        # that were explicitly overridden in the config are respected.
        #
        # Example: torch_optimizer.Lamb -> Apex FusedLAMB
        #
        #   Key              Injected value  Why correct
        #   ──────────────── ─────────────── ──────────────────────────────────
        #   bias_correction  True            Lamb applies it implicitly
        #   adam_w_mode      True            Both use decoupled weight decay
        #   max_grad_norm    0.0 (from cfg)  Avoids double-clipping with
        #                                    Lightning's grad_clip
        #   grad_averaging   True            FusedLAMB default (configured)
        #   set_grad_none    True            Memory opt, no semantic change
        #   use_nvlamb       False           Standard LAMB, not NVLAMB variant
        optimizer_states = checkpoint.get("optimizer_states")
        if optimizer_states is None:
            return

        try:
            reference_optim_dict = construct_optimizer(self, self.optimizer_cfg)
            ref_group = reference_optim_dict.param_groups[0]
        except Exception:
            return

        for opt_state in optimizer_states:
            for group in opt_state.get("param_groups", []):
                for key, val in ref_group.items():
                    if key not in group and key != "params":
                        group[key] = val

    def forward(self, input_and_condition: dict[str, torch.Tensor]):
        """Forward pass of the network.

        Args:
            input_and_condition: A dictionary containing the input and condition.
                Keys: "input" and "condition".

        Returns:
            The output of the network.
        """
        return self.network(input_and_condition)

    # =========================================================================
    # Timing utilities for forward/backward pass measurement
    # =========================================================================
    def _start_timing(self):
        """Start timing for forward pass using CUDA events."""
        if self.training and torch.cuda.is_available():
            self._cuda_start_event = torch.cuda.Event(enable_timing=True)
            self._cuda_forward_end_event = torch.cuda.Event(enable_timing=True)
            self._cuda_start_event.record()

    def _record_forward_end(self):
        """Record the end of forward pass."""
        if self._cuda_start_event is not None:
            self._cuda_forward_end_event.record()

    def _record_backward_end_and_accumulate(self):
        """Record backward end time and accumulate timing stats."""
        if self._cuda_start_event is not None:
            self._cuda_backward_end_event = torch.cuda.Event(enable_timing=True)
            self._cuda_backward_end_event.record()
            torch.cuda.synchronize()

            # Calculate times in milliseconds
            forward_time_ms = self._cuda_start_event.elapsed_time(self._cuda_forward_end_event)
            backward_time_ms = self._cuda_forward_end_event.elapsed_time(self._cuda_backward_end_event)

            self._timing_forward_accum += forward_time_ms
            self._timing_backward_accum += backward_time_ms
            self._timing_step_count += 1

            # Reset events
            self._cuda_start_event = None

    def _log_timing_if_needed(self):
        """Log accumulated timing stats every N steps."""
        if (
            self._timing_step_count > 0
            and self._timing_step_count % self.timing_log_interval == 0
            and self.logger is not None
        ):
            avg_forward_ms = self._timing_forward_accum / self._timing_step_count
            avg_backward_ms = self._timing_backward_accum / self._timing_step_count
            avg_total_ms = avg_forward_ms + avg_backward_ms

            self.log("timing/forward_ms", avg_forward_ms, prog_bar=False, sync_dist=self.distributed)
            self.log("timing/backward_ms", avg_backward_ms, prog_bar=False, sync_dist=self.distributed)
            self.log("timing/step_total_ms", avg_total_ms, prog_bar=False, sync_dist=self.distributed)
            self.log(
                "timing/throughput_steps_per_sec", 1000.0 / avg_total_ms, prog_bar=False, sync_dist=self.distributed
            )

            # Reset accumulators after logging
            self._timing_forward_accum = 0.0
            self._timing_backward_accum = 0.0
            self._timing_step_count = 0

    def on_before_backward(self, loss: torch.Tensor) -> None:
        """Called before backward pass - record forward end time."""
        self._record_forward_end()

    def on_after_backward(self) -> None:
        """Called after backward pass - record timing and log."""
        self._record_backward_end_and_accumulate()
        self._log_timing_if_needed()

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
        """Log the model architecture and parameter count to Weights & Biases once training starts."""
        super().on_fit_start()

        # Only log to WandB from the main process (rank 0) to avoid AttributeError
        # on non-rank-0 processes where logger.experiment is a dummy/function.
        if self.logger is not None and self.global_rank == 0:
            model_repr = str(self.network)
            # Log as HTML wrapped in <pre> to preserve formatting in the UI.
            self.logger.experiment.log(
                {
                    "model/architecture": wandb.Html(f"<pre>{model_repr}</pre>"),
                    "global_step": self.global_step,
                }
            )

            # Log parameter count to WandB config (appears in summary/overview)
            self.logger.experiment.config.update(
                {"model/num_params": self.num_params},
                allow_val_change=True,
            )

            # Also send to raw logs (stdout captured by W&B) and W&B terminal log
            self.print(f"Model architecture:\n{model_repr}")
            self.print(f"Total parameters: {self.num_params:,}")
            wandb.termlog(f"Model architecture:\n{model_repr}")
            wandb.termlog(f"Total parameters: {self.num_params:,}")

            # # Optionally watch the model to track gradients/parameters.
            # if hasattr(self.logger, "watch"):
            #     self.logger.watch(self.network, log="gradients", log_freq=100)
