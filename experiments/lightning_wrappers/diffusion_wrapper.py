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

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Lightning wrapper for continuous-time diffusion (JiT-style) experiments.

Provides :class:`DiffusionWrapper`, which implements the flow-matching / JiT
training loop: noises inputs according to a time-dependent schedule, trains a
denoiser network, and generates samples via an ODE integrator at inference time.

Adapted from https://github.com/implicit-long-convs/ccnn_v2.
"""

import copy
import math
import os
import shutil
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from experiments.default_cfg import DiffusionExperimentConfig
from experiments.lightning_wrappers.base_lightning_wrapper import LightningWrapperBase


class DiffusionWrapper(LightningWrapperBase):
    """Lightning module for JiT-style continuous-time diffusion."""

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: DiffusionExperimentConfig,
    ) -> None:
        """Initialize the DiffusionWrapper.

        Args:
            network: The neural network to be used as the denoiser model.
            cfg: The diffusion experiment configuration.
        """
        super().__init__(network=network, cfg=cfg)

        if not isinstance(cfg, DiffusionExperimentConfig):
            raise TypeError("DiffusionWrapper requires cfg to be a DiffusionExperimentConfig instance.")
        diffusion_cfg = cfg.diffusion
        if diffusion_cfg is None:
            raise ValueError("DiffusionWrapper requires cfg.diffusion to be provided.")

        # JiT continuous-time diffusion setup.
        self.num_train_timesteps = int(diffusion_cfg.num_train_timesteps)
        self.noise_scale = float(diffusion_cfg.noise_scale)
        self.p_mean = float(diffusion_cfg.p_mean)
        self.p_std = float(diffusion_cfg.p_std)
        self.cfg_interval_start = float(diffusion_cfg.cfg_interval_start)
        self.cfg_interval_end = float(diffusion_cfg.cfg_interval_end)

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

        # Sampling and logging configuration.
        self.example_input_shape: Optional[torch.Size] = None
        self.default_inference_steps = int(diffusion_cfg.num_inference_steps)
        self.log_samples = bool(diffusion_cfg.log_samples)
        self.num_generated_samples = int(diffusion_cfg.num_samples)

        # Optional online FID evaluation (JiT Style).
        self.fid_online_jit = bool(getattr(diffusion_cfg, "fid_online_jit", False))
        self.fid_stats_file = getattr(diffusion_cfg, "fid_stats_file", None)
        self.fid_num_samples = int(getattr(diffusion_cfg, "fid_num_samples", 50000))
        self.fid_interval = int(getattr(diffusion_cfg, "fid_interval", 50))
        self.fid_batch_size = int(getattr(diffusion_cfg, "fid_batch_size", 64))
        fid_steps_cfg = getattr(diffusion_cfg, "fid_num_inference_steps", None)
        self.fid_num_inference_steps = (
            int(fid_steps_cfg) if fid_steps_cfg is not None else self.default_inference_steps
        )

        # Classifier-free guidance (CFG) settings.
        # Configurable to support both conditional and unconditional training within the same wrapper.
        self.class_conditioning = diffusion_cfg.num_classes is not None
        self.cfg_enabled = bool(diffusion_cfg.use_classifier_free_guidance) and self.class_conditioning
        self.guidance_scale = float(diffusion_cfg.guidance_scale)

        # Dropout probability for the conditioning signal during training.
        self.condition_dropout_prob = float(diffusion_cfg.condition_dropout_prob) if self.class_conditioning else 0.0
        self.num_classes: Optional[int] = (
            int(diffusion_cfg.num_classes) if diffusion_cfg.num_classes is not None else None
        )

        if diffusion_cfg.use_classifier_free_guidance and not self.class_conditioning:
            raise ValueError(
                "Classifier-free guidance requires 'diffusion.num_classes' to be set so labels can be embedded."
            )

        if self.class_conditioning:
            if self.num_classes is None or self.num_classes <= 0:
                raise ValueError("diffusion.num_classes must be a positive integer when enabling class conditioning.")
            # We dedicate one additional embedding slot to represent the unconditional branch.
            # This embedding is learned during training (initialized to zero).
            self.null_label_index = self.num_classes
            self.label_embed = torch.nn.Embedding(self.num_classes + 1, hidden_dim)
            torch.nn.init.normal_(self.label_embed.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                # Initialize the unconditional embedding to zero.
                self.label_embed.weight[self.null_label_index].zero_()
        else:
            self.null_label_index = None
            self.label_embed = None

        # Exponential Moving Average (EMA) tracking.
        self.ema_enabled = bool(diffusion_cfg.ema_enabled)
        self.ema_decay = float(diffusion_cfg.ema_decay)
        self.ema_update_every = int(diffusion_cfg.ema_update_every)
        self.ema_warmup_steps = int(diffusion_cfg.ema_warmup_steps)
        self._ema_model: Optional[torch.nn.Module] = None
        self._ema_has_been_updated = False
        if self.ema_enabled:
            # Create a shadow copy of the network that does not receive gradients.
            self._ema_model = copy.deepcopy(self.network)
            for p in self._ema_model.parameters():
                p.detach_()
                p.requires_grad_(False)

        # Allow Hugging Face-backed models to register themselves for callbacks.
        register_fn = getattr(network, "hf_register_diffusion_wrapper", None)
        if callable(register_fn):
            register_fn(self)

    @staticmethod
    def _channels_last_to_first(tensor: torch.Tensor) -> torch.Tensor:
        """Convert an image tensor from channels-last to channels-first layout."""
        return torch.moveaxis(tensor, -1, 1).contiguous()

    @staticmethod
    def _channels_first_to_last(tensor: torch.Tensor) -> torch.Tensor:
        """Convert an image tensor from channels-first to channels-last layout."""
        return torch.moveaxis(tensor, 1, -1).contiguous()

    def _timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Standard sinusoidal embedding with configurable dimensionality, identical to the previous
        implementation so we preserve conditioning behaviour.

        Args:
            timesteps: a 1-D Tensor of N indices, one per batch element.
        """
        device = timesteps.device
        if timesteps.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            working = timesteps.to(torch.float32)
            embed_dtype = torch.float32
        else:
            working = timesteps
            embed_dtype = timesteps.dtype

        half_dim = self.timestep_dim // 2
        exponent = torch.arange(half_dim, device=device, dtype=torch.float32)
        exponent = -math.log(self.max_period) * exponent / max(half_dim - 1, 1)
        freqs = torch.exp(exponent).to(working.dtype)
        args = working.view(-1, 1) * freqs.view(1, -1)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(embed_dtype)
        if embedding.shape[-1] < self.timestep_dim:
            embedding = F.pad(embedding, (0, self.timestep_dim - embedding.shape[-1]))
        return embedding

    def _noiselevel_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed continuous timesteps t ∈ [0, 1] into the model hidden dimension."""
        target_dtype = self.time_mlp[0].weight.dtype
        # Scale t into the sinusoidal range expected by time_mlp (matching JiT continuous-time).
        scaled_t = timesteps.to(target_dtype).clamp_(0.0, 1.0) * float(self.num_train_timesteps)
        return self.time_mlp(self._timestep_embedding(scaled_t))

    def _condition_from_timesteps(
        self,
        timesteps: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        unconditional: bool = False,
        dropout_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return the conditioning vector and raw label embedding for a batch of timesteps.

        Returns a tuple ``(combined_condition, label_emb)`` where ``label_emb`` is the raw class
        embedding before adding the time embedding. Pass ``label_emb`` to the network as
        ``"class_emb"`` so that in-context tokens receive a pure class signal (no time information),
        matching the JiT reference. ``label_emb`` is ``None`` when class conditioning is disabled.

        Args:
            timesteps: Diffusion timestep indices sampled for each element in the batch.
            labels: Optional class labels associated with the batch. Required when class conditioning
                is enabled because each label switches us to a different guidance direction.
            unconditional: When ``True`` we force the method to emit the unconditional embedding by
                routing all samples to the extra "null" slot in ``self.label_embed``.
            dropout_mask: Boolean mask selecting which labels should be dropped for classifier-free
                guidance training. Elements set to ``True`` fall back to the unconditional embedding.
        """
        # Step 1: obtain the base time embedding as usual.
        time_emb = self._noiselevel_embedding(timesteps)

        # Without class conditioning the timestep embedding is the entire conditioning signal.
        if self.label_embed is None:
            return time_emb, None

        # If we do expect labels, make sure the caller provided them.
        if labels is None:
            if unconditional:
                # During sampling we sometimes request unconditional guidance without providing the
                # original labels, so we create a tensor filled with the null label index on demand.
                labels_to_embed = torch.full_like(timesteps, self.null_label_index, dtype=torch.long)
            else:
                raise ValueError(
                    "Class conditioning requested but no labels were provided. "
                    "Ensure the datamodule keeps labels (drop_labels=False) and the caller forwards them."
                )
        else:
            labels_to_embed = labels.to(timesteps.device, dtype=torch.long).view(-1)

        # Clone to avoid in-place edits that would leak outwards.
        labels_to_embed = labels_to_embed.clone()

        if unconditional:
            labels_to_embed.fill_(self.null_label_index)

        if dropout_mask is not None:
            if dropout_mask.shape != labels_to_embed.shape:
                raise ValueError("dropout_mask must match the shape of the labels tensor.")
            labels_to_embed[dropout_mask] = self.null_label_index

        if (labels_to_embed < 0).any():
            raise ValueError("Encountered negative labels while class conditioning; check datamodule configuration.")
        if (labels_to_embed > self.null_label_index).any():
            raise ValueError("Label index out of range for classifier-free guidance.")

        label_emb = self.label_embed(labels_to_embed)
        return time_emb + label_emb, label_emb

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_clean_images: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
        """Shared logic for training and validation steps.

        Args:
            batch: A datamodule batch containing at least the "input" key with clean images.
            return_clean_images: When ``True``, the method returns a tuple containing the loss
                and a dict with clean images for FID computation. Used during validation only.

        Returns:
            - During training: The computed loss tensor.
            - During validation with ``return_clean_images=True``: A tuple of the loss tensor and
              a dict with clean images under the "clean_images_bchw" key and optional labels.
        """
        # Inputs arrive in channels-last format from the datamodule; we keep that convention for the
        # network but convert to channels-first whenever diffusers expects it.
        images = batch["input"].to(self.device)
        if self.example_input_shape is None:
            # Cache the tensor shape so we can initialise random noise for sampling later on.
            self.example_input_shape = images.shape[1:]

        images_bchw = self._channels_last_to_first(images)
        batch_size = images_bchw.shape[0]

        labels_tensor: Optional[torch.Tensor] = None
        if self.class_conditioning:
            # Retrieve labels if the datamodule provided them. For class-conditioned runs the labels are
            # essential; otherwise we quietly fall back to the unconditional path.
            if "label" not in batch:
                raise RuntimeError(
                    "Class conditioning requires datamodule batches to include 'label'. "
                    "Set drop_labels=False on the datamodule to keep them."
                )
            labels_tensor = batch["label"].to(self.device, non_blocking=True).long().view(-1)

        # 1. Sample timestep t in [0, 1] from the JiT logit-normal distribution.
        t_logit = torch.randn(batch_size, device=self.device) * self.p_std + self.p_mean
        timesteps = torch.sigmoid(t_logit)

        # 2. Sample noise scaled by the configured noise_scale (matches JiT denoiser.py).
        eps_bchw = torch.randn_like(images_bchw) * self.noise_scale

        # 3. Mix clean image and noise to get z_t.
        t_b = timesteps.view(batch_size, 1, 1, 1)
        z_bchw = t_b * images_bchw + (1.0 - t_b) * eps_bchw
        target_v = images_bchw - eps_bchw

        # 4. Predict x from z_t and conditioning.
        z_cl = self._channels_first_to_last(z_bchw)
        dropout_mask = None
        if self.class_conditioning and self.condition_dropout_prob > 0.0:
            dropout_mask = torch.rand(batch_size, device=self.device) < self.condition_dropout_prob

        condition, class_emb = self._condition_from_timesteps(
            timesteps,
            labels=labels_tensor,
            dropout_mask=dropout_mask,
        )
        net_input = {"input": z_cl, "condition": condition}
        if class_emb is not None:
            net_input["class_emb"] = class_emb
        prediction = self.network(net_input)["logits"]
        prediction_bchw = self._channels_last_to_first(prediction)

        # 5. JiT objective: network predicts x, loss is applied in v-space.
        denominator = torch.clamp(1.0 - t_b, min=0.05)
        predicted_v = (prediction_bchw - z_bchw) / denominator
        loss = F.mse_loss(predicted_v, target_v)

        if return_clean_images:
            aux = {
                "clean_images_bchw": images_bchw.detach(),
                "labels": labels_tensor.detach() if labels_tensor is not None else None,
            }
            return loss, aux

        return loss

    def training_step(self, batch, batch_idx):
        """Run one training step and log the loss."""
        loss = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log(
            "global_step", float(self.trainer.global_step), on_step=True, on_epoch=False, prog_bar=True, logger=False
        )
        self.log("current_step", float(self.trainer.global_step), on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics."""
        super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        """Compute validation loss."""
        loss = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        return loss

    def test_step(self, batch, batch_idx):
        """Lightning test loop placeholder (no dedicated metric)."""
        # Generation experiments do not have a dedicated test metric.
        pass

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_inference_steps: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples with the JiT continuous-time sampler."""
        return self._sample_continuous(num_samples, num_inference_steps, labels)

    def on_fit_start(self) -> None:
        """Move EMA weights to device before training begins."""
        super().on_fit_start()
        if self.ema_enabled and self._ema_model is not None:
            self._ema_model.to(self.device)
            self._ema_model.eval()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Update EMA parameters at the configured interval."""
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
        """Compute and log validation summary metrics and sample grids."""
        if self.trainer.sanity_checking:
            return

        if self.fid_online_jit and self.current_epoch % self.fid_interval == 0:
            if self.global_rank == 0:
                print(f"Starting FID evaluation for epoch {self.current_epoch}...")
            self._run_jit_online_eval()

        if not self.log_samples:
            return
        if self.example_input_shape is None:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        num_samples = int(self.num_generated_samples)

        # When class conditioning is enabled we draw random labels so validation grids vary each epoch.
        # Guidance scale determines whether the conditional branch is actually used during sampling.
        labels_for_sampling = None
        if self.class_conditioning:
            assert self.num_classes is not None
            labels_for_sampling = torch.randint(
                low=0,
                high=self.num_classes,
                size=(num_samples,),
                device=self.device,
                dtype=torch.long,
            )

        samples = self.sample(num_samples=num_samples, labels=labels_for_sampling)
        value_range = (-1.0, 1.0)
        normalize_grid = True

        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None:
            unnormalize_fn = getattr(datamodule, "unnormalize", None)
            if callable(unnormalize_fn):
                try:
                    samples = unnormalize_fn(samples)
                except (TypeError, ValueError):
                    pass
                else:
                    samples = torch.clamp(samples, 0.0, 1.0)
                    normalize_grid = False
                    value_range = (0.0, 1.0)

        samples_bchw = self._channels_last_to_first(samples)
        grid = make_grid(
            samples_bchw.detach().cpu(),
            nrow=max(1, int(math.sqrt(num_samples))),
            normalize=normalize_grid,
            value_range=value_range,
        )
        self.logger.experiment.log(
            {
                "val/samples": wandb.Image(grid.cpu()),
                "global_step": self.global_step,
            }
        )

    # -------------------------------------------------------------------------
    # Continuous Time / Flow Matching Logic (JiT Port)
    # -------------------------------------------------------------------------

    def _sample_continuous(
        self,
        num_samples: int,
        num_inference_steps: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sampling loop for continuous time flow matching."""
        if self.example_input_shape is None:
            raise RuntimeError("Cannot sample before observing at least one training batch.")

        num_inference_steps = num_inference_steps or self.default_inference_steps
        device = self.device
        height, width, channels = self.example_input_shape

        # Prepare labels
        labels_tensor: Optional[torch.Tensor]
        if self.class_conditioning:
            if labels is None:
                labels_tensor = torch.randint(0, self.num_classes, (num_samples,), device=device, dtype=torch.long)
            else:
                labels_tensor = torch.as_tensor(labels, device=device, dtype=torch.long).view(-1)
                if labels_tensor.numel() == 1 and num_samples > 1:
                    labels_tensor = labels_tensor.expand(num_samples)
                if labels_tensor.shape[0] != num_samples:
                    raise ValueError("labels must either be a scalar or have the same length as num_samples.")
                if (labels_tensor < 0).any():
                    raise ValueError("labels must contain non-negative class indices.")
        else:
            if labels is not None:
                raise ValueError("labels were provided but the model was configured without class conditioning.")
            labels_tensor = None

        # 1. Initialize with noise scaled by the configured noise_scale (matches JiT denoiser.py).
        z = torch.randn((num_samples, channels, height, width), device=device) * self.noise_scale

        # 2. Integrate from t=0 (noise) to t=1 (data) using a Heun solver.
        dt = 1.0 / num_inference_steps

        # Use EMA model if enabled
        use_ema = self.ema_enabled and self._ema_model is not None and self._ema_has_been_updated
        inference_model = self._ema_model if use_ema else self.network
        was_training = inference_model.training
        inference_model.eval()
        inference_model = inference_model.to(device)

        for i in range(num_inference_steps - 1):
            t = i * dt
            z = self._heun_step(inference_model, z, t, dt, labels_tensor)

        # Last step: Euler only (matches JiT reference — avoids an extra model call at t≈1
        # where the ODE denominator is near its minimum clamp value).
        t_last = (num_inference_steps - 1) * dt
        v_last = self._model_forward_continuous(inference_model, z, t_last, labels_tensor)
        z = z + v_last * dt

        if was_training:
            inference_model.train()

        return torch.clamp(self._channels_first_to_last(z), -1.0, 1.0)

    def _model_forward_continuous(self, model, x, t_scalar, labels):
        # Broadcast t
        t_batch = torch.full((x.shape[0],), t_scalar, device=x.device, dtype=torch.float32)

        # 1. Condition
        do_cfg = self.cfg_enabled and (
            min(self.cfg_interval_start, self.cfg_interval_end)
            <= t_scalar
            <= max(self.cfg_interval_start, self.cfg_interval_end)
        )

        pred_bchw = None

        if do_cfg:
            # CFG
            cond_uncond, class_emb_uncond = self._condition_from_timesteps(
                t_batch,
                labels=labels,
                unconditional=True,
            )
            cond_cond, class_emb_cond = self._condition_from_timesteps(
                t_batch,
                labels=labels,
            )

            # Note: Model expects channels-last in 'input'
            x_cl = self._channels_first_to_last(x)

            inp_uncond = {"input": x_cl, "condition": cond_uncond}
            if class_emb_uncond is not None:
                inp_uncond["class_emb"] = class_emb_uncond
            inp_cond = {"input": x_cl, "condition": cond_cond}
            if class_emb_cond is not None:
                inp_cond["class_emb"] = class_emb_cond

            out_uncond = model(inp_uncond)["logits"]
            out_cond = model(inp_cond)["logits"]

            pred_uncond = self._channels_last_to_first(out_uncond)
            pred_cond = self._channels_last_to_first(out_cond)

            pred_bchw = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
        else:
            # Standard forward
            condition, class_emb = self._condition_from_timesteps(t_batch, labels=labels)
            x_cl = self._channels_first_to_last(x)
            inp = {"input": x_cl, "condition": condition}
            if class_emb is not None:
                inp["class_emb"] = class_emb
            out = model(inp)["logits"]
            pred_bchw = self._channels_last_to_first(out)

        # JiT mode predicts x and converts to velocity with denominator clipping.
        denominator = max(1.0 - t_scalar, 0.05)
        return (pred_bchw - x) / denominator

    def _heun_step(self, model, x, t, dt, labels):
        v = self._model_forward_continuous(model, x, t, labels)
        x_euler = x + v * dt
        v_next = self._model_forward_continuous(model, x_euler, t + dt, labels)
        x_next = x + 0.5 * dt * (v + v_next)
        return x_next

    def _run_jit_online_eval(self) -> None:
        """Run standard FID evaluation matching the JiT repository's methodology."""
        if self.fid_stats_file is None:
            if self.global_rank == 0:
                print("FID stats file not configured. Skipping online evaluation.")
            return

        if not os.path.exists(self.fid_stats_file) and not os.environ.get("SKIP_FID_STATS_CHECK"):
            if self.global_rank == 0:
                print(f"FID stats file {self.fid_stats_file} not found. Skipping online evaluation.")
            return

        fid_run_dir = os.path.join(self.trainer.default_root_dir, f"fid_eval_{self.global_step}")

        if self.global_rank == 0:
            os.makedirs(fid_run_dir, exist_ok=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        world_size = self.trainer.world_size
        global_rank = self.global_rank

        total_samples = self.fid_num_samples
        base_samples = total_samples // world_size
        remainder = total_samples % world_size

        # Non-overlapping slice [start_idx, start_idx + my_count) for this rank.
        start_idx = base_samples * global_rank + min(global_rank, remainder)
        my_count = base_samples + (1 if global_rank < remainder else 0)

        if self.num_classes is not None:
            samples_per_class = total_samples // self.num_classes
            remainder_classes = total_samples % self.num_classes

            class_labels = np.arange(self.num_classes).repeat(samples_per_class)

            if remainder_classes > 0:
                class_labels = np.concatenate([class_labels, np.zeros(remainder_classes, dtype=int)])

            end_idx = start_idx + my_count
            my_labels = class_labels[start_idx:end_idx]
            my_labels = torch.tensor(my_labels, device=self.device, dtype=torch.long)

        else:
            my_labels = None

        batches = (
            math.ceil(len(my_labels) / self.fid_batch_size)
            if my_labels is not None
            else math.ceil(my_count / self.fid_batch_size)
        )

        self.eval()

        samples_generated = 0
        batch_iterator = range(batches)
        if global_rank == 0:
            batch_iterator = tqdm(batch_iterator, total=batches, desc="FID sample generation", leave=True)

        for i in batch_iterator:
            current_batch_size = (
                min(self.fid_batch_size, len(my_labels) - samples_generated)
                if my_labels is not None
                else min(self.fid_batch_size, my_count - samples_generated)
            )

            batch_labels = (
                my_labels[samples_generated : samples_generated + current_batch_size]
                if my_labels is not None
                else None
            )

            with torch.no_grad():
                samples = self.sample(
                    num_samples=current_batch_size,
                    num_inference_steps=self.fid_num_inference_steps,
                    labels=batch_labels,
                )
                # sample() returns (B, H, W, C), but save_image expects (C, H, W).
                samples = samples.permute(0, 3, 1, 2)  # (B, C, H, W)

            # Denormalize
            samples = (samples + 1.0) / 2.0
            samples = torch.clamp(samples, 0.0, 1.0)

            for b in range(current_batch_size):
                filename = f"sample_{start_idx + samples_generated + b:08d}.png"
                save_path = os.path.join(fid_run_dir, filename)
                save_image(samples[b], save_path)

            samples_generated += current_batch_size

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if global_rank == 0:
            try:
                from torch_fidelity.metric_fid import fid_featuresdict_to_statistics, fid_statistics_to_metric
                from torch_fidelity.metric_isc import isc_featuresdict_to_metric
                from torch_fidelity.utils import (
                    create_feature_extractor,
                    extract_featuresdict_from_input_id,
                    resolve_feature_layer_for_metric,
                )

                feat_layer_fid = resolve_feature_layer_for_metric("fid", fid=True)
                feat_layer_isc = resolve_feature_layer_for_metric("isc", isc=True)
                feat_layers = list({feat_layer_fid, feat_layer_isc})

                feat_extractor = create_feature_extractor("inception-v3-compat", feat_layers, cuda=True, verbose=False)

                featuresdict = extract_featuresdict_from_input_id(
                    input_id=1, feat_extractor=feat_extractor, input1=fid_run_dir, cuda=True, verbose=False
                )

                stats_1 = fid_featuresdict_to_statistics(featuresdict, feat_layer_fid)
                f = np.load(self.fid_stats_file)
                stats_2 = {"mu": f["mu"], "sigma": f["sigma"]}
                f.close()

                stats_1["mu"] = stats_1["mu"].astype(np.float64)
                stats_1["sigma"] = stats_1["sigma"].astype(np.float64)
                stats_2["mu"] = stats_2["mu"].astype(np.float64)
                stats_2["sigma"] = stats_2["sigma"].astype(np.float64)

                metrics_fid = fid_statistics_to_metric(stats_1, stats_2, verbose=False)
                metrics_isc = isc_featuresdict_to_metric(featuresdict, feat_layer_isc, isc_splits=10)

                fid = metrics_fid["frechet_inception_distance"]
                isc = metrics_isc["inception_score_mean"]

                print(f"FID: {fid:.4f}, IS: {isc:.4f}")

                self.log("metrics/fid_online", fid, sync_dist=False)
                self.log("metrics/is_online", isc, sync_dist=False)

                if self.logger is not None and hasattr(self.logger, "experiment"):
                    self.logger.experiment.log(
                        {"metrics/fid_online": fid, "metrics/is_online": isc, "global_step": self.global_step}
                    )

            except Exception as e:
                print(f"Error calculating FID: {e}")
            finally:
                print(f"Removing temporary FID directory: {fid_run_dir}")
                shutil.rmtree(fid_run_dir, ignore_errors=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
