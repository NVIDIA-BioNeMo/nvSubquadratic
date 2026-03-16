# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Lightning wrappers for the Classification and Regression experiments."""

import copy
import math
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from diffusers import DDIMScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

from experiments.default_cfg import DiffusionExperimentConfig
from experiments.lightning_wrappers.base_lightning_wrapper import LightningWrapperBase


class _FallbackFIDMetric:
    """Minimal FID metric replacement that keeps tests runnable without torch-fidelity."""

    def __init__(self) -> None:
        self._device = torch.device("cpu")
        self.reset()

    def to(self, device: torch.device) -> "_FallbackFIDMetric":
        self._device = torch.device(device)
        return self

    def reset(self) -> None:
        self._reals: list[torch.Tensor] = []
        self._fakes: list[torch.Tensor] = []

    def update(self, images: torch.Tensor, *, real: bool) -> None:
        data = images.detach().to("cpu", dtype=torch.float32)
        if real:
            self._reals.append(data)
        else:
            self._fakes.append(data)

    def compute(self) -> torch.Tensor:
        if not self._reals or not self._fakes:
            return torch.tensor(float("nan"), device=self._device)
        real = torch.cat(self._reals, dim=0)
        fake = torch.cat(self._fakes, dim=0)
        value = torch.linalg.norm(real.mean(dim=0) - fake.mean(dim=0))
        return value.to(self._device)


class DiffusionWrapper(LightningWrapperBase):
    """Lightning module for DDPM/DDIM-style training with CKConv backbones.

    .. TODO(@dmknigge): Resume support (critical for long diffusion runs)
        - The manual EMA model (``self._ema_model``) is NOT saved or restored
          in checkpoints. On resume, the EMA shadow copy is re-initialized
          from ``deepcopy(self.network)`` (i.e., the *resumed* weights), losing
          all accumulated averaging. Add ``on_save_checkpoint`` /
          ``on_load_checkpoint`` hooks to serialize ``_ema_model.state_dict()``
          and ``_ema_has_been_updated`` into the checkpoint.
          Alternatively, migrate to the ``LabeledEMAWeightAveraging`` callback
          used by the classification pipeline, which handles this correctly.
        - Persist ``example_input_shape`` so that sampling works immediately
          after resume without waiting for the first training batch.
        - Add corresponding tests in ``tests/test_checkpoint_resume.py``.
    """

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

        # Store diffusion hyper-parameters from the configuration so that we can reuse them
        # across training and sampling without constantly reaching outside the module.
        self.num_train_timesteps = int(diffusion_cfg.num_train_timesteps)
        self.beta_schedule = diffusion_cfg.beta_schedule
        self.beta_start = float(diffusion_cfg.beta_start)
        self.beta_end = float(diffusion_cfg.beta_end)

        # Set the prediction type and validate it.
        assert diffusion_cfg.prediction_type in ["epsilon", "sample", "v_prediction"]
        self.prediction_type = diffusion_cfg.prediction_type

        trained_betas = None
        beta_schedule = self.beta_schedule
        if self.beta_schedule == "cosine_interpolated":
            trained_betas = self._build_cosine_interpolated_betas(
                num_steps=self.num_train_timesteps,
                logsnr_min=diffusion_cfg.cosine_schedule_logsnr_min,
                logsnr_max=diffusion_cfg.cosine_schedule_logsnr_max,
                image_resolution=diffusion_cfg.cosine_schedule_image_resolution,
                noise_res_low=diffusion_cfg.cosine_schedule_noise_res_low,
                noise_res_high=diffusion_cfg.cosine_schedule_noise_res_high,
            )
            beta_schedule = "linear"  # unused when trained_betas is provided

        # Instantiate the diffusers scheduler, delegating all diffusion math (alphas, betas, posteriors, etc.)
        # to a single well-tested implementation rather than maintaining our own copy here.
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=beta_schedule,
            prediction_type=self.prediction_type,
            clip_sample=False,
            set_alpha_to_one=False,
            trained_betas=trained_betas,
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
        self.use_sigmoid_loss_weighting = bool(diffusion_cfg.use_sigmoid_loss_weighting)
        self.sigmoid_loss_bias = float(diffusion_cfg.sigmoid_loss_bias)

        # Optional online FID evaluation.
        self.fid_max_batches = max(int(getattr(diffusion_cfg, "fid_num_batches", 0) or 0), 0)
        self.fid_enabled = bool(getattr(diffusion_cfg, "fid_enabled", False)) and self.fid_max_batches > 0
        fid_steps_cfg = getattr(diffusion_cfg, "fid_num_inference_steps", None)
        self.fid_num_inference_steps = (
            int(fid_steps_cfg) if fid_steps_cfg is not None else self.default_inference_steps
        )
        self.fid_metric: Optional[FrechetInceptionDistance | _FallbackFIDMetric] = self._build_fid_metric()
        self._fid_batches_seen = 0
        if self.fid_metric is None:
            self.fid_enabled = False

        # Classifier-free guidance (CFG) specific settings ---------------------------------------------------------
        # We make the behaviour completely configurable so the same wrapper can serve
        # both unconditional and class-conditional runs without branching out to a
        # dedicated LightningModule. The goal is to keep the training loop readable
        # while still exposing the knobs that practitioners expect.
        self.class_conditioning = diffusion_cfg.num_classes is not None
        # Guidance is only meaningful when we have a class embedding to steer the
        # model. If the user forgets to specify a class count we fall back to the
        # unconditional path and later raise a helpful error when guidance is used.
        self.cfg_enabled = bool(diffusion_cfg.use_classifier_free_guidance) and self.class_conditioning
        self.guidance_scale = float(diffusion_cfg.guidance_scale)
        # During training we optionally drop the conditioning signal at random so the
        # network learns an unconditional branch that we can later reuse at inference.
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
            # We dedicate one additional embedding slot to represent the unconditional
            # branch. Using a learnable parameter keeps the code flexible (e.g. if we
            # later decide to fine-tune the unconditional vector instead of keeping it
            # at zero).
            self.null_label_index = self.num_classes
            self.label_embed = torch.nn.Embedding(self.num_classes + 1, hidden_dim)
            torch.nn.init.normal_(self.label_embed.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                # Starting from an explicit zero vector gives the unconditional branch a
                # deterministic meaning: it simply relies on the time embedding.
                self.label_embed.weight[self.null_label_index].zero_()
        else:
            self.null_label_index = None
            self.label_embed = None

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

        # Allow Hugging Face-backed models to register themselves for callbacks.
        register_fn = getattr(network, "hf_register_diffusion_wrapper", None)
        if callable(register_fn):
            register_fn(self)

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

    def _sigmoid_weighted_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SiD2-style sigmoid loss weighting based on the per-sample log SNR."""
        if not hasattr(self.scheduler, "alphas_cumprod"):
            raise AttributeError("Current scheduler does not expose 'alphas_cumprod' required for log-SNR weighting.")

        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=prediction.device, dtype=prediction.dtype)
        alphas = alphas_cumprod[timesteps]
        eps = torch.finfo(alphas.dtype).eps
        alphas = alphas.clamp(min=eps, max=1.0 - eps)
        log_snr = torch.log(alphas / (1.0 - alphas))

        weights = torch.sigmoid(log_snr - self.sigmoid_loss_bias).to(dtype=prediction.dtype)
        squared_error = (prediction - target) ** 2
        view_shape = (weights.shape[0],) + (1,) * (squared_error.ndim - 1)
        weighted_error = weights.view(view_shape) * squared_error
        return weighted_error.mean()

    @staticmethod
    def _cosine_interpolated_logsnr(
        t: torch.Tensor,
        *,
        logsnr_min: float,
        logsnr_max: float,
        image_resolution: int,
        noise_res_low: int,
        noise_res_high: int,
    ) -> torch.Tensor:
        """Return the cosine-interpolated log-SNR schedule from SiD2 Appendix B."""
        if noise_res_high <= 0 or noise_res_low <= 0:
            raise ValueError("Noise resolutions for cosine schedule must be positive.")

        log_change_high = math.log(float(image_resolution)) - math.log(float(noise_res_high))
        log_change_low = math.log(float(image_resolution)) - math.log(float(noise_res_low))
        b = math.atan(math.exp(-0.5 * logsnr_max))
        a = math.atan(math.exp(-0.5 * logsnr_min)) - b
        logsnr_cosine = -2.0 * torch.log(torch.tan(a * t + b))
        logsnr_high = logsnr_cosine + log_change_high
        logsnr_low = logsnr_cosine + log_change_low
        return (1.0 - t) * logsnr_high + t * logsnr_low

    def _build_cosine_interpolated_betas(
        self,
        *,
        num_steps: int,
        logsnr_min: float,
        logsnr_max: float,
        image_resolution: int,
        noise_res_low: int,
        noise_res_high: int,
    ) -> np.ndarray:
        """Generate a beta schedule that matches the cosine-interpolated log-SNR used in SiD2."""
        if num_steps <= 0:
            raise ValueError("Number of diffusion steps must be positive.")

        device = torch.device("cpu")
        t = torch.linspace(0.0, 1.0, steps=num_steps, dtype=torch.float64, device=device)
        logsnr = self._cosine_interpolated_logsnr(
            t,
            logsnr_min=logsnr_min,
            logsnr_max=logsnr_max,
            image_resolution=image_resolution,
            noise_res_low=noise_res_low,
            noise_res_high=noise_res_high,
        )
        alphas_cumprod = torch.sigmoid(logsnr)
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-7, max=1.0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas_cumprod.dtype, device=device), alphas_cumprod[:-1]]
        )
        alphas = alphas_cumprod / alphas_cumprod_prev
        betas = 1.0 - alphas
        betas = torch.clamp(betas, min=1e-8, max=0.999)
        return betas.cpu().numpy().astype(np.float32)

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
        """Embed diffusion steps via their log-SNR rather than raw indices."""
        alphas_cumprod = getattr(self.scheduler, "alphas_cumprod", None)
        if alphas_cumprod is None:
            raise RuntimeError("Scheduler does not have precomputed alphas_cumprod for noise-level embedding.")
        alphas_cumprod = alphas_cumprod.to(timesteps.device)
        indices = timesteps.to(torch.long)
        indices = torch.clamp(indices, min=0, max=alphas_cumprod.shape[0] - 1)
        alpha = alphas_cumprod[indices].clamp_(1e-7, 1.0 - 1e-7)
        logsnr = torch.log(alpha) - torch.log1p(-alpha)
        target_dtype = self.time_mlp[0].weight.dtype if hasattr(self.time_mlp[0], "weight") else logsnr.dtype
        logsnr = logsnr.to(dtype=target_dtype)
        return self.time_mlp(self._timestep_embedding(logsnr))

    def _condition_from_timesteps(
        self,
        timesteps: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        unconditional: bool = False,
        dropout_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the conditioning vector for a batch of timesteps (and optional class labels).

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
            return time_emb

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
        # We simply add the two embeddings together so the backbone receives a single
        # conditioning vector. This mirrors the standard DDPM/DiT approach and keeps the
        # interface consistent with the time-only case.
        return time_emb + label_emb

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

            # Bernoulli mask indicating which samples should use the unconditional branch so the model
            # learns to ignore class information part of the time. This is the core of classifier-free guidance.
            dropout_mask = None
            if self.condition_dropout_prob > 0.0:
                dropout_mask = torch.rand(batch_size, device=self.device) < self.condition_dropout_prob
            condition = self._condition_from_timesteps(
                timesteps,
                labels=labels_tensor,
                dropout_mask=dropout_mask,
            )
        else:
            # Purely time-conditioned diffusion behaves exactly as before.
            condition = self._condition_from_timesteps(timesteps)

        # The denoiser returns all of its outputs in a dict; training only needs the raw logits tensor.
        prediction = self.network({"input": noisy_images, "condition": condition})["logits"]

        # Convert prediction to channels-first for loss computation.
        prediction_bchw = self._channels_last_to_first(prediction)

        # Compute the appropriate training target and return the MSE loss.
        target = self._compute_training_target(images_bchw, noise_bchw, timesteps)

        # Optionally use the sigmoid-weighted loss from SiD2 instead of plain MSE.
        if self.use_sigmoid_loss_weighting:
            loss = self._sigmoid_weighted_mse(prediction_bchw, target, timesteps)
        else:
            loss = self.loss_fn(prediction_bchw, target)

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
        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset FID metrics at the start of validation."""
        super().on_validation_epoch_start()
        if self.fid_metric is not None:
            self.fid_metric.reset()
            self._fid_batches_seen = 0

    def validation_step(self, batch, batch_idx):
        """Compute validation loss and optionally accumulate FID statistics."""
        collect_images = self._should_collect_fid()
        shared = self._shared_step(batch, return_clean_images=collect_images)
        if collect_images:
            assert isinstance(shared, tuple)
            loss, aux = shared
            self._update_fid_metrics(aux["clean_images_bchw"], aux["labels"])
        else:
            assert isinstance(shared, torch.Tensor)
            loss = shared
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        return loss

    def _should_collect_fid(self) -> bool:
        return self.fid_metric is not None and self._fid_batches_seen < self.fid_max_batches

    def _build_fid_metric(self):
        if not self.fid_enabled:
            return None
        try:
            return FrechetInceptionDistance(feature=2048, normalize=True)
        except ModuleNotFoundError:
            warnings.warn(
                "Torch-fidelity is not installed; using a lightweight FID metric for tests. "
                "Install `torchmetrics[image]` for the full metric.",
                stacklevel=2,
            )
            return _FallbackFIDMetric()

    def _prepare_images_for_fid(self, images_bchw: torch.Tensor) -> torch.Tensor:
        images = images_bchw.detach()
        if images.dtype != torch.float32:
            images = images.float()
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        return images

    def _update_fid_metrics(self, clean_images_bchw: torch.Tensor, labels: Optional[torch.Tensor]) -> None:
        if not self._should_collect_fid() or self.example_input_shape is None:
            return
        assert self.fid_metric is not None
        fid_metric = self.fid_metric.to(self.device)

        real = self._prepare_images_for_fid(clean_images_bchw)
        fid_metric.update(real, real=True)

        with torch.no_grad():
            sample_kwargs = {}
            if self.class_conditioning and labels is not None:
                sample_kwargs["labels"] = labels.to(self.device)
            generated = self.sample(
                num_samples=real.shape[0],
                num_inference_steps=self.fid_num_inference_steps,
                **sample_kwargs,
            )
            generated_bchw = self._channels_last_to_first(generated)
            fake = self._prepare_images_for_fid(generated_bchw)
            fid_metric.update(fake, real=False)

        self._fid_batches_seen += 1

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
        """Generate samples using the current diffusion model (EMA if available)."""
        if self.example_input_shape is None:
            raise RuntimeError("Cannot sample before observing at least one training batch.")

        num_inference_steps = num_inference_steps or self.default_inference_steps
        device = self.device
        height, width, channels = self.example_input_shape

        # When class conditioning is active we need a label for each generated sample. If the caller did not
        # specify one we default to a simple deterministic pattern (cycling through classes) so logged grids
        # stay easy to interpret.
        labels_tensor: Optional[torch.Tensor]
        if self.class_conditioning:
            if labels is None:
                # Random fallback when no labels were provided.
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

        # Start from pure Gaussian noise in channels-first format because that's what the scheduler expects.
        sample_bchw = torch.randn((num_samples, channels, height, width), device=device)

        # Prepare the scheduler timesteps on the current device - this mirrors the standard diffusers pipeline.
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        use_ema = self.ema_enabled and self._ema_model is not None and self._ema_has_been_updated
        inference_model = self._ema_model if use_ema else self.network
        was_training = inference_model.training
        inference_model.eval()
        inference_model = inference_model.to(device)

        for timestep in self.scheduler.timesteps:
            # Broadcast the scalar timestep to a batch so we can embed it and feed the denoiser.
            t_batch = torch.full((num_samples,), timestep.item(), device=device, dtype=torch.long)

            # Convert the working sample back to channels-last before asking the network for a prediction.
            model_input = self._channels_first_to_last(sample_bchw)

            if self.cfg_enabled:
                # Run the denoiser twice: once on the unconditional branch and once with the actual labels.
                cond_uncond = self._condition_from_timesteps(
                    t_batch,
                    labels=labels_tensor,
                    unconditional=True,
                )
                cond_cond = self._condition_from_timesteps(
                    t_batch,
                    labels=labels_tensor,
                )
                outputs_uncond = inference_model({"input": model_input, "condition": cond_uncond})["logits"]
                outputs_cond = inference_model({"input": model_input, "condition": cond_cond})["logits"]
                pred_uncond = self._channels_last_to_first(outputs_uncond)
                pred_cond = self._channels_last_to_first(outputs_cond)
                # Linear interpolation between unconditional and conditional predictions as described in
                # Ho & Salimans (2022). guidance_scale=1 leaves the result unchanged, larger values push
                # generations closer to the conditional manifold.
                model_output_bchw = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
            else:
                condition = self._condition_from_timesteps(t_batch, labels=labels_tensor)
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

        if self.fid_metric is not None and self._fid_batches_seen > 0:
            fid_value = self.fid_metric.compute()
            self.log("metrics/fid", fid_value, prog_bar=False, sync_dist=self.distributed)
            if self.logger is not None and hasattr(self.logger, "experiment"):
                try:
                    self.logger.experiment.log(
                        {
                            "metrics/fid": float(fid_value.item()),
                            "global_step": self.global_step,
                        }
                    )
                except Exception:
                    pass

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
