"""Unit tests for the DiffusionWrapper with and without classifier-free guidance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from experiments.default_cfg import DiffusionConfig, DiffusionExperimentConfig
from experiments.lightning_wrappers.diffusion_wrapper import DiffusionWrapper


class RecordingDenoiser(torch.nn.Module):
    """Minimal denoiser that records every conditioning vector it receives."""

    def __init__(self, hidden_dim: int, channels: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.calls: list[torch.Tensor] = []
        # Register a parameter so Lightning can infer the device even though the model is almost empty.
        self.offset = torch.nn.Parameter(torch.zeros(1))

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        condition = data["condition"]
        # Store a detached copy so tests can inspect the unconditional vs. conditional branches.
        self.calls.append(condition.detach().cpu())
        logits = torch.tanh(data["input"] + self.offset)
        return {"logits": logits}


def _make_wrapper(
    *,
    num_classes: int | None,
    use_cfg: bool,
    guidance_scale: float = 1.0,
    condition_dropout_prob: float = 0.0,
    num_inference_steps: int = 2,
    diffusion_overrides: dict[str, Any] | None = None,
) -> DiffusionWrapper:
    """Helper to instantiate a DiffusionWrapper with a lightweight recording network."""

    diff_cfg = DiffusionConfig()
    diff_cfg.num_train_timesteps = 8
    diff_cfg.num_inference_steps = num_inference_steps
    diff_cfg.num_samples = 2
    diff_cfg.log_samples = False
    diff_cfg.ddim_eta = 0.0
    diff_cfg.ema_enabled = False
    diff_cfg.use_classifier_free_guidance = use_cfg
    diff_cfg.guidance_scale = guidance_scale
    diff_cfg.condition_dropout_prob = condition_dropout_prob
    diff_cfg.num_classes = num_classes
    if diffusion_overrides:
        for key, value in diffusion_overrides.items():
            setattr(diff_cfg, key, value)

    exp_cfg = DiffusionExperimentConfig(diffusion=diff_cfg)
    exp_cfg.train.iterations = 10
    exp_cfg.train.batch_size = 1
    exp_cfg.train.grad_clip = 0.0

    network = RecordingDenoiser(hidden_dim=8, channels=3)
    wrapper = DiffusionWrapper(network=network, cfg=exp_cfg)
    wrapper.to(torch.device("cpu"))
    return wrapper


def test_class_conditioning_requires_labels() -> None:
    """Ensure the wrapper refuses to run class-conditioned diffusion without labels."""
    wrapper = _make_wrapper(num_classes=10, use_cfg=True, guidance_scale=3.0)
    images = torch.randn(2, 4, 4, 3)
    batch = {"input": images}
    with pytest.raises(RuntimeError, match="Class conditioning requires datamodule"):
        wrapper._shared_step(batch)  # type: ignore[arg-type]


def test_cfg_sampling_invokes_both_branches() -> None:
    """Classifier-free guidance should evaluate unconditional and conditional paths per step."""
    torch.manual_seed(0)
    wrapper = _make_wrapper(num_classes=4, use_cfg=True, guidance_scale=2.5, num_inference_steps=2)
    wrapper.example_input_shape = torch.Size((4, 4, 3))

    samples = wrapper.sample(num_samples=1, labels=torch.tensor([1]))
    assert samples.shape == (1, 4, 4, 3)
    # Two timesteps * two passes (unconditional + conditional).
    assert len(wrapper.network.calls) == 4  # type: ignore[attr-defined]
    unconditional_condition = wrapper.network.calls[0]  # type: ignore[attr-defined]
    conditional_condition = wrapper.network.calls[1]  # type: ignore[attr-defined]
    assert not torch.allclose(unconditional_condition, conditional_condition)


def test_sampling_without_guidance_uses_single_branch() -> None:
    """When guidance is disabled the denoiser should run exactly once per timestep."""
    wrapper = _make_wrapper(num_classes=4, use_cfg=False, num_inference_steps=3)
    wrapper.example_input_shape = torch.Size((4, 4, 3))
    wrapper.network.calls.clear()  # type: ignore[attr-defined]

    samples = wrapper.sample(num_samples=1, labels=torch.tensor([2]))
    assert samples.shape == (1, 4, 4, 3)
    # Three timesteps, single pass each.
    assert len(wrapper.network.calls) == 3  # type: ignore[attr-defined]


def test_sigmoid_loss_weighting_matches_manual_computation() -> None:
    """Verify that the sigmoid-weighted loss matches a manual computation."""
    bias = -0.75
    wrapper = _make_wrapper(
        num_classes=None,
        use_cfg=False,
        diffusion_overrides={
            "use_sigmoid_loss_weighting": True,
            "sigmoid_loss_bias": bias,
            "num_train_timesteps": 8,
        },
    )

    prediction = torch.tensor([0.2, -0.1], dtype=torch.float32).view(2, 1, 1, 1)
    target = torch.tensor([0.0, 0.3], dtype=torch.float32).view(2, 1, 1, 1)
    timesteps = torch.tensor([0, 5], dtype=torch.long)

    loss = wrapper._sigmoid_weighted_mse(prediction, target, timesteps)
    alphas = wrapper.scheduler.alphas_cumprod[timesteps].to(dtype=prediction.dtype)
    log_snr = torch.log(alphas / (1.0 - alphas))
    weights = torch.sigmoid(log_snr - bias).view(-1, 1, 1, 1)
    expected_loss = ((prediction - target) ** 2 * weights).mean()

    assert torch.allclose(loss, expected_loss, atol=1e-6)


def test_cosine_interpolated_schedule_matches_reference_formula() -> None:
    """Ensure the cosine-interpolated schedule matches the closed-form expression."""
    overrides = {
        "beta_schedule": "cosine_interpolated",
        "num_train_timesteps": 16,
        "cosine_schedule_logsnr_min": -12.0,
        "cosine_schedule_logsnr_max": 12.0,
        "cosine_schedule_image_resolution": 128,
        "cosine_schedule_noise_res_low": 32,
        "cosine_schedule_noise_res_high": 128,
    }
    wrapper = _make_wrapper(num_classes=None, use_cfg=False, diffusion_overrides=overrides)

    scheduler_betas = wrapper.scheduler.betas.cpu().numpy()
    expected_betas = wrapper._build_cosine_interpolated_betas(
        num_steps=overrides["num_train_timesteps"],
        logsnr_min=overrides["cosine_schedule_logsnr_min"],
        logsnr_max=overrides["cosine_schedule_logsnr_max"],
        image_resolution=overrides["cosine_schedule_image_resolution"],
        noise_res_low=overrides["cosine_schedule_noise_res_low"],
        noise_res_high=overrides["cosine_schedule_noise_res_high"],
    )

    np.testing.assert_allclose(scheduler_betas, expected_betas, rtol=1e-6, atol=1e-8)
