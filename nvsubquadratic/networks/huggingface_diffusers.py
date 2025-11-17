# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Wrapper modules that adapt Hugging Face diffusers models to the nvSubQuadratic diffusion pipeline."""

from __future__ import annotations

import types
import weakref
from dataclasses import asdict, dataclass, is_dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import DiTTransformer2DModel
from diffusers.models.unets.uvit_2d import UVit2DModel


class _SharedTimestepState:
    __slots__ = ("latest",)

    def __init__(self) -> None:
        self.latest: torch.Tensor | None = None

    def __deepcopy__(self, memo) -> "_SharedTimestepState":
        memo[id(self)] = self
        return self


@dataclass
class HuggingFaceDiTConfig:
    """Configuration for DiT backbones."""

    sample_size: int = 28
    patch_size: int | None = 2
    in_channels: int = 1
    out_channels: int | None = None

    num_layers: int = 12
    num_attention_heads: int = 16
    attention_head_dim: int = 64
    dropout: float = 0.0
    norm_num_groups: int = 32
    attention_bias: bool = False
    activation_fn: str = "gelu-approximate"
    num_embeds_ada_norm: int | None = 1_000
    upcast_attention: bool = False
    norm_type: str = "ada_norm_zero"
    norm_elementwise_affine: bool = True
    norm_eps: float = 1e-6

    dtype: str = "float32"


@dataclass
class HuggingFaceUVitConfig:
    """Configuration for UVit backbones."""

    sample_size: int = 32
    in_channels: int = 3
    out_channels: int | None = None

    hidden_size: int = 256
    cond_embed_dim: int = 128
    encoder_hidden_size: int = 128
    block_out_channels: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 512
    layer_norm_eps: float = 1e-5
    micro_cond_encode_dim: int | None = None
    micro_cond_embed_dim: int | None = None
    codebook_size: int | None = None
    vocab_size: int | None = None

    dtype: str = "float32"


class DiffusersDiTWrapper(nn.Module):
    """Adapter for DiT-style denoisers that expect discrete timesteps."""

    def __init__(
        self,
        hf_config: HuggingFaceDiTConfig,
        in_channels: int | None = None,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.hf_config = hf_config
        if in_channels is not None:
            self.hf_config.in_channels = in_channels
        if out_channels is not None:
            self.hf_config.out_channels = out_channels

        self.transformer = self._build_transformer(self.hf_config)
        self.hidden_dim = self.transformer.config.attention_head_dim * self.transformer.config.num_attention_heads

        self._timestep_state = _SharedTimestepState()
        self._registered_wrapper_ref: weakref.ReferenceType | None = None

    def _get_latest_timesteps(self) -> torch.Tensor | None:
        return self._timestep_state.latest

    def _set_latest_timesteps(self, value: torch.Tensor | None) -> None:
        self._timestep_state.latest = value

    _latest_timesteps = property(_get_latest_timesteps, _set_latest_timesteps)

    def _build_transformer(self, cfg: HuggingFaceDiTConfig) -> nn.Module:
        if DiTTransformer2DModel is None:
            raise ImportError("diffusers>=0.27 with DiTTransformer2DModel support is required for DiffusersDiTWrapper")

        def _filtered_kwargs(pairs: Iterable[tuple[str, object | None]]) -> dict[str, object]:
            return {k: v for k, v in pairs if v is not None}

        shared_args = (
            ("sample_size", cfg.sample_size),
            ("patch_size", cfg.patch_size),
            ("in_channels", cfg.in_channels),
            ("out_channels", cfg.out_channels),
            ("num_layers", cfg.num_layers),
            ("num_attention_heads", cfg.num_attention_heads),
            ("attention_head_dim", cfg.attention_head_dim),
            ("dropout", cfg.dropout),
            ("norm_num_groups", cfg.norm_num_groups),
            ("attention_bias", cfg.attention_bias),
            ("activation_fn", cfg.activation_fn),
            ("num_embeds_ada_norm", cfg.num_embeds_ada_norm),
            ("upcast_attention", cfg.upcast_attention),
            ("norm_type", cfg.norm_type),
            ("norm_elementwise_affine", cfg.norm_elementwise_affine),
            ("norm_eps", cfg.norm_eps),
        )

        kwargs = _filtered_kwargs(shared_args)
        return DiTTransformer2DModel(**kwargs)

    def hf_register_diffusion_wrapper(self, wrapper: "DiffusionWrapper") -> None:
        if self._registered_wrapper_ref is not None and self._registered_wrapper_ref() is wrapper:
            return

        self._registered_wrapper_ref = weakref.ref(wrapper)

        if not hasattr(wrapper, "_hf_timestep_callbacks"):
            original_condition = wrapper._condition_from_timesteps

            def patched_condition(
                self_wrapper,
                timesteps: torch.LongTensor,
                *args,
                **kwargs,
            ) -> torch.Tensor:
                conditioned = original_condition(timesteps, *args, **kwargs)
                for callback in getattr(self_wrapper, "_hf_timestep_callbacks", []):
                    callback(timesteps)
                return conditioned

            wrapper._hf_timestep_callbacks: list[Callable[[torch.LongTensor], None]] = []
            wrapper._condition_from_timesteps = types.MethodType(patched_condition, wrapper)

        def _update_timesteps(timesteps: torch.LongTensor) -> None:
            self._latest_timesteps = timesteps

        wrapper._hf_timestep_callbacks.append(_update_timesteps)

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._latest_timesteps is None:
            raise RuntimeError(
                "DiffusersDiTWrapper.forward() was called before timesteps were populated. "
                "Ensure that hf_register_diffusion_wrapper has been called."
            )

        noisy_sample = input_and_condition["input"]
        sample_bchw = torch.moveaxis(noisy_sample, -1, 1).contiguous()

        try:
            model_dtype = next(self.transformer.parameters()).dtype
        except StopIteration:  # pragma: no cover
            model_dtype = sample_bchw.dtype
        sample_bchw = sample_bchw.to(dtype=getattr(torch, self.hf_config.dtype, model_dtype))

        timesteps = self._latest_timesteps.to(device=sample_bchw.device, dtype=torch.long)

        if isinstance(self.transformer, DiTTransformer2DModel):
            class_labels = torch.zeros(sample_bchw.shape[0], device=sample_bchw.device, dtype=torch.long)
            output = self.transformer(sample_bchw, timestep=timesteps, class_labels=class_labels, return_dict=True)
        else:
            output = self.transformer(sample_bchw, timestep=timesteps, return_dict=True)

        prediction = torch.moveaxis(output.sample, 1, -1).contiguous().to(dtype=noisy_sample.dtype)
        return {"logits": prediction}

    def extra_repr(self) -> str:  # pragma: no cover - debugging helper
        if hasattr(self.hf_config, "items"):
            items = self.hf_config.items()
        elif is_dataclass(self.hf_config):
            items = asdict(self.hf_config).items()
        else:
            items = vars(self.hf_config).items()
        trimmed = {k: v for k, v in items if v is not None}
        return f"transformer={self.transformer.__class__.__name__}, config={trimmed}"


class DiffusersUVitWrapper(nn.Module):
    """Adapter for UVit models that require explicit conditioning tensors."""

    def __init__(
        self,
        hf_config: HuggingFaceUVitConfig,
        in_channels: int | None = None,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        if UVit2DModel is None:
            raise ImportError("diffusers>=0.35 with UVit2DModel support is required for DiffusersUVitWrapper")

        self.hf_config = hf_config
        if in_channels is not None:
            self.hf_config.in_channels = in_channels
        # UVit outputs logits over the latent codebook; out_channels unused but accepted for interface parity
        self.hf_config.out_channels = out_channels

        self.transformer = self._build_uvit(self.hf_config)
        self.hidden_dim = getattr(self.transformer.config, "block_out_channels", self.hf_config.in_channels)

        self._registered_wrapper_ref: weakref.ReferenceType | None = None
        self._timestep_state = _SharedTimestepState()

    def _get_latest_timesteps(self) -> torch.Tensor | None:
        return self._timestep_state.latest

    def _set_latest_timesteps(self, value: torch.Tensor | None) -> None:
        self._timestep_state.latest = value

    _latest_timesteps = property(_get_latest_timesteps, _set_latest_timesteps)

    def _build_uvit(self, cfg: HuggingFaceUVitConfig) -> UVit2DModel:
        kwargs = {
            "sample_size": cfg.sample_size,
            "in_channels": cfg.in_channels,
            "hidden_size": cfg.hidden_size,
            "cond_embed_dim": cfg.cond_embed_dim,
            "encoder_hidden_size": cfg.encoder_hidden_size,
            "block_out_channels": cfg.block_out_channels,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "intermediate_size": cfg.intermediate_size,
            "layer_norm_eps": cfg.layer_norm_eps,
        }
        if cfg.micro_cond_encode_dim is not None:
            kwargs["micro_cond_encode_dim"] = cfg.micro_cond_encode_dim
        if cfg.micro_cond_embed_dim is not None:
            kwargs["micro_cond_embed_dim"] = cfg.micro_cond_embed_dim
        if cfg.codebook_size is not None:
            kwargs["codebook_size"] = cfg.codebook_size
        if cfg.vocab_size is not None:
            kwargs["vocab_size"] = cfg.vocab_size
        return UVit2DModel(**kwargs)

    def hf_register_diffusion_wrapper(self, wrapper) -> None:
        if self._registered_wrapper_ref is not None and self._registered_wrapper_ref() is wrapper:
            return

        self._registered_wrapper_ref = weakref.ref(wrapper)

        if not hasattr(wrapper, "_hf_timestep_callbacks"):
            original_condition = wrapper._condition_from_timesteps

            def patched_condition(
                self_wrapper,
                timesteps: torch.LongTensor,
                *args,
                **kwargs,
            ) -> torch.Tensor:
                conditioned = original_condition(timesteps, *args, **kwargs)
                for callback in getattr(self_wrapper, "_hf_timestep_callbacks", []):
                    callback(timesteps)
                return conditioned

            wrapper._hf_timestep_callbacks: list[Callable[[torch.LongTensor], None]] = []
            wrapper._condition_from_timesteps = types.MethodType(patched_condition, wrapper)

        def _update_timesteps(timesteps: torch.LongTensor) -> None:
            self._latest_timesteps = timesteps

        wrapper._hf_timestep_callbacks.append(_update_timesteps)

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the UVit wrapper."""
        batch = input_and_condition["input"]
        sample_bchw = torch.moveaxis(batch, -1, 1).contiguous()

        conditioning = input_and_condition.get("condition")
        config = self.transformer.config
        batch_size = sample_bchw.shape[0]
        device = sample_bchw.device
        dtype = sample_bchw.dtype

        if isinstance(conditioning, dict):
            encoder_hidden_states = conditioning.get("encoder_hidden_states")
            pooled_text_emb = conditioning.get("pooled_text_emb")
            micro_conds = conditioning.get("micro_conds")
            input_ids = conditioning.get("input_ids", sample_bchw)
        else:
            encoder_hidden_states = None
            pooled_text_emb = conditioning
            micro_conds = None
            input_ids = sample_bchw

        if encoder_hidden_states is None:
            enc_dim = getattr(config, "encoder_hidden_size", self.hidden_dim)
            encoder_hidden_states = torch.zeros(batch_size, 1, enc_dim, device=device, dtype=dtype)

        cond_dim = getattr(
            config,
            "cond_embed_dim",
            pooled_text_emb.shape[-1] if isinstance(pooled_text_emb, torch.Tensor) else self.hidden_dim,
        )
        if pooled_text_emb is None:
            pooled_text_emb = torch.zeros(batch_size, cond_dim, device=device, dtype=dtype)
        else:
            pooled_text_emb = pooled_text_emb.to(device=device, dtype=dtype)
            if pooled_text_emb.shape[-1] != cond_dim:
                if pooled_text_emb.shape[-1] < cond_dim:
                    pad = cond_dim - pooled_text_emb.shape[-1]
                    pooled_text_emb = F.pad(pooled_text_emb, (0, pad))
                else:
                    pooled_text_emb = pooled_text_emb[..., :cond_dim]

        if micro_conds is None:
            if self._latest_timesteps is None:
                raise RuntimeError(
                    "UVit wrapper requires timestep information; ensure hf_register_diffusion_wrapper was invoked"
                )
            timesteps = self._latest_timesteps.to(device=device, dtype=torch.float32)
            micro_encode_dim = getattr(config, "micro_cond_encode_dim", 1) or 1
            micro_embed_dim = getattr(config, "micro_cond_embed_dim", micro_encode_dim)
            repeat = max(1, micro_embed_dim // micro_encode_dim)
            micro_conds = timesteps.unsqueeze(1).repeat(1, repeat)
        else:
            micro_conds = micro_conds.to(device=device)

        input_ids = self._ensure_token_ids(input_ids)

        logits = self.transformer(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            pooled_text_emb=pooled_text_emb,
            micro_conds=micro_conds,
            cross_attention_kwargs=None,
        )

        target_channels = self.hf_config.out_channels or self.hf_config.in_channels
        if logits.shape[1] != target_channels:
            logits = logits[:, :target_channels]

        prediction = torch.moveaxis(logits, 1, -1).contiguous().to(dtype=batch.dtype)
        return {"logits": prediction}

    def _ensure_token_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(input_ids):
            tokens = input_ids.long()
        else:
            if input_ids.dim() == 4:
                values = input_ids.mean(dim=1)
            elif input_ids.dim() == 3:
                values = input_ids
            else:
                raise ValueError("UVit wrapper expects conditioning images shaped [B, C, H, W] or [B, H, W]")

            vocab_size = getattr(self.transformer.config, "vocab_size", 8192)
            tokens = ((values + 1.0) * (vocab_size - 1) / 2.0).round()
            tokens = tokens.clamp_(0, vocab_size - 1).to(dtype=torch.long)

        if tokens.dim() == 4 and tokens.shape[1] == 1:
            tokens = tokens.squeeze(1)
        return tokens
