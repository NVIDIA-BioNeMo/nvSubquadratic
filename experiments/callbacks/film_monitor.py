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

"""Callback to monitor FiLM conditioning parameters during training.

Registers forward hooks on KernelFiLMGenerator and SIRENPositionalEmbeddingND
modules to capture actual gamma/beta output statistics and the effective
sin() argument magnitude. Logs a compact text report to wandb plus a few
key scalars for chart overlay with loss.

Tracked failure modes:
    1. Gamma sign flip (effective multiplier < 0)
    2. Omega-0 explosion (gamma magnitude growing large)
    3. Phase randomization (beta >> pi)
    4. Batch divergence (gamma varies wildly across inputs)
    5. Sin-argument overflow (argument to sin exceeds safe range)

Also tracks:
    - Weight delta from initialization (how much FiLM weights moved during finetuning)
    - Per-block input dependence (batch_std of gamma — higher means more input conditioning)

Ported from commit 7bad43f (dwromero/muon-optimizer branch), extended with
weight-delta tracking and film_after_pos_embed-aware disruption analysis.
"""

from __future__ import annotations

import math
import re

import pytorch_lightning as pl
import torch


class FiLMMonitorCallback(pl.Callback):
    """Logs a compact FiLM diagnostic text report to wandb.

    Args:
        log_every_n_steps: How often to log (in global steps).
        num_film_layers: Number of FiLM layers per generator.
        film_on_pos_embed: If True, the first FiLM layer is pos-embed.
        film_after_pos_embed: If True, the pos-embed FiLM is applied *after*
            sin() (i.e. ``gamma * sin(x) + beta``), not before.
    """

    def __init__(  # noqa: D107
        self,
        log_every_n_steps: int = 50,
        num_film_layers: int = 3,
        film_on_pos_embed: bool = True,
        film_after_pos_embed: bool = True,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.num_film_layers = num_film_layers
        self.film_on_pos_embed = film_on_pos_embed
        self.film_after_pos_embed = film_after_pos_embed

        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._film_outputs: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._pre_act: dict[int, torch.Tensor] = {}
        # Snapshot of initial FiLM weights for delta tracking
        self._init_weights: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D102
        from nvsubquadratic.modules.film import KernelFiLMGenerator
        from nvsubquadratic.modules.kernels_nd import SIRENPositionalEmbeddingND

        network = pl_module.network
        if hasattr(network, "_orig_mod"):
            network = network._orig_mod

        n_film = 0
        n_posemb = 0
        for name, module in network.named_modules():
            block_match = re.search(r"blocks\.(\d+)", name)
            if block_match is None:
                continue
            block_id = int(block_match.group(1))

            if isinstance(module, KernelFiLMGenerator):
                hook = module.register_forward_hook(self._film_hook(block_id))
                self._hooks.append(hook)
                # Snapshot initial weights for delta tracking
                for pname, param in module.named_parameters():
                    key = f"blocks.{block_id}.film.{pname}"
                    self._init_weights[key] = param.data.detach().clone()
                n_film += 1

            if isinstance(module, SIRENPositionalEmbeddingND):
                hook = module.linear.register_forward_hook(self._linear_hook(block_id))
                self._hooks.append(hook)
                n_posemb += 1

        if trainer.is_global_zero:
            print(f"[FiLMMonitor] Hooks: {n_film} FiLM generators, {n_posemb} pos-embed linears")

    def _film_hook(self, block_id: int):
        def fn(_module, _input, output):
            self._film_outputs[block_id] = output

        return fn

    def _linear_hook(self, block_id: int):
        def fn(_module, _input, output):
            self._pre_act[block_id] = output.detach()

        return fn

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: D102
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        if not self._film_outputs:
            return
        if not trainer.is_global_zero:
            return

        import wandb

        report_lines = [f"step={trainer.global_step}  epoch={trainer.current_epoch}"]
        report_lines.append("")

        posemb_gamma_stds = []
        posemb_sin_disruptions = []
        posemb_neg_fracs = []
        w_norm_outs = []
        w_deltas = []
        batch_stds = []

        network = pl_module.network
        if hasattr(network, "_orig_mod"):
            network = network._orig_mod

        for block_id in sorted(self._film_outputs.keys()):
            film_out = self._film_outputs[block_id]
            report_lines.append(f"--- Block {block_id} ---")

            for i, (gamma, beta) in enumerate(film_out):
                is_posemb = self.film_on_pos_embed and i == 0
                label = "POSEMB" if is_posemb else f"HIDDEN_{i - (1 if self.film_on_pos_embed else 0)}"

                with torch.no_grad():
                    line, stats = self._format_layer(gamma, beta, label)
                    report_lines.append(line)
                    batch_stds.append(stats["batch_std"])

                    if is_posemb and block_id in self._pre_act:
                        sin_line, sin_stats = self._format_sin_arg(
                            gamma,
                            beta,
                            self._pre_act[block_id],
                            after_sin=self.film_after_pos_embed,
                        )
                        report_lines.append(sin_line)
                        posemb_sin_disruptions.append(sin_stats["disruption_mean"])

                    if is_posemb:
                        posemb_gamma_stds.append(stats["gamma_std"])
                        posemb_neg_fracs.append(stats["neg_frac"])

            # Weight norms and deltas from init
            from nvsubquadratic.modules.film import KernelFiLMGenerator

            for name, module in network.named_modules():
                if isinstance(module, KernelFiLMGenerator) and f"blocks.{block_id}." in name:
                    n_in = module.mlp[0].weight.norm().item()
                    n_out = module.mlp[-1].weight.norm().item()
                    # Compute L2 delta from initial weights
                    total_delta_sq = 0.0
                    total_norm_sq = 0.0
                    for pname, param in module.named_parameters():
                        key = f"blocks.{block_id}.film.{pname}"
                        if key in self._init_weights:
                            total_delta_sq += (
                                (param.data - self._init_weights[key].to(param.device)).pow(2).sum().item()
                            )
                            total_norm_sq += self._init_weights[key].pow(2).sum().item()
                    rel_delta = math.sqrt(total_delta_sq) / (math.sqrt(total_norm_sq) + 1e-8)
                    report_lines.append(
                        f"  weights: in_norm={n_in:.2f}  out_norm={n_out:.2f}"
                        f"  delta_from_init={math.sqrt(total_delta_sq):.4f} (rel={rel_delta:.4f})"
                    )
                    w_norm_outs.append(n_out)
                    w_deltas.append(rel_delta)
                    break

            report_lines.append("")

        report_text = "\n".join(report_lines)
        html = f"<pre>{report_text}</pre>"
        trainer.logger.experiment.log({"film_report": wandb.Html(html), "trainer/global_step": trainer.global_step})

        # Key scalars for chart overlay with loss
        scalars = {}
        if posemb_gamma_stds:
            scalars["film/posemb_gamma_std"] = sum(posemb_gamma_stds) / len(posemb_gamma_stds)
        if posemb_neg_fracs:
            scalars["film/posemb_gamma_neg_frac"] = sum(posemb_neg_fracs) / len(posemb_neg_fracs)
        if posemb_sin_disruptions:
            scalars["film/posemb_sin_disruption"] = sum(posemb_sin_disruptions) / len(posemb_sin_disruptions)
        if w_norm_outs:
            scalars["film/w_norm_out_avg"] = sum(w_norm_outs) / len(w_norm_outs)
            scalars["film/w_norm_out_max"] = max(w_norm_outs)
        if w_deltas:
            scalars["film/weight_delta_rel_avg"] = sum(w_deltas) / len(w_deltas)
            scalars["film/weight_delta_rel_max"] = max(w_deltas)
        if batch_stds:
            scalars["film/input_dependence_avg"] = sum(batch_stds) / len(batch_stds)
            scalars["film/input_dependence_max"] = max(batch_stds)
        if scalars:
            pl_module.log_dict(scalars, on_step=True, on_epoch=False, rank_zero_only=True)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_layer(gamma: torch.Tensor, beta: torch.Tensor, label: str) -> tuple[str, dict]:
        """One-line summary for a FiLM layer."""
        B = gamma.shape[0]
        g_mean = gamma.mean().item()
        g_std = gamma.std().item()
        g_min = gamma.min().item()
        g_max = gamma.max().item()
        neg_frac = (gamma < 0).float().mean().item()

        b_abs = beta.abs().mean().item()
        b_over_pi = b_abs / math.pi

        batch_std = gamma.mean(dim=-1).std().item() if B > 1 else 0.0

        line = (
            f"  {label:8s}: "
            f"gamma [{g_min:+.4f}, {g_max:+.4f}] mean={g_mean:+.4f} std={g_std:.4f} neg={neg_frac:.1%} "
            f"batch_std={batch_std:.4f} | "
            f"beta [{beta.min().item():+.4f}, {beta.max().item():+.4f}] "
            f"|β|={b_abs:.4f} |β|/π={b_over_pi:.3f}"
        )
        stats = {"gamma_std": g_std, "neg_frac": neg_frac, "batch_std": batch_std}
        return line, stats

    @staticmethod
    def _format_sin_arg(
        gamma: torch.Tensor,
        beta: torch.Tensor,
        pre_act: torch.Tensor,
        *,
        after_sin: bool = False,
    ) -> tuple[str, dict]:
        """Summary of the effective modulation on the positional embedding.

        When ``after_sin=False`` (FiLM before sin): analyses ``sin(gamma * x + beta)``
        vs ``sin(x)``.
        When ``after_sin=True`` (film_after_pos_embed): analyses ``gamma * sin(x) + beta``
        vs ``sin(x)``.
        """
        spatial = pre_act.shape[1:-1]
        H = spatial[0]
        W = spatial[1] if len(spatial) > 1 else 1

        positions = [(H // 2, W // 2)]
        if H > 1 and W > 1:
            positions += [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]

        pa_samples = []
        for h, w in positions:
            pa = pre_act[0, h, w, :] if len(spatial) > 1 else pre_act[0, h, :]
            pa_samples.append(pa.unsqueeze(0))

        pa_stack = torch.cat(pa_samples, dim=0)  # [num_pos, D]
        base_sin = torch.sin(pa_stack)  # sin(x) — identity baseline

        if after_sin:
            # film_after_pos_embed: gamma * sin(x) + beta
            mod_out = gamma[: len(positions)] * base_sin.unsqueeze(0) + beta[: len(positions)]
            # Use first batch element for display
            mod_display = mod_out[0]
            base_display = base_sin
        else:
            # FiLM before sin: sin(gamma * x + beta)
            sin_args = []
            for i, (h, w) in enumerate(positions):
                sin_args.append(gamma * pa_samples[i] + beta)  # [B, D]
            sin_arg = torch.cat(sin_args, dim=0)

            mod_display = torch.sin(sin_arg[: len(positions)])
            base_display = base_sin

        disruption = (mod_display - base_display).abs().mean().item()
        disruption_max = (mod_display - base_display).abs().max().item()

        if after_sin:
            # For after_sin, report stats on the modulated output directly
            abs_mean = mod_display.abs().mean().item()
            abs_max = mod_display.abs().max().item()
            frac_gt_1 = (mod_display.abs() > 1.0).float().mean().item()
            line = (
                f"           after_sin: |mean|={abs_mean:.2f} |max|={abs_max:.2f} "
                f">1.0={frac_gt_1:.1%} | "
                f"disruption: mean={disruption:.4f} max={disruption_max:.4f}"
            )
        else:
            sin_arg_flat = torch.cat([gamma * pa_samples[i] + beta for i, _ in enumerate(positions)], dim=0)
            abs_mean = sin_arg_flat.abs().mean().item()
            abs_max = sin_arg_flat.abs().max().item()
            frac_gt_pi = (sin_arg_flat.abs() > math.pi).float().mean().item()
            frac_gt_2pi = (sin_arg_flat.abs() > 2 * math.pi).float().mean().item()
            line = (
                f"           sin_arg: |mean|={abs_mean:.2f} |max|={abs_max:.2f} "
                f">π={frac_gt_pi:.1%} >2π={frac_gt_2pi:.1%} | "
                f"disruption: mean={disruption:.4f} max={disruption_max:.4f}"
            )

        stats = {"disruption_mean": disruption}
        return line, stats

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def on_fit_end(self, trainer, pl_module):  # noqa: D102
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._film_outputs.clear()
        self._pre_act.clear()
        self._init_weights.clear()
