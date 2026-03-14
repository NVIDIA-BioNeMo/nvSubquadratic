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
    """

    def __init__(  # noqa: D107
        self,
        log_every_n_steps: int = 50,
        num_film_layers: int = 3,
        film_on_pos_embed: bool = True,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.num_film_layers = num_film_layers
        self.film_on_pos_embed = film_on_pos_embed

        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._film_outputs: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._pre_act: dict[int, torch.Tensor] = {}

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

        for block_id in sorted(self._film_outputs.keys()):
            film_out = self._film_outputs[block_id]
            report_lines.append(f"--- Block {block_id} ---")

            for i, (gamma, beta) in enumerate(film_out):
                is_posemb = self.film_on_pos_embed and i == 0
                label = "POSEMB" if is_posemb else f"HIDDEN_{i - (1 if self.film_on_pos_embed else 0)}"

                with torch.no_grad():
                    line, stats = self._format_layer(gamma, beta, label)
                    report_lines.append(line)

                    if is_posemb and block_id in self._pre_act:
                        sin_line, sin_stats = self._format_sin_arg(gamma, beta, self._pre_act[block_id])
                        report_lines.append(sin_line)
                        posemb_sin_disruptions.append(sin_stats["disruption_mean"])

                    if is_posemb:
                        posemb_gamma_stds.append(stats["gamma_std"])
                        posemb_neg_fracs.append(stats["neg_frac"])

            # Weight norms
            from nvsubquadratic.modules.film import KernelFiLMGenerator

            network = pl_module.network
            if hasattr(network, "_orig_mod"):
                network = network._orig_mod
            for name, module in network.named_modules():
                if isinstance(module, KernelFiLMGenerator) and f"blocks.{block_id}." in name:
                    n_in = module.mlp[0].weight.norm().item()
                    n_out = module.mlp[-1].weight.norm().item()
                    report_lines.append(f"  weights: in_norm={n_in:.2f}  out_norm={n_out:.2f}")
                    w_norm_outs.append(n_out)
                    break

            report_lines.append("")

        report_text = "\n".join(report_lines)
        html = f"<pre>{report_text}</pre>"
        trainer.logger.experiment.log({"film_report": wandb.Html(html), "trainer/global_step": trainer.global_step})

        # A few key scalars for chart overlay with loss
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
        stats = {"gamma_std": g_std, "neg_frac": neg_frac}
        return line, stats

    @staticmethod
    def _format_sin_arg(gamma: torch.Tensor, beta: torch.Tensor, pre_act: torch.Tensor) -> tuple[str, dict]:
        """Summary of the actual sin() argument after FiLM modulation."""
        spatial = pre_act.shape[1:-1]
        H = spatial[0]
        W = spatial[1] if len(spatial) > 1 else 1

        # Sample a few spatial positions (center + corners)
        positions = [(H // 2, W // 2)]
        if H > 1 and W > 1:
            positions += [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]

        sin_args = []
        for h, w in positions:
            pa = pre_act[0, h, w, :] if len(spatial) > 1 else pre_act[0, h, :]
            sin_args.append(gamma * pa.unsqueeze(0) + beta)  # [B, D]

        sin_arg = torch.cat(sin_args, dim=0)

        abs_mean = sin_arg.abs().mean().item()
        abs_max = sin_arg.abs().max().item()
        frac_gt_pi = (sin_arg.abs() > math.pi).float().mean().item()
        frac_gt_2pi = (sin_arg.abs() > 2 * math.pi).float().mean().item()

        # Disruption vs identity (gamma=1, beta=0)
        identity_args = []
        for h, w in positions:
            pa = pre_act[0, h, w, :] if len(spatial) > 1 else pre_act[0, h, :]
            identity_args.append(pa.unsqueeze(0))  # [1, D]
        identity = torch.cat(identity_args, dim=0)  # [num_pos, D]
        base_sin = torch.sin(identity)
        mod_sin = torch.sin(sin_arg[: len(positions)])
        disruption = (mod_sin - base_sin).abs().mean().item()
        disruption_max = (mod_sin - base_sin).abs().max().item()

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
