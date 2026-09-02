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

r"""Freeze a pretrained backbone so only named parameter groups train.

Required for parameter-efficient fine-tuning arms.  Without it every "head only"
or "head + adapter" arm silently trains the full network, making those arms
indistinguishable from a full fine-tune — a failure mode that is invisible in a
loss curve and easy to mistake for a scientific result.

The callback runs in :meth:`setup`, which Lightning invokes **before**
``configure_optimizers``.  That ordering matters: the optimizer's param-group
builder in
:mod:`experiments.lightning_wrappers.base_lightning_wrapper` skips parameters
with ``requires_grad=False``, so freezing afterwards would leave the frozen
tensors in the optimizer with live momentum state.

Usage::

    config.callbacks = [
        LazyConfig(FreezeBackboneCallback)(trainable_patterns=["out_proj"]),
    ]

A ``ValueError`` is raised if no parameter matches ``trainable_patterns`` — a
typo that silently freezes the entire network would otherwise produce a run
that trains nothing at all.
"""

import pytorch_lightning as pl


# Adapter parameter name fragments, matching
# nvsubquadratic.modules.vit5_parallel_hyena_adapter.ViT5ParallelHyenaSequenceMixer.
ADAPTER_PATTERNS = ("w_down", "w_up", "inner_mixer")

# Classification head of ViT5ClassificationNet.
HEAD_PATTERNS = ("out_proj", "out_norm")


class FreezeBackboneCallback(pl.Callback):
    r"""Freeze all parameters except those whose name matches a trainable pattern.

    Args:
        trainable_patterns: Substrings identifying parameters to keep trainable.
            A parameter trains if any pattern is a substring of its qualified
            name.  Defaults to head + adapter parameters
            (``out_proj``, ``out_norm``, ``w_down``, ``w_up``, ``inner_mixer``).
        verbose: Print a per-group summary of what was frozen at setup time.

    Attributes:
        trainable_patterns (tuple[str, ...]): Patterns kept trainable.
        num_trainable (int): Trainable element count, populated during setup.
        num_total (int): Total element count, populated during setup.
    """

    def __init__(
        self,
        trainable_patterns: list[str] | tuple[str, ...] | None = None,
        verbose: bool = True,
    ) -> None:
        """Store the trainable-name patterns."""
        super().__init__()
        if trainable_patterns is None:
            trainable_patterns = HEAD_PATTERNS + ADAPTER_PATTERNS
        self.trainable_patterns = tuple(trainable_patterns)
        self.verbose = verbose
        self.num_trainable = 0
        self.num_total = 0

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Freeze non-matching parameters before the optimizer is built.

        Args:
            trainer: Lightning trainer.
            pl_module: Module wrapping the network.
            stage: Lightning stage; freezing is applied for ``"fit"`` only.

        Raises:
            ValueError: If no parameter name matches ``trainable_patterns``.
        """
        if stage != "fit":
            return

        matched: list[str] = []
        for name, param in pl_module.named_parameters():
            keep = any(pattern in name for pattern in self.trainable_patterns)
            param.requires_grad = keep
            if keep:
                matched.append(name)

        if not matched:
            available = sorted({n.split(".")[-2] for n, _ in pl_module.named_parameters()})
            raise ValueError(
                f"FreezeBackboneCallback: no parameter matched {self.trainable_patterns}. "
                f"The whole network would be frozen and training would be a no-op. "
                f"Available name components include: {available[:20]}"
            )

        self.num_trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        self.num_total = sum(p.numel() for p in pl_module.parameters())

        if self.verbose:
            pct = 100.0 * self.num_trainable / max(1, self.num_total)
            print(
                f"[freeze] patterns={self.trainable_patterns} -> "
                f"{len(matched)} tensors, {self.num_trainable:,}/{self.num_total:,} params ({pct:.2f}%) trainable"
            )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log the trainable fraction to the experiment logger."""
        if self.num_total and trainer.logger is not None:
            trainer.logger.log_hyperparams(
                {
                    "freeze/trainable_params": self.num_trainable,
                    "freeze/total_params": self.num_total,
                    "freeze/trainable_fraction": self.num_trainable / self.num_total,
                }
            )
