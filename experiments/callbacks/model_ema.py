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

"""EMA callbacks for model training.

Re-exports Lightning's ``EMAWeightAveraging`` (requires Lightning >= 2.6) and
provides ``LabeledEMAWeightAveraging`` which additionally sets a metric suffix
on the LightningModule so that validation metrics logged with EMA weights are
clearly named ``val/acc_ema``, ``val/loss_ema``, etc.

Internally uses ``torch.optim.swa_utils.AveragedModel`` and swaps via
``param.data.copy_()`` — compatible with ``torch.compile``.

Usage in experiment config::

    from experiments.callbacks.model_ema import LabeledEMAWeightAveraging
    config.callbacks = [LazyConfig(LabeledEMAWeightAveraging)(decay=0.99996)]
"""

import pytorch_lightning as pl


try:
    from pytorch_lightning.callbacks import EMAWeightAveraging
except ImportError:
    from lightning.pytorch.callbacks.weight_averaging import EMAWeightAveraging  # type: ignore[import-not-found]


class LabeledEMAWeightAveraging(EMAWeightAveraging):
    """``EMAWeightAveraging`` that labels validation metrics with an ``_ema`` suffix.

    Sets ``pl_module._val_metric_suffix = "_ema"`` while EMA weights are
    swapped in for validation, so that any wrapper that honours the suffix
    (e.g. ``ClassificationWrapper``) logs to ``val/acc_ema`` instead of
    ``val/acc``.
    """

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Swap in EMA weights and set metric suffix before validation."""
        super().on_validation_epoch_start(trainer, pl_module)
        if self._average_model is not None:
            pl_module._val_metric_suffix = "_ema"

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Restore training weights after validation; suffix is left for module hooks."""
        # Don't reset suffix here — PL calls callback hooks before module hooks,
        # so the module's on_validation_epoch_end still needs the suffix to read
        # the correct metric keys.  The suffix is re-set every epoch in
        # on_validation_epoch_start, and the __init__ default is "".
        super().on_validation_epoch_end(trainer, pl_module)


__all__ = ["EMAWeightAveraging", "LabeledEMAWeightAveraging"]
