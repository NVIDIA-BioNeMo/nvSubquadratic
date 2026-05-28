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

# TODO: Add license header here

"""Custom LR scheduler utilities.

Currently provides :class:`ResumableSequentialLR`, a bug-fixed subclass of
:class:`torch.optim.lr_scheduler.SequentialLR` that correctly restores the
learning rate to the optimizer's ``param_groups`` after a checkpoint resume.

**Background**

A typical training schedule consists of multiple phases — e.g. a linear warmup
followed by a cosine decay.  PyTorch's ``SequentialLR`` chains together a list of
sub-schedulers and advances through them when configured milestone epochs are
reached.  However, as of PyTorch 2.10 its ``load_state_dict`` method restores
the scheduler's internal bookkeeping but silently omits propagating the restored
LR back to the optimizer, causing the schedule to restart from the initial warmup
LR after every checkpoint resume.

:class:`ResumableSequentialLR` patches this by calling
``optimizer.param_groups[i]["lr"] = _last_lr[i]`` immediately after the parent
``load_state_dict`` completes.
"""

import torch


class ResumableSequentialLR(torch.optim.lr_scheduler.SequentialLR):
    """``SequentialLR`` with a corrected ``load_state_dict``.

    Bug (PyTorch <= 2.10, confirmed on 2.10.0+cu129):
        ``SequentialLR.load_state_dict`` correctly deserializes its internal
        bookkeeping (``_last_lr``, sub-scheduler states, ``last_epoch``) but
        **never writes the restored learning rates back to
        ``optimizer.param_groups``**.  As a result, after loading a checkpoint
        the optimizer silently continues with the LR that the freshly
        constructed scheduler initialized (typically the warmup start value),
        rather than the LR the training had reached before the checkpoint was
        saved.  In practice this means the LR schedule restarts from zero on
        every job resume.

    Fix:
        After the parent ``load_state_dict`` finishes, copy ``_last_lr`` into
        the optimizer's ``param_groups`` so the next ``optimizer.step()`` uses
        the correct restored learning rate.

    See ``tests/test_checkpoint_resume.py::TestResumableSequentialLR`` for
    round-trip verification and a sentinel test that confirms the upstream bug
    still exists.
    """

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state and propagate restored LRs to the optimizer.

        Calls the parent ``SequentialLR.load_state_dict``, then immediately
        copies each value in ``self._last_lr`` into the corresponding
        ``optimizer.param_groups[i]["lr"]``.  This ensures that the first
        ``optimizer.step()`` after a resume uses the learning rate that was
        active when the checkpoint was saved, rather than the freshly
        initialised (warmup-start) value.

        Args:
            state_dict: Scheduler state dictionary as produced by
                :meth:`state_dict`.  Typically loaded from a checkpoint with
                ``torch.load`` and passed directly to this method.
        """
        super().load_state_dict(state_dict)
        if hasattr(self, "_last_lr") and self._last_lr:
            for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
                param_group["lr"] = lr
