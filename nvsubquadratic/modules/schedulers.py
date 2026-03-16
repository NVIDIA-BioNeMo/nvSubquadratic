# TODO: Add license header here

"""Custom LR scheduler utilities."""

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

    def load_state_dict(self, state_dict):
        """Load state and apply restored LR to optimizer param groups."""
        super().load_state_dict(state_dict)
        if hasattr(self, "_last_lr") and self._last_lr:
            for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
                param_group["lr"] = lr
