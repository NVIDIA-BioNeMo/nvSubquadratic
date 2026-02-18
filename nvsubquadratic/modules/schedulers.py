# TODO: Add license header here


"""Chained scheduler implementation for PyTorch.

This allows chaining multiple learning rate schedulers, e.g., LinearWarmupLR and CosineAnnealingLR.
"""

import torch



class WSDScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Warmup-Stable-Decay Learning Rate Scheduler.

    Reference: MiniCPM (Hu et al., 2024)
    https://arxiv.org/abs/2404.06395

    The schedule consists of three phases:
    1. Warmup: Linear increase from 0 to peak LR.
    2. Stable: Constant peak LR.
    3. Decay: Linear decay from peak LR to min LR.
    """

    def __init__(
        self,
        optimizer,
        total_iterations: int,
        warmup_iterations: int = 0,
        decay_iterations_percentage: float = 0.1,
        min_lr_ratio: float = 0.01,
        last_epoch: int = -1,
    ):
        self.total_iterations = total_iterations
        self.warmup_iterations = warmup_iterations
        self.decay_iterations = int(total_iterations * decay_iterations_percentage)
        self.stable_iterations = total_iterations - self.decay_iterations
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_iterations:
             # Warmup phase: linear increase from 0 (or very small) to peak
             # We use max(1, ...) to avoid division by zero
             alpha = step / max(1, self.warmup_iterations)
             return [base_lr * alpha for base_lr in self.base_lrs]
        
        elif step < self.stable_iterations:
            # Stable phase: maintain peak LR
            return [base_lr for base_lr in self.base_lrs]
        else:
            # Decay phase: linear decay to min_lr_ratio
            decay_step = step - self.stable_iterations
            # Prevent division by zero if decay_iterations is 0
            decay_denom = max(self.decay_iterations, 1)
            # Ensure progress is capped at 1.0 (though step usually stops at total_iterations)
            decay_progress = min(decay_step / decay_denom, 1.0)
            lr_ratio = 1.0 - (1.0 - self.min_lr_ratio) * decay_progress
            return [base_lr * lr_ratio for base_lr in self.base_lrs]


class ChainedScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Chains list of learning rate schedulers.

    It takes a list of chainable learning rate schedulers and performs consecutive step() functions belong to them by just one call.

    Args:
        schedulers (list): List of chained schedulers.

    Example:
        # >>> # Assuming optimizer uses lr = 1. for all groups
        # >>> # lr = 0.09     if epoch == 0
        # >>> # lr = 0.081    if epoch == 1
        # >>> # lr = 0.729    if epoch == 2
        # >>> # lr = 0.6561   if epoch == 3
        # >>> # lr = 0.59049  if epoch >= 4
        # >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        # >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        # >>> scheduler = ChainedScheduler([scheduler1, scheduler2])
        # >>> for epoch in range(100):
        # >>>     train(...)
        # >>>     validate(...)
        # >>>     scheduler.step()
    """

    def __init__(self, schedulers):
        """Initialize the ChainedScheduler.

        Args:
            schedulers: List of schedulers to chain.
        """
        for scheduler_idx in range(1, len(schedulers)):
            if schedulers[scheduler_idx].optimizer != schedulers[0].optimizer:
                raise ValueError(
                    "ChainedScheduler expects all schedulers to belong to the same optimizer, but "
                    "got schedulers at index {} and {} to be different".format(0, scheduler_idx)
                )
        self._schedulers = list(schedulers)
        self.optimizer = self._schedulers[0].optimizer

    def step(self):
        """Step the schedulers."""
        for scheduler in self._schedulers:
            scheduler.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ("optimizer", "_schedulers")}
        state_dict["_schedulers"] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict["_schedulers"][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop("_schedulers")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["_schedulers"] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)
