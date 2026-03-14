"""Composite Muon + AdamW optimizer for training with a single optimizer interface.

Muon (MomentUm Orthogonalized by Newton-schulz) applies orthogonalized updates
to 2D hidden-layer weight matrices, yielding faster convergence per step.
All other parameters (embeddings, biases, norms, classifier heads) are handled
by AdamW.

With ``adjust_lr_fn="match_rms_adamw"`` (Moonshot), Muon re-scales its update
so that both sub-optimizers can share the same learning rate and weight decay.
This means the existing LR scheduler works on the combined ``param_groups``
without any special handling.

Usage::

    optimizer = MuonAdamW(model.named_parameters(), lr=4e-3, weight_decay=0.05)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def _is_muon_eligible(name: str, param: nn.Parameter) -> bool:
    """Return True if a parameter should be optimized with Muon.

    Criteria:
    - Must be 2D (weight matrix)
    - Must not be an embedding (name contains "embed")
    - Must not be a classifier head (name contains "head" or "classifier")
    - Must not be a positional embedding linear (inside SIRENPositionalEmbeddingND)
    """
    if param.ndim != 2:
        return False
    name_lower = name.lower()
    exclude_patterns = ("embed", "head", "classifier", "cls_token", "pos_embed")
    return not any(pat in name_lower for pat in exclude_patterns)


class MuonAdamW(torch.optim.Optimizer):
    """Composite optimizer: Muon for 2D hidden-layer weights, AdamW for the rest.

    Presents a single ``torch.optim.Optimizer`` interface so that LR schedulers,
    Lightning's ``configure_optimizers``, and checkpoint saving all work unchanged.

    Internally maintains two sub-optimizers and delegates ``step``, ``zero_grad``,
    ``state_dict``, and ``load_state_dict`` to both.

    Args:
        params: Iterable of named parameters (use ``model.named_parameters()``).
            Also accepts plain parameter iterables or param-group dicts, but
            named parameters are required for automatic Muon/AdamW splitting.
        lr: Learning rate for both sub-optimizers.
        weight_decay: Weight decay for both sub-optimizers.
        betas: AdamW beta coefficients.
        eps: AdamW epsilon.
        muon_momentum: Muon momentum factor.
        muon_ns_steps: Number of Newton-Schulz iterations.
        muon_nesterov: Whether to use Nesterov momentum in Muon.
    """

    def __init__(  # noqa: D107
        self,
        params: Iterable,
        lr: float = 4e-3,
        weight_decay: float = 0.05,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_nesterov: bool = True,
    ):
        # Consume the iterator once and classify parameters
        muon_groups, adamw_groups = self._split_params(params, weight_decay=weight_decay)

        n_muon = sum(len(g["params"]) for g in muon_groups)
        n_adamw = sum(len(g["params"]) for g in adamw_groups)
        logger.info("MuonAdamW: %d params -> Muon, %d params -> AdamW", n_muon, n_adamw)

        self._muon = torch.optim.Muon(
            muon_groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            adjust_lr_fn="match_rms_adamw",
        )
        self._adamw = torch.optim.AdamW(
            adamw_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

        # Build a combined defaults dict for the Optimizer base class.
        # We use dummy param groups here -- the real groups are in the sub-optimizers.
        defaults = {"lr": lr, "weight_decay": weight_decay}
        # Bypass normal Optimizer.__init__ param processing: just init the base.
        super().__init__([{"params": []}], defaults)
        # Remove the empty dummy group that Optimizer.__init__ created.
        self.param_groups.clear()

        # Expose combined param_groups so schedulers can iterate them.
        # Tag each group so we know which sub-optimizer owns it.
        for g in self._muon.param_groups:
            g["_optimizer"] = "muon"
            self.param_groups.append(g)
        for g in self._adamw.param_groups:
            g["_optimizer"] = "adamw"
            self.param_groups.append(g)

    @staticmethod
    def _split_params(
        params: Iterable,
        weight_decay: float,
    ) -> tuple[list[dict], list[dict]]:
        """Split named parameters into Muon-eligible and AdamW groups.

        Respects per-parameter ``_weight_decay`` and ``_no_weight_decay`` flags.
        """
        muon_wd: list[nn.Parameter] = []
        muon_no_wd: list[nn.Parameter] = []
        muon_custom_wd: dict[float, list[nn.Parameter]] = {}
        adamw_wd: list[nn.Parameter] = []
        adamw_no_wd: list[nn.Parameter] = []
        adamw_custom_wd: dict[float, list[nn.Parameter]] = {}
        seen: set[int] = set()

        named_params = list(params)
        # Support both named_parameters() tuples and plain parameter lists
        if named_params and isinstance(named_params[0], nn.Parameter):
            named_params = [(f"param_{i}", p) for i, p in enumerate(named_params)]

        for name, param in named_params:
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)

            use_muon = _is_muon_eligible(name, param)
            custom_wd = getattr(param, "_weight_decay", None)
            no_wd = getattr(param, "_no_weight_decay", False)

            if use_muon:
                if custom_wd is not None:
                    muon_custom_wd.setdefault(custom_wd, []).append(param)
                elif no_wd:
                    muon_no_wd.append(param)
                else:
                    muon_wd.append(param)
            else:
                if custom_wd is not None:
                    adamw_custom_wd.setdefault(custom_wd, []).append(param)
                elif no_wd:
                    adamw_no_wd.append(param)
                else:
                    adamw_wd.append(param)

        def _build_groups(
            wd_params: list, no_wd_params: list, custom_wd_buckets: dict, default_wd: float
        ) -> list[dict]:
            groups = []
            if wd_params:
                groups.append({"params": wd_params, "weight_decay": default_wd})
            if no_wd_params:
                groups.append({"params": no_wd_params, "weight_decay": 0.0})
            for wd_val, ps in sorted(custom_wd_buckets.items()):
                groups.append({"params": ps, "weight_decay": wd_val})
            return groups

        muon_groups = _build_groups(muon_wd, muon_no_wd, muon_custom_wd, weight_decay)
        adamw_groups = _build_groups(adamw_wd, adamw_no_wd, adamw_custom_wd, weight_decay)

        # Ensure at least one group for each sub-optimizer (required by PyTorch)
        if not muon_groups:
            muon_groups = [{"params": [], "weight_decay": weight_decay}]
        if not adamw_groups:
            adamw_groups = [{"params": [], "weight_decay": weight_decay}]

        return muon_groups, adamw_groups

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        """Perform a single optimization step on both sub-optimizers."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._muon.step()
        self._adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Reset gradients for both sub-optimizers."""
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return a merged state dict for checkpointing."""
        return {
            "muon": self._muon.state_dict(),
            "adamw": self._adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a merged state dict from a checkpoint."""
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])
        # Re-sync param_groups references
        self.param_groups.clear()
        for g in self._muon.param_groups:
            g["_optimizer"] = "muon"
            self.param_groups.append(g)
        for g in self._adamw.param_groups:
            g["_optimizer"] = "adamw"
            self.param_groups.append(g)
