# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-exact JiT replication configs for ImageNet pixel-space diffusion.

Wraps the JiT model port at :mod:`nvsubquadratic.networks.jit` (itself a
faithful port of `LTH14/JiT <https://github.com/LTH14/JiT>`__) with the
training recipe from the paper:

- Paper:  Li & He, "Back to Basics: Let Denoising Generative Models Denoise"
  (arxiv `2511.13720 <https://arxiv.org/abs/2511.13720>`__, 2025).
- Reference impl: ``~/projects/JiT/`` (PyTorch+GPU re-implementation by the
  authors).

These configs target the official JiT-paper setups (256×256 / 512×512,
patch 16 / 32, 600 epochs).  The Lightning wrapper is the existing
:class:`~experiments.lightning_wrappers.diffusion_wrapper.DiffusionWrapper`,
opted into three JiT-exactness knobs (defaults are off so legacy non-JiT
configs keep working unchanged):

- ``diffusion.clamp_target_v=True`` (+ ``t_eps_clamp=5e-2``): matches
  JiT's v-loss target clamping ``(x - z) / (1 - t).clamp_min(t_eps)``.
- ``diffusion.network_handles_conditioning=True``: the wrapper bypasses
  its own time-embedding MLP and label embedding table; ``JiT`` uses its
  own ``TimestepEmbedder`` + ``LabelEmbedder`` instead (matching the
  reference's exact shapes and bringing both embedders into the EMA
  tracker — they were not EMA-tracked when held in the wrapper).
- ``diffusion.ema_decay_secondary=0.9996``: tracks JiT's second EMA in
  addition to the primary ``ema_decay=0.9999``.  Unused for sampling
  (matches the released JiT eval scripts) but checkpointed for parity
  and for Karras-style post-hoc EMA interpolation experiments.

With these flags the wrapper and the reference produce identical training
trajectories up to floating-point precision; see
``tests/networks/test_diffusion_wrapper_jit_exact.py`` for the pinned
behaviour and ``tests/networks/test_jit_modules.py`` for the
module-level byte-equivalence checks against the reference math.
"""
