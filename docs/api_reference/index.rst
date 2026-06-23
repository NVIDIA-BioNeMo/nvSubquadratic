.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. module:: nvsubquadratic

API Reference
=============

Organised bottom-up: low-level convolution primitives first, then the
mixer modules that compose them, then full networks, then the
parallel, core utility, and experiments layers.

Start with the :doc:`Ops Primer <../ops/README>` — the math motivation
behind the FFT-based ops (the convolution theorem, the linear/circular
flavours, and a decision tree for picking a function).  ``docs-tracker.md``
at the repo root tracks the documentation coverage plan.

.. toctree::
   :maxdepth: 2

   Ops Primer <../ops/README>
   ops
   modules
   networks
   parallel
   core
   experiments
