.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. module:: nvsubquadratic

API Reference
=============

Organised bottom-up: low-level convolution primitives first, then the
mixer modules that compose them, then full networks, then the
parallel, core utility, and experiments layers.

See `ops/README.md <../ops/README.html>`_ for the math motivation behind
the FFT-based ops, and ``docs-tracker.md`` at the repo root for the
documentation coverage plan.

.. toctree::
   :maxdepth: 2

   ops
   modules
   networks
   parallel
   core
   experiments
