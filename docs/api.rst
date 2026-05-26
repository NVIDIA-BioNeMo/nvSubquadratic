.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. module:: nvsubquadratic

API Reference
=============

The reference is organised bottom-up: low-level FFT convolution primitives
first, then the mixer modules that compose them, then full networks, then
parallel and utility helpers.  See `ops/README.md <ops/README.html>`_ for
the math motivation behind the FFT-based ops, and ``docs-tracker.md`` at
the repo root for the documentation coverage plan.

.. toctree::
   :maxdepth: 2

   api/ops
   api/modules
   api/networks
   api/parallel
   api/utilities
