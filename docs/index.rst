.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

nvSubquadratic Documentation
============================

``nvsubquadratic`` is a unified PyTorch-native library for subquadratic
alternatives to quadratic attention. It consolidates efforts from across
NVIDIA Research teams (nvResearch, NeMo, BioNeMo) into a single, consistent
API. The current release supports multi-dimensional (1D, 2D, 3D) Hyena
operators backed by optimized CUDA kernels from
:mod:`subquadratic_ops_torch`. Hyena operators provide subquadratic
alternatives to attention, achieving ``O(N log N)`` complexity compared with
``O(N^2)`` for traditional attention.

Installation
------------

The package is installed from source:

.. code-block:: bash

    pip install -e .

To enable the optional fused RMSNorm kernel on Hopper / Blackwell GPUs:

.. code-block:: bash

    pip install -e ".[quack]"

Requirements
------------

- CUDA-compatible NVIDIA GPU (Ampere or Hopper architecture)
- CUDA Toolkit 12.0 or higher
- Python 3.11 or higher

Where to go next
----------------

- **Ops Overview** — a math primer and decision tree for the FFT
  convolution primitives at the bottom of the stack.
- **API Reference** — auto-generated reference for the curated public
  surface (ops + mixer modules).

Related projects
----------------

``nvsubquadratic`` is the high-level PyTorch interface; the underlying
CUDA kernels live in a separate library:

- `subquadratic-ops <https://nvidia-digital-bio.github.io/subquadraticOps-docs/>`_ —
  optimized CUDA kernels (causal conv1d, FFT conv1d/2d, B2B causal conv1d,
  implicit filters, rearrange) that nvSubquadratic delegates to via
  :mod:`subquadratic_ops_torch`. Refer to its API reference for kernel-level
  signatures, supported dtypes, and GPU-architecture coverage.

.. toctree::
   :maxdepth: 1
   :hidden:

   Overview <self>

.. toctree::
   :maxdepth: 2

   Ops Overview <ops/README>
   API Reference <api>
