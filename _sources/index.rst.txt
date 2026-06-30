.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

nvSubquadratic Documentation
============================

Attention is global, but it is quadratic and it ignores geometry.
Every token attends to every other token, so compute and memory grow as
``O(N^2)``. A 256×256 image is already 65k tokens, and video or 3D
volumes are out of reach. To apply attention to an image at all you have
to flatten the grid into a 1D sequence and let the model relearn that
neighbouring pixels are neighbours.

``nvsubquadratic`` is a unified, PyTorch-native library for subquadratic
alternatives to attention. They keep its global receptive field while
running in ``O(N log N)``, directly on native 1D / 2D / 3D
geometry. It consolidates efforts from across NVIDIA Research teams
(nvResearch, NeMo, BioNeMo) into a single, consistent API. The current
release centres on multi-dimensional **HyenaND** operators backed by
optimized CUDA kernels from :mod:`subquadratic_ops_torch`.

The figure below summarises the trade-off. *(Left)* Attention is natively
multi-dimensional but scales quadratically. Mamba is subquadratic but
inherently 1D, so it needs an ad-hoc 1D scan order to touch
multi-dimensional data, and no single ordering respects 2D locality.
HyenaND is global, natively multi-dimensional, and subquadratic at
the same time. *(Right)* That ``O(N log N)`` complexity is real wall-clock time:
HyenaND scales to million-token sequences while attention
collapses at long context.

If you are new to the library, start with :doc:`how_hyenand_works`, which builds
the operator up from attention in a few minutes. Then come back for install and the
package tour.

.. raw:: html

   <figure style="margin:1.5em 0">
     <div style="display:flex;align-items:flex-start;gap:2%">
       <div style="flex:0 0 64%">
         <table style="width:100%;border-collapse:collapse;text-align:center;table-layout:fixed">
           <tr>
             <th style="width:25%;font-weight:600;font-size:0.9em;padding-bottom:6px">Attention</th>
             <th colspan="2" style="width:50%;font-weight:600;font-size:0.9em;padding-bottom:6px">Mamba</th>
             <th style="width:25%;font-weight:600;font-size:0.9em;padding-bottom:6px">HyenaND (Ours)</th>
           </tr>
           <tr>
             <td style="padding:0 4px">
               <img src="_static/attn_hyena.jpg" style="width:100%;display:block" alt="Attention receptive field">
             </td>
             <td style="padding:0 4px">
               <img src="_static/mamba1_hyena.jpg" style="width:100%;display:block" alt="Mamba scan order 1">
             </td>
             <td style="padding:0 4px">
               <img src="_static/mamba2_hyena.jpg" style="width:100%;display:block" alt="Mamba scan order 2">
             </td>
             <td style="padding:0 4px">
               <img src="_static/hyena_hyena.jpg" style="width:100%;display:block" alt="HyenaND receptive field">
             </td>
           </tr>
           <tr>
             <td style="font-size:0.9em;padding-top:6px"><em>𝒪(L²)</em></td>
             <td colspan="2" style="font-size:0.9em;padding-top:6px"><em>𝒪(L)</em></td>
             <td style="font-size:0.9em;padding-top:6px"><em>𝒪(L log L)</em></td>
           </tr>
         </table>
       </div>
       <div style="flex:0 0 33%">
         <img src="_static/throughput_scaling.png" style="width:100%;display:block" alt="Forward time vs sequence length">
       </div>
     </div>
     <figcaption style="font-size:0.85em;margin-top:0.8em;color:#444">
       <strong>Figure 1.</strong>
       <em>(Left)</em> Receptive field and complexity of global operators by
       token count <em>L</em>: Attention <em>𝒪(L²)</em>, Mamba <em>𝒪(L)</em>,
       HyenaND <em>𝒪(L log L)</em>.
       <em>(Right)</em> Forward-pass time vs. sequence length for
       <code>flash-attention</code>, the official <code>mamba_chunk_scan_combined</code>
       Mamba2 kernel, and <code>nSubQ</code> (HyenaND).
     </figcaption>
   </figure>

Installation
------------

.. code-block:: bash

    pip install nvsubquadratic

This installs the full training/experiment stack — nvSubquadratic targets GPU
workflows.

Optional extras:

.. code-block:: bash

    pip install "nvsubquadratic[cuda]"         # accelerated fused FFT-conv / causal-conv CUDA kernels
    pip install "nvsubquadratic[quack]"        # fused RMSNorm kernel (Hopper/Blackwell only)
    pip install "nvsubquadratic[dali]"         # NVIDIA DALI data pipelines for the examples
    pip install "nvsubquadratic[distributed]"  # megatron-core, for context-parallel / distributed training
    pip install "nvsubquadratic[baselines]"    # timm, for the ConvNeXt UNet baseline models
    pip install "nvsubquadratic[all]"          # all of the above

The accelerated CUDA kernels (``[cuda]``) are a source build requiring ``nvcc``
and are kept out of core, so ``pip install nvsubquadratic`` also succeeds in
environments without the CUDA toolkit. The operators default to the portable
``torch.fft`` backend; ``fft_backend="subq_ops"`` without ``[cuda]`` raises a
clear ``ImportError``.

For development (editable install from source):

.. code-block:: bash

    pip install -e ".[all]"

Requirements
------------

- Python 3.10 or higher
- For GPU execution: a CUDA-compatible NVIDIA GPU and CUDA Toolkit 12.0+
- For the accelerated kernels (``[cuda]``): ``nvcc`` to build ``subquadratic-ops-torch-cu12``

Where to go next
----------------

- :doc:`How HyenaND Works <how_hyenand_works>`: the conceptual on-ramp.
  It builds the operator up from attention (global receptive field +
  data-dependence) and shows how it gets both for ``O(N log N)`` via
  implicit kernels, the FFT, and gating.
- :doc:`Getting Started <getting_started>`: install, requirements, and a
  minimal "Hello, Hyena" forward pass.
- :doc:`Architecture <architecture>`: the three-layer nvSubquadratic /
  subquadratic-ops / megatron-core story and the BHL/BLH naming
  conventions.
- :doc:`Repository Overview <repository_overview>`: bottom-up tour of
  what's inside ``nvsubquadratic/`` (ops / modules / networks / parallel /
  utils).
- :doc:`Lazy-Config System <lazy_config>`: how every run is described by
  one config file, with deferred instantiation, ``${...}`` interpolation, and
  the base-config + ablation workflow.
- :doc:`Benchmarks <benchmarks>`: FLOP scaling, kernel speedups, and a
  worked ViT-5-Small ImageNet training optimization case study.
- :doc:`Reports <reports>`: long-form technical reports backed by
  reproducible scripts and figures.
- :doc:`Glossary <glossary>`: quick definitions for SIREN, FiLM, implicit
  filter, Toeplitz, register tokens, BHL/BLH.
- :doc:`API Reference <api_reference/index>`: auto-generated reference for
  the curated public surface organised by package (ops, modules, networks,
  parallel, core, experiments), opening with the FFT-convolution **ops
  primer** (math motivation + function decision tree).

Contributor docs
----------------

- `CONVENTIONS.md <https://github.com/NVIDIA-BioNeMo/nvSubquadratic/blob/main/CONVENTIONS.md>`_:
  Google-style docstring guide and PR checklist (lives at the repo root).

Related projects
----------------

``nvsubquadratic`` is the high-level PyTorch interface; the underlying
CUDA kernels live in a separate library:

- `subquadratic-ops <https://nvidia-bionemo.github.io/subquadraticOps-docs/>`_:
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
   :hidden:

   How HyenaND Works <how_hyenand_works>
   Getting Started <getting_started>
   Architecture <architecture>
   Repository Overview <repository_overview>
   Lazy-Config System <lazy_config>
   Examples <examples/index>
   Benchmarks <benchmarks>
   Reports <reports>
   Glossary <glossary>
   API Reference <api_reference/index>
