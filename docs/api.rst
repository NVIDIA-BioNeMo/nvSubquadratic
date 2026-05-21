.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. module:: nvsubquadratic
.. currentmodule:: nvsubquadratic

API Reference
=============

The reference is organised bottom-up: low-level FFT convolution primitives
first, then the mixer modules that compose them. See [`docs/ops/README.md`](ops/README.md)
for the math motivation behind the FFT-based ops, and
``docs-tracker.md`` at the repo root for the documentation coverage plan.

Ops — FFT convolutions (reference fp32)
---------------------------------------

Reference implementations in pure PyTorch FFT. Use these for correctness
and as the spec the CUDA kernels must match.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.fftconv.fftconv1d_fp32_blh
   ~ops.fftconv.fftconv2d_fp32_blh
   ~ops.fftconv.fftconv3d_fp32_blh
   ~ops.fftconv.causal_fftconv1d_fp32_blh
   ~ops.fftconv.fftconv1d_fp32_bhl
   ~ops.fftconv.fftconv2d_fp32_bhl
   ~ops.fftconv.fftconv3d_fp32_bhl
   ~ops.fftconv.causal_fftconv1d_fp32_bhl

Ops — FFT convolutions (CUDA-accelerated)
-----------------------------------------

Drop-in wrappers around the ``subquadratic_ops_torch`` fused CUDA kernels.
2D non-causal and 1D causal long-conv variants share the same API as the
fp32 reference ops above.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.fftconv_custom.fftconv2d_blh
   ~ops.fftconv_custom.fftconv2d_bhl
   ~ops.fftconv_custom.fftconv2d_bhl_w_reshape
   ~ops.fftconv_custom.causal_fftconv1d_blh
   ~ops.fftconv_custom.causal_fftconv1d_bhl
   ~ops.fftconv_custom.causal_fftconv1d_bhl_w_reshape

Ops — Direct 1D causal convolutions (CUDA-accelerated)
------------------------------------------------------

Non-FFT CUDA kernels for short and fused 1D causal convolutions. Useful for
small kernel sizes (where FFT overhead dominates) and as building blocks
for fused Hyena variants.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.causal_conv1d_custom.causal_conv1d
   ~ops.causal_conv1d_custom.b2b_causal_conv1d

Ops — Circular FFT convolutions
-------------------------------

Periodic-boundary FFT convolutions for global mixing without zero padding.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.circular_fftconv.circular_fftconv1d_fp32_bhl
   ~ops.circular_fftconv.circular_fftconv2d_fp32_bhl
   ~ops.circular_fftconv.circular_fftconv3d_fp32_bhl

Ops — Multi-head FFT convolutions
---------------------------------

Multi-head variants used by Hyena-style mixers, including low-rank
factorizations.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.fftconv_multihead.fftconv2d_multihead_bhl
   ~ops.fftconv_multihead.fftconv2d_multihead_lowrank_bhl
   ~ops.fftconv_multihead.fftconv2d_multihead_circular_bhl
   ~ops.fftconv_multihead.fftconv2d_multihead_lowrank_circular_bhl

Ops — Chunking utilities
------------------------

Helpers to bound the FFT working-set memory by processing along the
sequence axis in chunks.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.fftconv_chunked.enable_chunking
   ~ops.fftconv_chunked.chunking_enabled
   ~ops.fftconv_chunked.set_default_chunk_size
   ~ops.fftconv_chunked.get_default_chunk_size

Modules — Mixers
----------------

High-level PyTorch ``nn.Module`` sequence/spatial mixers.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.hyena_nd.Hyena
   ~modules.mamba_nd.Mamba
   ~modules.attention.Attention

Modules — Convolutions
----------------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.causal_conv1d.CausalConv1D
   ~modules.subq_ops_causal_conv1d.SubqOpsCausalConv1d
