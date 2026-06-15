.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. currentmodule:: nvsubquadratic

Ops
===

Low-level convolution primitives.  Pure-PyTorch reference implementations
double as the spec the CUDA kernels must match; the
``subquadratic_ops_torch`` wrappers are the production path on GPU.

FFT convolutions (reference fp32)
---------------------------------

Use these for correctness and as the spec for the CUDA kernels below.

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

FFT convolutions (CUDA-accelerated)
-----------------------------------

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

Direct 1D causal convolutions (CUDA-accelerated)
------------------------------------------------

Non-FFT CUDA kernels for short and fused 1D causal convolutions.  Useful
for small kernel sizes (where FFT overhead dominates) and as building
blocks for fused Hyena variants.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.causal_conv1d_custom.causal_conv1d
   ~ops.causal_conv1d_custom.b2b_causal_conv1d

Circular FFT convolutions
-------------------------

Periodic-boundary FFT convolutions for global mixing without zero padding.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.circular_fftconv.circular_fftconv1d_fp32_bhl
   ~ops.circular_fftconv.circular_fftconv2d_fp32_bhl
   ~ops.circular_fftconv.circular_fftconv3d_fp32_bhl

Chunking utilities
------------------

Helpers to bound the FFT working-set memory by processing along the
sequence axis in chunks.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.fftconv_chunked.enable_chunking
   ~ops.fftconv_chunked.chunking_enabled
   ~ops.fftconv_chunked.set_default_chunk_size
   ~ops.fftconv_chunked.get_default_chunk_size

Mixed boundary-condition FFT convolutions
-----------------------------------------

FFT convolutions with per-axis boundary conditions — periodic on some
spatial axes, zero-padded on others.  See
:doc:`../ops/mixed_boundary_conditions` for the per-axis algorithm and the
``fft_padding`` API.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.mixed_fftconv.mixed_fftconv1d_fp32_bhl
   ~ops.mixed_fftconv.mixed_fftconv2d_fp32_bhl
   ~ops.mixed_fftconv.mixed_fftconv3d_fp32_bhl
   ~ops.mixed_fftconv.mixed_fftconv1d_fp32_bhl_w_reshape
   ~ops.mixed_fftconv.mixed_fftconv2d_fp32_bhl_w_reshape
   ~ops.mixed_fftconv.mixed_fftconv3d_fp32_bhl_w_reshape
   ~ops.mixed_fftconv.mixed_fftconv1d_fp32_bhl_chunked
   ~ops.mixed_fftconv.mixed_fftconv2d_fp32_bhl_chunked
   ~ops.mixed_fftconv.mixed_fftconv3d_fp32_bhl_chunked
