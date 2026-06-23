# Glossary

Quick definitions for the terms that show up across these docs and the
code.  For the narrative that ties them together, see
{doc}`how_hyenand_works`.

```{glossary}
HyenaND
  The library's flagship operator: a `Short Conv → Gate → Long Conv → Gate`
  sandwich that achieves a global, data-dependent receptive field in
  $O(N \log N)$ on native 1D / 2D / 3D grids.  See
  {doc}`how_hyenand_works`.

Subquadratic
  Scaling better than $O(N^2)$ in the number of tokens $N$.  HyenaND is
  $O(N \log N)$; attention is $O(N^2)$.

Receptive field
  The span of input a single output position can depend on.  A *global*
  receptive field means any output can see the entire input — attention's
  defining property, which HyenaND reproduces via long convolutions.

Data-dependence
  Whether the operator's mixing weights are computed from the input
  (data-dependent) or fixed after training.  Attention is data-dependent
  through the attention matrix $A(x)$; HyenaND is data-dependent through
  {term}`gating`.

Gating
  Element-wise multiplication of a signal by a data-derived mask
  (e.g. $q \odot \mathrm{SiLU}(k)$).  HyenaND interleaves gates with its
  long convolution to make the effective operator depend on the input
  without ever materialising an $N \times N$ matrix.

Implicit filter
  A convolution kernel produced by evaluating a small network
  ({term}`SIREN`) on grid coordinates, rather than stored as one learnable
  weight per tap.  Compact, and — because it is a continuous function of
  position — samplable on a grid of any size or aspect ratio without
  retraining.  Contrast with an *explicit* filter, whose taps are
  learned parameters (as in a classical CNN).

SIREN
  A sinusoidal-activation MLP ($f_\theta$) used to parametrise implicit
  filters.  Its frequency is controlled by an $\omega_0$ hyperparameter
  that scales with grid resolution and dimensionality — see
  {doc}`reports` for the dimensional-scaling rule.  Implemented in
  {mod}`nvsubquadratic.modules.kernels_nd`.

FiLM
  Feature-wise Linear Modulation.  Conditions the synthesised kernel on a
  control variable $z(\mathbf{x})$ pooled from the input's
  {term}`register tokens`, making the kernel input-dependent.  Implemented
  in {mod}`nvsubquadratic.modules.film`.

Register tokens
  Auxiliary tokens carried alongside the data tokens whose pooled state
  feeds the {term}`FiLM` conditioning of the Hyena kernel.

FFT convolution (FFTConv)
  Computing a convolution as an element-wise product in the frequency
  domain, $y = \mathcal{F}^{-1}(\mathcal{F}(x) \odot \mathcal{F}(K))$.
  Each FFT is $O(N \log N)$ and the total cost is independent of kernel
  size — the reason a global kernel is affordable.  Implemented in
  {doc}`ops/README`.

Convolution theorem
  The identity that convolution in the spatial domain equals
  element-wise multiplication in the frequency domain.  The mathematical
  basis for {term}`FFT convolution (FFTConv)`.

Toeplitz matrix
  The matrix form of a 1D convolution: each row is a shifted copy of the
  filter.  Convolving a signal with a filter is the same as multiplying
  by the corresponding Toeplitz matrix; a *causal* convolution is a
  lower-triangular one.

Linear vs circular convolution
  **Linear** zero-pads the input so the kernel never wraps around
  (matches `torch.nn.ConvNd`); **circular** treats the input as periodic
  so the kernel wraps at the boundary (useful for PDEs and periodic
  signals).  See {doc}`ops/README`.

BHL / BLH
  Memory layout.  **BHL** is channels-first (`[B, H, *spatial]`, matches
  `torch.nn.ConvNd`); **BLH** is channels-last (`[B, *spatial, H]`, common
  in transformer code).  The FFT is faster on contiguous spatial axes, so
  BHL is the fast path; `_w_reshape` wrappers accept BLH and convert.

Causal
  An operator where output position $n$ depends only on inputs at
  positions $\le n$ — no leakage from the future.  Required for
  autoregressive 1D sequence modelling.

Mixer
  An operator with the shared $(q, k, v)$ signature that
  {class}`nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`
  dispatches over — Hyena, attention, CKConv, or Mamba — so a network can
  swap one for another via a one-line config change.

CKConv
  Continuous-Kernel Convolution: a convolution whose kernel is an
  {term}`implicit filter` $k_\theta(p)$.  A close relative of Hyena's
  long-conv path; see {mod}`nvsubquadratic.modules.ckconv_nd`.

LazyConfig
  The library's deferred-instantiation system: a config object that
  records *what* to build and *how* without building it yet, so example
  recipes describe a whole experiment as a tree of configs.  See
  {class}`nvsubquadratic.lazy_config.LazyConfig`.
```

</content>
