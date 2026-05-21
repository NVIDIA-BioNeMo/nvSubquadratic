# FP16 Circular FFT Convolution: Derivation

This document derives the numerically stable FP16 circular FFT convolution
used in `circular_fftconv_fp16.py`.  The technique applies to 1D, 2D, and
3D and is a drop-in replacement for the FP32 variants.

## 1. Problem Statement

We want to compute the circular (periodic) convolution of a signal
**x** of length $N$ with a kernel **k** of length $K \le N$, using
half-precision (FP16) FFTs for speed and memory savings.

The standard approach is:

$$
y = \text{IFFT}\!\bigl( \text{FFT}(x) \odot \text{FFT}(k_{\text{pad}}) \bigr)
$$

where $k_{\text{pad}}$ is **k** zero-padded to length $N$.

### Why naive FP16 fails

cuFFT supports FP16 transforms for power-of-2 sizes, but `float16` can
only represent values up to 65504.  Two overflow paths exist:

1. **DC-bin overflow.**  The DC bin of an un-normalized FFT is
   $\hat{x}[0] = \sum_n x[n]$.  For a signal of length $N$ with mean
   $\mu_x$, this is $\mu_x N$.  Even for moderate means ($\mu_x = 5$,
   $N = 16384$), $\mu_x N = 81920 > 65504$.  The frequency-domain
   product at DC is $\mu_x \mu_k N^2$, which overflows even more
   easily.

1. **Internal accumulation overflow.**  cuFFT computes butterfly sums
   in the working precision.  In FP16, partial sums during the FFT
   reach $O(\mu N)$ before any normalization, causing intermediate
   `Inf` or `NaN` even if the final result would be representable.

### Using `norm="ortho"`

PyTorch's `norm="ortho"` divides the forward FFT by $\sqrt{N}$ and the
inverse by $\sqrt{N}$, so the round-trip gives
$\text{ortho-IFFT}(\text{ortho-FFT}(x) \odot \text{ortho-FFT}(k)) = y / \sqrt{N}$.
We multiply by $\sqrt{N}$ after the inverse to recover the correct
scale:

$$
y = \sqrt{N} \cdot \text{IFFT}_{\text{ortho}}\!\bigl(
\text{FFT}_{\text{ortho}}(x) \odot \text{FFT}_{\text{ortho}}(k_{\text{pad}})
\bigr)
$$

This reduces the DC bin to $\mu_x \sqrt{N}$, and the DC product to
$\mu_x \mu_k N$, which still overflows for large $N$.  More importantly,
it does **not** fix internal accumulation overflow (#2 above), because
cuFFT performs the $\sqrt{N}$ scaling *after* the butterfly, not during.

## 2. Solution: Dual Mean-Centering

### Core idea

Remove the DC component from both signals before the FFT:

$$
x_c = x - \mu_x, \qquad k_c = k - \mu_k
$$

Both centered signals have zero mean, so:

- Their DC bins are exactly 0 (fixing overflow path #1).
- Internal FFT sums are $O(\sigma)$ instead of $O(\mu N)$ (fixing #2).

We then recover the exact convolution result analytically.

### 1D Derivation

Let $L = N$ be the circular convolution length.  The kernel has $K$
nonzero elements ($K \le L$, typically $K = L$ or $K = L-1$) and is
zero-padded to length $L$.

Define:

- $\mu_x = \frac{1}{L}\sum_{n} x[n]$, and $x_c = x - \mu_x$
- $\mu_k = \frac{1}{K}\sum_{m} k[m]$, and $k_c = k - \mu_k$
- $\delta[m]$ = the "mean indicator": $\mu_k$ for $0 \le m < K$, zero
  otherwise (i.e., $k_{\text{pad}} - k_{c,\text{pad}} = \delta$)

Then the zero-padded kernel decomposes as:

$$
k_{\text{pad}} = k_{c,\text{pad}} + \delta
$$

Expanding the circular convolution:

$$
y[n] =
\underbrace{(x_c * k_{c,\text{pad}})[n]}_{T_1}
+ \underbrace{(x_c * \delta)[n]}_{T_2}
+ \underbrace{\mu_x \sum_m k_{c,\text{pad}}[m]}_{T_3}
+ \underbrace{\mu_x \cdot \mu_k \cdot K}_{T_4}
$$

where $*$ denotes circular convolution.

**Term T1** — the "safe" convolution of two zero-mean signals.  Both DC
bins are 0, internal magnitudes are $O(\sigma)$.  Computed via FP16 FFT.

**Term T3** = 0, because $\sum_m k_c[m] = 0$ by construction.

**Term T4** is a scalar constant, computed in FP32.

**Term T2** — the centering correction — requires care.

#### T2: Centering correction (1D)

$$
T_2[n] = \sum_m x_c[(n-m) \bmod L] \cdot \delta[m]
= \mu_k \sum_{m=0}^{K-1} x_c[(n-m) \bmod L]
$$

**Case $K = L$** (kernel covers the full circle):

$$
T_2[n] = \mu_k \sum_{m=0}^{L-1} x_c[(n-m) \bmod L] = \mu_k \cdot 0 = 0
$$

because $\sum x_c = 0$.  **No correction needed.**

**Case $K = L-1$** (one zero-padded position, at index $L-1$):

The sum covers all positions except $m = L-1$:

$$
T_2[n] = \mu_k \!\!\sum_{m \ne L-1}\!\! x_c[(n-m) \bmod L]
       = -\mu_k \cdot x_c[(n+1) \bmod L]
$$

This is just $-\mu_k$ times a circular shift of $x_c$ by $-1$.

#### Phase-ramp absorption

In frequency domain, a circular shift by $s$ corresponds to
multiplication by the phase ramp $\phi_s[f] = e^{-2\pi i f s / L}$.

When the kernel is centered with shift $s = -\lfloor(K-1)/2\rfloor$,
the zero-padded position moves, and the correction becomes:

$$
T_2[n] = -\mu_k \cdot x_c[(n - s + 1) \bmod L]
$$

In frequency domain, the effective kernel spectrum (including T1 + T2)
is:

$$
\hat{k}_{\text{eff}}[f] = \phi_s[f] \cdot
\biggl(\hat{k}_c[f] - \tfrac{\mu_k}{\sqrt{L}} \cdot \phi_{-1}[f]\biggr)
$$

where $\hat{k}_c$ is the ortho-normalized FFT of $k_c$ (zero-padded),
and $\phi_{-1}[f] = e^{2\pi i f / L}$ is the DFT of the delta at
position $L-1$.

The $1/\sqrt{L}$ factor keeps intermediate values small (matching the
ortho scaling); the compensating $\sqrt{L}$ is applied after the inverse
FFT in FP32.

### Final 1D formula

$$
y = \sqrt{L}\,\text{IFFT}_{\text{ortho}}\!\bigl(
\hat{x}_c \odot \hat{k}_{\text{eff}}
\bigr) + \mu_x \mu_k K
$$

where $\hat{x}_c = \text{FFT}_{\text{ortho}}(x_c)$ is computed in FP16,
$\hat{k}_{\text{eff}}$ is assembled in FP32 (small tensor, no batch dim)
and cast to `complex32` before the element-wise multiply, and the
$\sqrt{L}$ rescaling and DC correction are done in FP32.

## 3. Extension to nD

For $d$-dimensional signals of shape $N_1 \times \cdots \times N_d$
with kernel shape $K_1 \times \cdots \times K_d$, the decomposition
generalizes.  The T1 and T4 terms carry over directly:

$$
T_1 = \text{circ\_conv}(x_c, k_{c,\text{pad}}), \qquad
T_4 = \mu_x \mu_k \prod_i K_i
$$

T3 is again zero.  **T2 becomes an inclusion-exclusion sum over
corrected axes** — those axes where $K_i < N_i$ (i.e., there is at
least one zero-padded position).

### Geometric correction factor

Define $\mathcal{C} = {i : K_i < N_i}$ as the set of corrected axes.
For each nonempty subset $S \subseteq \mathcal{C}$, define the phase
factor $p_i[f_i] = e^{2\pi i f_i / N_i}$ (the DFT of a delta at
position $N_i - 1$ along axis $i$).

The geometric correction in frequency domain is:

$$
\text{geo}[\mathbf{f}] = \sum_{\emptyset \ne S \subseteq \mathcal{C}}
(-1)^{|S|} \Bigl(\prod_{i \in S} p_i[f_i]\Bigr)
\Bigl(\prod_{j \notin S} N_j\Bigr)
$$

The corrected effective kernel spectrum is:

$$
\hat{k}_{\text{eff}}[\mathbf{f}] = \hat{k}_c[\mathbf{f}]
+ \tfrac{\mu_k}{\sqrt{N}} \cdot \text{geo}[\mathbf{f}]
$$

followed by the phase-ramp shift for kernel centering.

### 2D example

For a 2D signal $X \times Y$ with kernel $K_x \times K_y$:

**Both axes corrected** ($K_x < X$ and $K_y < Y$):

$$
\text{geo}[f_1, f_2] = -Y \cdot p_x[f_1]\,\delta_{f_2=0}
- X \cdot \delta_{f_1=0}\,p_y[f_2]
+ p_x[f_1]\,p_y[f_2]
$$

**One axis corrected** (e.g., $K_x < X$, $K_y = Y$):

$$
\text{geo}[f_1, f_2] = -Y \cdot p_x[f_1]\,\delta_{f_2=0}
$$

### Caching

The geometric factor `geo` depends only on
$(K_1, \ldots, K_d, N_1, \ldots, N_d, \text{device})$ and is
**constant** during training.  It is computed once and cached with an
LRU policy.  Only the scalar $\mu_k / \sqrt{N}$ changes per forward
call.

## 4. Implementation Details

### Precision strategy

| Operation                              | Precision        | Rationale                                                |
| -------------------------------------- | ---------------- | -------------------------------------------------------- |
| Mean computation ($\mu_x$, $\mu_k$)    | FP16             | Cheap, small values                                      |
| Forward FFT                            | FP16 (complex32) | Centered signals, cuFFT half-precision                   |
| $\hat{k}_{\text{eff}}$ assembly        | FP32 (complex64) | Small tensor (no batch), needs precision for phase ramps |
| $\hat{x}_c \odot \hat{k}_{\text{eff}}$ | complex32        | Large tensor, cast $\hat{k}_{\text{eff}}$ down           |
| Inverse FFT                            | FP16 (complex32) | Same cuFFT path as forward                               |
| $\sqrt{N}$ rescaling + DC correction   | FP32             | Avoid overflow in final scaling                          |
| Output cast                            | Original dtype   | Match caller expectations                                |

### cuFFT power-of-2 constraint

cuFFT only supports FP16 transforms for power-of-2 sizes.  Since we use
same-size circular FFTs (no padding), the **input spatial dimensions
must be powers of 2**.  This is asserted at runtime.

### Phase ramps

Alignment shifts are implemented as frequency-domain phase ramps
(precomputed and cached in FP32, cast to complex32 before multiply).
This avoids a spatial `torch.roll` which would prevent fusion under
`torch.compile`.

## 5. Correctness Guarantees

- **Mathematically exact**: the dual-centering decomposition introduces
  no approximation.  The only source of error is finite-precision
  arithmetic in the FP16 FFT (vs FP32).
- **Validated**: automated tests in `tests/test_circular_fftconv_fp16.py`
  verify 1D, 2D, and 3D implementations against FP32 references across
  multiple shapes and kernel sizes.
- **Inference validated**: on a trained 177M-parameter Euler Hyena model,
  the FP16 path produces identical validation loss (to 4 significant
  figures) as the FP32 reference.
