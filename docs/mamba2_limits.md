# Mamba-2 sequence-length limits

Why the Mamba-2 baseline stops early in our forward-time sweeps, measured rather
than assumed. Two **separate** implementation limits, neither of them memory, and
neither inherent to the SSD algorithm.

This page exists because [`benchmarks.md`](benchmarks.md) previously attributed the
stop to the kernel running out of memory. It does not run out of memory. The
*reach* reported there is reproducible; the stated cause was wrong.

Measured on **GB200 (184 GiB)** with **mamba-ssm 2.3.2.post1**, causal-conv1d
1.6.2.post1, triton 3.7.1, torch 2.12.1+cu130 — 2.3.2.post1 being the latest
release on PyPI at the time of writing.

---

## Summary

| Limit | Where | Bound | Scales with |
|---|---|---|---|
| 32-bit index overflow | `ssd_chunk_state.py` → `_chunk_state_fwd` (Triton) | ~2<sup>31</sup> **elements** | model width — longer sequences allowed at narrower widths |
| CUDA grid-dimension cap | `causal_conv1d` channels-last path | **4,194,240 tokens** = 65,535 × 64 | nothing — a fixed token count |

**No configuration we tested ran out of memory.** Peak allocation never exceeded
87 GiB of 184 GiB, and at the 1M failure it was 22 GiB (12%).

---

## Limit 1 — 32-bit index overflow in the SSD scan

At realistic widths this is what stops Mamba-2 first. The fault site, with
`CUDA_LAUNCH_BLOCKING=1`:

```
mamba2.forward → mamba_split_conv1d_scan_combined
  → _mamba_chunk_scan_combined_fwd  (ssd_combined.py:375)
    → _chunk_state_fwd              (ssd_chunk_state.py:830)
      → triton autotuner → do_bench → kernel_call
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

An *illegal memory access* is a memory-safety fault — a thread dereferenced an
out-of-bounds address. It is not an allocator failure, and it is not the same
class of event as an OOM.

### Evidence: the threshold scales inversely with width

Bisected at `headdim=64, expand=2, d_state=128, ngroups=8`:

| hidden | widest tensor width | last OK L | first FAIL L | elements at last OK |
|---:|---:|---:|---:|---:|
| 768 | 3,584 | 598,016 | 606,208 | 2.143e9 = **99.8% of 2<sup>31</sup>** |
| 1536 | 8,240 | 253,952 | 262,144 | 2.093e9 = **97.5% of 2<sup>31</sup>** |

Halving the reach when width doubles is the signature of an **element-count**
limit. A fixed sequence-length cap would not move with width; a memory limit
would show as an OOM at a much larger footprint. Both boundaries land just under
2<sup>31</sup> elements of the widest tensor in play.

(The *binding* tensor differs between the two configs, so the claim here is
"~2<sup>31</sup> elements", not a pinpointed expression.)

### Confirmed by patching

Casting `tl.program_id(...)` to `tl.int64` before it feeds pointer arithmetic
fixes the 1M case:

| Configuration | L = 1,048,576 |
|---|---|
| Stock 2.3.2.post1 | illegal memory access |
| + int64 in `ssd_chunk_scan.py` + `layernorm_gated.py` | illegal memory access (**no change**) |
| + int64 also in `ssd_chunk_state.py` | **passes** |

Necessary but **not sufficient**: with all three files patched, L=2,097,152 still
fails, with a Triton `invalid argument` — a launch-configuration rejection (grid
geometry), a different limit behind this one.

### Upstream status

Reported as [state-spaces/mamba#686](https://github.com/state-spaces/mamba/issues/686)
("Long Sequence Length Inference Mamba2: CUDA error: an illegal memory access was
encountered"), still open. As of this writing there is no fix:

* 2.3.2.post1 is the latest PyPI release.
* `ssd_chunk_state.py` and `ssd_chunk_scan.py` are byte-identical between the
  `v2.3.2` tag and `main`.
* The `mamba_ssm` tree contains no int64 index widening at all — the only two
  `int64` occurrences are a dropout-seed dtype in `layer_norm.py` and a comment in
  `mamba3_mimo.py`.

The `wdykas@bfec072` ("more int64") commit referenced in that thread **does not
fix this crash**: it widens `program_id` in `ssd_chunk_scan.py` and
`layernorm_gated.py`, not in `ssd_chunk_state.py` where the fault occurs. We
verified this by applying a strict superset of its changes to both files (40 casts
versus its 4) — L=1M still failed.

---

## Limit 2 — CUDA grid-dimension cap in `causal_conv1d`

At small widths the SSD scan survives much further, and a different limit binds:
the short conv. It appears as `CUDA error: invalid argument`
(`cudaErrorInvalidValue`) — the driver *rejecting the launch*, so nothing executes.

The boundary is exact and independent of channel count:

| L | blocks (L / 64) | result |
|---:|---:|---|
| 4,194,176 | 65,534 | ok |
| **4,194,240** | **65,535** | **ok** |
| 4,194,304 | 65,536 | fail |

`65,535` is the CUDA cap on a grid dimension, and the channels-last kernel tiles
the sequence at 64 elements per block, giving **L ≤ 65,535 × 64 = 4,194,240**.

It only affects the **channels-last** path — which is the layout Mamba-2 passes,
via `rearrange(ensure_stride(xBC), "b s d -> b d s")`. A contiguous `[b, d, s]`
tensor runs to at least 32M tokens at every width we tried.

---

## Why this is not an architectural limit

The SSD chunk-scan is O(L) with fixed state per chunk; nothing in the algorithm
requires 32-bit offsets or caps a grid axis. Both limits are properties of the
current kernels:

* the overflow is a pointer-arithmetic width choice, liftable with int64 casts
  (demonstrated above);
* the grid cap is an unhandled input size in one layout, and the other layout has
  no such bound.

On the same GPU at the same widths, HyenaND reaches **16.7M tokens**.

So these results say *"the shipping Mamba-2 kernels do not reach these lengths
today"* — a statement about an implementation, with a known and unmerged fix. They
do **not** support the stronger claim that Mamba-2 as an architecture cannot reach
them, and should not be cited that way.

---

## Reproducing

Both limits reproduce in a few lines. Mamba-2's package `__init__` eagerly imports
`mamba3` → `tilelang`, which collides with the `apache-tvm-ffi` that
subquadratic-ops-torch >= 0.2.2 installs, hence the shim.

```python
import sys, torch
sys.modules.setdefault("tilelang", None)   # see note above

# Limit 1 — overflow in the SSD scan (fails ~600K at hidden 768)
from mamba_ssm import Mamba2
m = Mamba2(d_model=768, headdim=64, expand=2, d_state=128, ngroups=8,
           device="cuda", dtype=torch.bfloat16)
x = torch.randn(1, 1_048_576, 768, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    m(x)                                   # illegal memory access, ~22 GiB peak

# Limit 2 — grid cap in causal_conv1d, channels-last only
from causal_conv1d import causal_conv1d_fn
w = torch.randn(272, 4, device="cuda", dtype=torch.bfloat16)
for L in (4_194_240, 4_194_304):           # 65,535 vs 65,536 blocks
    xc = torch.randn(1, L, 272, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    causal_conv1d_fn(x=xc, weight=w, bias=None, activation="silu")
```

The sweeps that surfaced this are `scripts/slurm/submit_forward_time_nd.sh` (the
`mamba` / `mamba_causal` series) and `scripts/slurm/submit_nemotron_1d.sh`.
