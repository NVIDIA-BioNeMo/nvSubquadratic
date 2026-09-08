# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused vs subq_ops vs torch_fft, 2D — the comparison behind the published claim.

The README/CHANGELOG/docs claim 3.6-3.9x over torch_fft and 1.2-2.4x over
subq_ops, measured on H200 at B=8, hidden=768, forward+backward, bf16. Nothing
in benchmarks/ reproduces that: bench_fftconv2d.py covers torch_fft and
subq_ops only. This script closes that gap.

Run on an SM90+ node (spatial 64 needs the 128 FFT tile):

    python benchmarks/ops/bench_fused_fftconv2d.py                    # the claim's exact config
    python benchmarks/ops/bench_fused_fftconv2d.py --hidden 256 --batch 4
    python benchmarks/ops/bench_fused_fftconv2d.py --dtype float16

Requires SM90+ for the spatial-64 row (128 FFT tile); 16 and 32 run anywhere.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

# NOTE: fftconv2d_fp32_bhl exists in BOTH ops/fftconv.py and ops/fftconv_chunked.py.
# The reference path is the fftconv.py one; importing the wrong module silently
# benchmarks the chunked implementation instead.
from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl

# fftconv2d_bhl is the repo's subq_ops wrapper, and the only correct way to time
# subq_ops here. Do NOT call subquadratic_ops_torch.fft_conv2d directly: that raw
# op is fp32-only and raises NotImplementedError on bf16/fp16, which is exactly
# the dtype this comparison is about. The wrapper is a documented drop-in for
# fftconv2d_fp32_bhl — it handles the fp32 round-trip and takes the same
# [1|B, H, Kx, Ky] kernel shape as the other two paths.
from nvsubquadratic.ops.fftconv_custom import fftconv2d_bhl as subq_fftconv2d_bhl
from nvsubquadratic.ops.fftconv_custom import fused_fftconv2d_bhl, resolve_fused_fft_size


try:
    import subquadratic_ops_torch  # noqa: F401  (probe only; the wrapper imports lazily)

    HAVE_SUBQ = True
except Exception as exc:  # pragma: no cover - environment dependent
    HAVE_SUBQ = False
    _SUBQ_ERR = exc

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _time_fwd_bwd(fn, inputs, warmup: int, iters: int) -> float:
    """Median ms per forward+backward, CUDA-synchronised, grads zeroed each iter."""
    for _ in range(warmup):
        fn(*inputs).sum().backward()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        for t in inputs:
            if torch.is_tensor(t) and t.grad is not None:
                t.grad = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(*inputs).sum().backward()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=8, help="batch size (claim used 8)")
    ap.add_argument("--hidden", type=int, default=768, help="hidden dim (claim used 768)")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    ap.add_argument("--spatials", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    dtype = DTYPES[args.dtype]
    cap = torch.cuda.get_device_capability()
    print(f"GPU            : {torch.cuda.get_device_name(0)}  (sm_{cap[0]}{cap[1]})")
    print(f"torch          : {torch.__version__}")
    print(f"subq_ops       : {'available' if HAVE_SUBQ else f'NOT IMPORTABLE ({_SUBQ_ERR!r})'}")
    print(f"config         : B={args.batch} hidden={args.hidden} {args.dtype} fwd+bwd, median of {args.iters} iters\n")
    if cap < (9, 0):
        print("!! sm_90+ required for spatial 64 (128 FFT tile); those rows will fail.\n")

    hdr = f"{'spatial':>7} {'K':>5} {'tile':>5} {'torch_fft':>11} {'subq_ops':>11} {'fused':>11} {'vs torch':>9} {'vs subq':>9}"
    print(hdr)
    print("-" * len(hdr))

    for n in args.spatials:
        k = 2 * n - 1  # CKConvND double-grid kernel
        try:
            tile = resolve_fused_fft_size(n, n, k, k)
        except ValueError as exc:
            print(f"{n:>7} {k:>5}  resolve_fused_fft_size rejected: {exc}")
            continue

        x = torch.randn(args.batch, args.hidden, n, n, device="cuda", dtype=dtype, requires_grad=True)
        # Scaled down so the convolution output stays in range for fp16.
        k_bhl = (torch.randn(1, args.hidden, k, k, device="cuda", dtype=dtype) * 0.05).requires_grad_()

        try:
            t_ref = _time_fwd_bwd(lambda a, b: fftconv2d_fp32_bhl(a, b, None), (x, k_bhl), args.warmup, args.iters)
        except Exception as exc:
            print(f"{n:>7} {k:>5} {tile:>5}  torch_fft FAILED: {exc!r}")
            continue

        try:
            t_fused = _time_fwd_bwd(lambda a, b: fused_fftconv2d_bhl(a, b, None), (x, k_bhl), args.warmup, args.iters)
        except Exception as exc:
            print(f"{n:>7} {k:>5} {tile:>5} {t_ref:>10.3f}ms  fused FAILED: {exc!r}")
            continue

        t_subq = float("nan")
        if HAVE_SUBQ:
            try:
                t_subq = _time_fwd_bwd(
                    lambda a, b: subq_fftconv2d_bhl(a, b, None), (x, k_bhl), args.warmup, args.iters
                )
            except Exception as exc:
                print(f"    (subq_ops failed at spatial={n}: {exc!r})")

        vs_torch = t_ref / t_fused
        vs_subq = t_subq / t_fused if t_subq == t_subq else float("nan")
        subq_str = f"{t_subq:>10.3f}ms" if t_subq == t_subq else f"{'n/a':>12}"
        vs_subq_str = f"{vs_subq:>8.2f}x" if vs_subq == vs_subq else f"{'n/a':>9}"
        print(
            f"{n:>7} {k:>5} {tile:>5} {t_ref:>10.3f}ms {subq_str} {t_fused:>10.3f}ms {vs_torch:>8.2f}x {vs_subq_str}"
        )

    print("\nPublished claim: 3.6-3.9x over torch_fft, 1.2-2.4x over subq_ops")
    print("(H200, B=8, hidden=768, fwd+bwd, bf16). If these numbers disagree, the")
    print("four sites to correct are README.md:70, CHANGELOG.md:17,")
    print("docs/ops/README.md:127 and nvsubquadratic/modules/ckconv_nd.py:621.")


if __name__ == "__main__":
    main()
