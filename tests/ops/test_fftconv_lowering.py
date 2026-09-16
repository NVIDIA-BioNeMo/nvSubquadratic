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

"""Tests for the torch.compile lowering in ``nvsubquadratic.ops.fftconv_lowering``.

A silent graph pass has two distinct failure modes, and this file covers both:

  - **False negative** — it quietly does not fire, and the user sees no speedup
    and no error. Every positive test therefore asserts on
    ``lowering_stats()["rewritten"]``, not just on numerics.
  - **False positive** — it rewrites a subgraph that is not actually the
    reference recipe, silently changing results. The guard tests assert the
    specific skip reason *and* bit-exact agreement with eager, so a rewrite
    that slipped through would show up as a numeric difference.

Usage (requires GPU — run inside SLURM):
    srun --gres=gpu:1 -c 16 --partition low \\
        conda run -n nv-subq python -m pytest tests/ops/test_fftconv_lowering.py -v -o addopts=""
"""

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomGraphPass

from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl
from nvsubquadratic.ops.fftconv_lowering import (
    FusedFFTConv2dLowering,
    fused_fftconv2d_lowering,
    fused_fftconv2d_options,
    lowering_stats,
    reset_lowering_stats,
)
from tests.conftest import L2_TOL_GRAD, assert_l2_close, requires_sm90, requires_subq_ops_fused


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

pytestmark = [requires_subq_ops_fused, requires_cuda]


@pytest.fixture(autouse=True)
def _clean_compile_state():
    """Give every test a fresh dynamo cache, no inductor caching, and zeroed counters.

    Disabling the caches is load-bearing, not hygiene. Inductor's FX graph cache
    is on *disk* and survives across processes; on a cache hit it skips every
    pre-grad pass, so the counters these tests assert on would stay empty and
    the assertions would be measuring the cache rather than the pass.
    """
    from torch._inductor import config as inductor_config

    previous = inductor_config.force_disable_caches
    inductor_config.force_disable_caches = True
    torch._dynamo.reset()
    reset_lowering_stats()
    try:
        yield
    finally:
        torch._dynamo.reset()
        reset_lowering_stats()
        inductor_config.force_disable_caches = previous


def _compile_with_lowering(fn, *args, allow_reduced_precision=True):
    """Compile via the ``options=`` route — the primary way to enable the pass."""
    options = fused_fftconv2d_options(allow_reduced_precision=allow_reduced_precision)
    return torch.compile(fn, fullgraph=True, options=options)(*args)


class _RecordingPass(CustomGraphPass):
    """Wrap the real pass and keep the node targets it left behind.

    The counters prove a rewrite *happened*; this proves the old chain is
    *gone*. Both matter: a pass that inserts the fused node but fails to erase
    the cuFFT chain still reports ``rewritten`` and still passes numerics.
    """

    def __init__(self, inner):
        self.inner = inner
        self.remaining_targets: list = []

    def __call__(self, graph):
        graph = self.inner(graph)
        self.remaining_targets = [n.target for n in graph.nodes if n.op.startswith("call")]
        return graph

    def uuid(self):
        return f"recording-{self.inner.uuid()}"


_FFT_CHAIN_TARGETS = (torch.fft.rfft2, torch.fft.irfft2, "mul_")


def _compile_and_record(fn, *args):
    """Compile with the pass and return ``(output, targets_left_in_graph)``."""
    recorder = _RecordingPass(FusedFFTConv2dLowering())
    out = torch.compile(fn, fullgraph=True, options={"pre_grad_custom_pass": recorder})(*args)
    return out, recorder.remaining_targets


def _assert_chain_erased(targets):
    leftover = [t for t in targets if t in _FFT_CHAIN_TARGETS]
    assert not leftover, f"FFT chain nodes survived the rewrite: {leftover}"


def _bhl_inputs(spatial=16, kernel_size=31, dtype=torch.float32, channels=8, batch=2, kernel_batch=1):
    torch.manual_seed(0)
    x = torch.randn(batch, channels, spatial, spatial, device="cuda", dtype=dtype)
    kernel = (torch.randn(kernel_batch, channels, kernel_size, kernel_size, device="cuda") * 0.05).to(dtype)
    shortcut = torch.randn(channels, device="cuda", dtype=dtype)
    return x, kernel, shortcut


# ---------------------------------------------------------------------------
# The pass fires and preserves semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rewrites_and_matches_eager(dtype):
    x, kernel, shortcut = _bhl_inputs(dtype=dtype)
    expected = fftconv2d_fp32_bhl(x, kernel, shortcut)

    out, remaining = _compile_and_record(fftconv2d_fp32_bhl, x, kernel, shortcut)

    assert lowering_stats().get("rewritten") == 1
    _assert_chain_erased(remaining)
    assert_l2_close(out, expected, dtype)


def test_rewrites_without_shortcut():
    """The shortcut add is optional and sits outside the matched chain."""
    x, kernel, _ = _bhl_inputs()
    expected = fftconv2d_fp32_bhl(x, kernel)

    out = _compile_with_lowering(fftconv2d_fp32_bhl, x, kernel)

    assert lowering_stats().get("rewritten") == 1
    assert_l2_close(out, expected, torch.float32)


@pytest.mark.parametrize("spatial", [8, 16, 32, pytest.param(64, marks=requires_sm90)])
def test_rewrites_across_supported_spatial_sizes(spatial):
    x, kernel, shortcut = _bhl_inputs(spatial=spatial, kernel_size=2 * spatial - 1, channels=4, batch=1)
    expected = fftconv2d_fp32_bhl(x, kernel, shortcut)

    out = _compile_with_lowering(fftconv2d_fp32_bhl, x, kernel, shortcut)

    assert lowering_stats().get("rewritten") == 1
    assert_l2_close(out, expected, torch.float32)


def test_rewrites_film_per_sample_kernel():
    x, kernel, shortcut = _bhl_inputs(kernel_batch=2)
    expected = fftconv2d_fp32_bhl(x, kernel, shortcut)

    out = _compile_with_lowering(fftconv2d_fp32_bhl, x, kernel, shortcut)

    assert lowering_stats().get("rewritten") == 1
    assert_l2_close(out, expected, torch.float32)


def test_rewrites_every_matching_chain():
    """One graph may contain multiple independent FFT-conv chains."""
    x, kernel_a, _ = _bhl_inputs()
    kernel_b = torch.randn_like(kernel_a) * 0.05

    def two_convs(x, kernel_a, kernel_b):
        return fftconv2d_fp32_bhl(x, kernel_a), fftconv2d_fp32_bhl(x, kernel_b)

    expected = two_convs(x, kernel_a, kernel_b)
    out, remaining = _compile_and_record(two_convs, x, kernel_a, kernel_b)

    assert lowering_stats().get("rewritten") == 2
    _assert_chain_erased(remaining)
    for actual, want in zip(out, expected):
        assert_l2_close(actual, want, torch.float32)


def test_rewrite_preserves_unrelated_inplace_call_method():
    """Cleanup must not prune mutations outside the matched FFT-conv chain."""
    x, kernel, _ = _bhl_inputs()

    def conv_and_mutate(x, kernel, state):
        state.add_(1)
        return fftconv2d_fp32_bhl(x, kernel), state

    expected_state = torch.zeros((), device="cuda")
    expected_out, _ = conv_and_mutate(x, kernel, expected_state)
    actual_state = torch.zeros((), device="cuda")
    actual_out, _ = _compile_with_lowering(conv_and_mutate, x, kernel, actual_state)

    assert lowering_stats().get("rewritten") == 1
    assert_l2_close(actual_out, expected_out, torch.float32)
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)


def _reference_recipe(x, kernel, *, multiply):
    """The reference padding/crop recipe with a pluggable spectrum multiply."""
    dim_x, dim_y = x.shape[-2:]
    k_x, k_y = kernel.shape[-2:]
    s = (min(dim_x + (k_x + 1) // 2, 2 * dim_x), min(dim_y + (k_y + 1) // 2, 2 * dim_y))
    spec_x = torch.fft.rfft2(x.float(), s=s, dim=(2, 3))
    spec_k = torch.fft.rfft2(kernel.float(), s=s, dim=(2, 3))
    product = multiply(spec_x, spec_k)
    out = torch.fft.irfft2(product, s=s, dim=(2, 3))
    return out[..., k_x // 2 : k_x // 2 + dim_x, k_y // 2 : k_y // 2 + dim_y].to(x.dtype)


def test_rewrites_out_of_place_multiply():
    """``a * b`` is the second spelling the matcher accepts; it must fully erase too."""
    x, kernel, _ = _bhl_inputs()

    def conv(x, kernel):
        return _reference_recipe(x, kernel, multiply=lambda a, b: a * b)

    expected = conv(x, kernel)
    out, remaining = _compile_and_record(conv, x, kernel)

    assert lowering_stats().get("rewritten") == 1
    _assert_chain_erased(remaining)
    assert_l2_close(out, expected, torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_through_lowered_graph(dtype):
    """Autograd is derived from the custom op's registered backward.

    This is the payoff for running as a pre-grad pass rather than post-grad.
    """
    x0, k0, s0 = _bhl_inputs()
    grad_out = torch.randn_like(x0)

    def run(fn, cast_to):
        x = x0.to(cast_to).detach().requires_grad_()
        k = k0.to(cast_to).detach().requires_grad_()
        s = s0.to(cast_to).detach().requires_grad_()
        fn(x, k, s).backward(grad_out.to(cast_to))
        return x.grad, k.grad, s.grad

    expected = run(fftconv2d_fp32_bhl, torch.float32)
    compiled = torch.compile(fftconv2d_fp32_bhl, fullgraph=True, options=fused_fftconv2d_options())
    got = run(compiled, dtype)

    assert lowering_stats().get("rewritten") == 1
    for actual, want, name in zip(got, expected, ("grad_x", "grad_kernel", "grad_shortcut")):
        assert_l2_close(actual, want, dtype, tol_table=L2_TOL_GRAD, name=name)


# ---------------------------------------------------------------------------
# Guards — the pass must decline, and declining must be a no-op
# ---------------------------------------------------------------------------


def _assert_declined(fn, args, reason):
    """Assert the pass skipped for ``reason`` and left the eager path in place.

    The counters are the exact proof that no rewrite happened. The numeric check
    is a second line of defence and is deliberately *not* bit-exact: inductor
    perturbs results on its own through fusion and reduction ordering, so
    compiled-without-our-pass differs from eager at the 1e-6 (fp32) level
    regardless. What it does rule out is a rewrite having slipped through, which
    would show up far above this budget.
    """
    expected = fn(*args)
    out = _compile_with_lowering(fn, *args)

    stats = lowering_stats()
    assert stats.get("rewritten", 0) == 0, f"expected no rewrite, got {stats}"
    assert stats.get(f"skipped:{reason}") == 1, f"expected skip reason {reason!r}, got {stats}"
    assert_l2_close(out, expected, out.dtype, name="declined-path")


@pytest.mark.parametrize("spatial", [65, 96, 128])
def test_declines_oversized_spatial(spatial):
    """Beyond the 64x64 cap the pass must leave the eager cuFFT path alone."""
    x, kernel, shortcut = _bhl_inputs(spatial=spatial, kernel_size=2 * spatial - 1, channels=4, batch=1)
    _assert_declined(fftconv2d_fp32_bhl, (x, kernel, shortcut), "spatial-too-large")


def test_declines_on_cpu():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16)
    kernel = torch.randn(1, 4, 31, 31) * 0.05
    _assert_declined(fftconv2d_fp32_bhl, (x, kernel), "arch-unsupported")


def test_declines_a_different_crop_convention():
    """A same-shaped FFT conv that is not the reference recipe must be untouched.

    This is the false-positive guard: identical node types, different crop.
    """

    def other_crop(x, kernel):
        spec = torch.fft.rfft2(x, s=(32, 32), dim=(2, 3)) * torch.fft.rfft2(kernel, s=(32, 32), dim=(2, 3))
        return torch.fft.irfft2(spec, s=(32, 32), dim=(2, 3))[..., 0:16, 0:16]

    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16, device="cuda")
    kernel = torch.randn(1, 4, 31, 31, device="cuda") * 0.05
    _assert_declined(other_crop, (x, kernel), "not-the-reference-recipe")


def test_declines_when_spectrum_has_other_users():
    """Extra math on the spectrum is not part of the recipe and must block the rewrite.

    Rewriting here would silently drop the extra scale, since the fused kernel
    never materialises the spectrum.
    """
    x, kernel, _ = _bhl_inputs()

    def scaled(x, kernel):
        def multiply(a, b):
            a.mul_(b)
            a.mul_(2.0)
            return a

        return _reference_recipe(x, kernel, multiply=multiply)

    _assert_declined(scaled, (x, kernel), "spectrum-has-other-users")


def test_declines_inplace_multiply_after_irfft():
    """A ``mul_`` ordered after the ``irfft2`` does not feed it; eager is not a convolution."""
    x, kernel, _ = _bhl_inputs()

    def misordered(x, kernel):
        dim_x, dim_y = x.shape[-2:]
        k_x, k_y = kernel.shape[-2:]
        s = (min(dim_x + (k_x + 1) // 2, 2 * dim_x), min(dim_y + (k_y + 1) // 2, 2 * dim_y))
        spec_x = torch.fft.rfft2(x.float(), s=s, dim=(2, 3))
        spec_k = torch.fft.rfft2(kernel.float(), s=s, dim=(2, 3))
        out = torch.fft.irfft2(spec_x, s=s, dim=(2, 3))
        out = out[..., k_x // 2 : k_x // 2 + dim_x, k_y // 2 : k_y // 2 + dim_y].to(x.dtype)
        spec_x.mul_(spec_k)
        return out

    _assert_declined(misordered, (x, kernel), "no-rfft2-product")


def test_declines_non_fp32_upcast():
    """An fp64 conv is a precision choice, not the reference's fp32 upcast; keep it."""
    x, kernel, _ = _bhl_inputs()

    def fp64_conv(x, kernel):
        dim_x, dim_y = x.shape[-2:]
        k_x, k_y = kernel.shape[-2:]
        s = (min(dim_x + (k_x + 1) // 2, 2 * dim_x), min(dim_y + (k_y + 1) // 2, 2 * dim_y))
        spec_x = torch.fft.rfft2(x.double(), s=s, dim=(2, 3))
        spec_k = torch.fft.rfft2(kernel.double(), s=s, dim=(2, 3))
        spec_x.mul_(spec_k)
        out = torch.fft.irfft2(spec_x, s=s, dim=(2, 3))
        return out[..., k_x // 2 : k_x // 2 + dim_x, k_y // 2 : k_y // 2 + dim_y].to(x.dtype)

    _assert_declined(fp64_conv, (x, kernel), "unsupported-dtype")


def test_declines_non_default_norm():
    """``norm="ortho"`` rescales by 1/sqrt(N); the fused kernel is backward-normalised only."""
    x, kernel, _ = _bhl_inputs()

    def ortho_conv(x, kernel):
        dim_x, dim_y = x.shape[-2:]
        k_x, k_y = kernel.shape[-2:]
        s = (min(dim_x + (k_x + 1) // 2, 2 * dim_x), min(dim_y + (k_y + 1) // 2, 2 * dim_y))
        spec_x = torch.fft.rfft2(x.float(), s=s, dim=(2, 3), norm="ortho")
        spec_k = torch.fft.rfft2(kernel.float(), s=s, dim=(2, 3), norm="ortho")
        spec_x.mul_(spec_k)
        out = torch.fft.irfft2(spec_x, s=s, dim=(2, 3), norm="ortho")
        return out[..., k_x // 2 : k_x // 2 + dim_x, k_y // 2 : k_y // 2 + dim_y].to(x.dtype)

    _assert_declined(ortho_conv, (x, kernel), "irfft-args-mismatch")


def test_declines_kernel_dtype_mismatch():
    """An fp32 kernel with bf16 activations stays fp32 through the reference FFT.

    The fused kernel would round it to bf16 first, which the reference never does.
    """
    x, _, _ = _bhl_inputs(dtype=torch.bfloat16)
    _, kernel, _ = _bhl_inputs(dtype=torch.float32)
    _assert_declined(fftconv2d_fp32_bhl, (x, kernel), "kernel-dtype-or-device-mismatch")


def test_declines_when_output_is_not_cast_back():
    """Without the trailing ``.to(x.dtype)`` the graph observes an fp32 result; keep it."""
    x, kernel, _ = _bhl_inputs(dtype=torch.bfloat16)

    def uncast(x, kernel):
        dim_x, dim_y = x.shape[-2:]
        k_x, k_y = kernel.shape[-2:]
        s = (min(dim_x + (k_x + 1) // 2, 2 * dim_x), min(dim_y + (k_y + 1) // 2, 2 * dim_y))
        spec = torch.fft.rfft2(x.float(), s=s, dim=(2, 3)) * torch.fft.rfft2(kernel.float(), s=s, dim=(2, 3))
        out = torch.fft.irfft2(spec, s=s, dim=(2, 3))
        return out[..., k_x // 2 : k_x // 2 + dim_x, k_y // 2 : k_y // 2 + dim_y]

    _assert_declined(uncast, (x, kernel), "output-dtype-or-device-mismatch")


def test_rewrites_keyword_input_spelling():
    """``rfft2(input=...)`` is the same chain; it must neither crash nor be missed."""
    x, kernel, _ = _bhl_inputs()

    def keyword_conv(x, kernel):
        dim_x, dim_y = x.shape[-2:]
        k_x, k_y = kernel.shape[-2:]
        s = (min(dim_x + (k_x + 1) // 2, 2 * dim_x), min(dim_y + (k_y + 1) // 2, 2 * dim_y))
        spec_x = torch.fft.rfft2(input=x.float(), s=s, dim=(2, 3))
        spec_k = torch.fft.rfft2(input=kernel.float(), s=s, dim=(2, 3))
        spec_x.mul_(spec_k)
        out = torch.fft.irfft2(input=spec_x, s=s, dim=(2, 3))
        return out[..., k_x // 2 : k_x // 2 + dim_x, k_y // 2 : k_y // 2 + dim_y].to(x.dtype)

    expected = keyword_conv(x, kernel)
    out, remaining = _compile_and_record(keyword_conv, x, kernel)

    assert lowering_stats().get("rewritten") == 1
    _assert_chain_erased(remaining)
    assert_l2_close(out, expected, torch.float32)


def test_declines_a_different_fft_size():
    """Right crop offset, wrong padding recipe — still not ours."""

    def oversized_fft(x, kernel):
        spec = torch.fft.rfft2(x, s=(64, 64), dim=(2, 3)) * torch.fft.rfft2(kernel, s=(64, 64), dim=(2, 3))
        return torch.fft.irfft2(spec, s=(64, 64), dim=(2, 3))[..., 15:31, 15:31]

    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16, device="cuda")
    kernel = torch.randn(1, 4, 31, 31, device="cuda") * 0.05
    _assert_declined(oversized_fft, (x, kernel), "not-the-reference-recipe")


def test_declines_reduced_precision_when_disabled():
    """``allow_reduced_precision=False`` keeps bf16 graphs on the fp32-internal path."""
    x, kernel, shortcut = _bhl_inputs(dtype=torch.bfloat16)
    expected = fftconv2d_fp32_bhl(x, kernel, shortcut)

    out = _compile_with_lowering(fftconv2d_fp32_bhl, x, kernel, shortcut, allow_reduced_precision=False)

    stats = lowering_stats()
    assert stats.get("rewritten", 0) == 0
    assert stats.get("skipped:reduced-precision-disabled") == 1
    # The counters above are the real assertion. Numerics cannot distinguish the
    # two paths here: both round the result to bf16 at the end, and that final
    # cast alone accounts for ~3e-3 normwise — the same order as the native-bf16
    # kernel's own error.
    assert_l2_close(out, expected, torch.bfloat16)


def test_fp32_still_rewritten_when_reduced_precision_disabled():
    """The flag gates only reduced precision, where the rewrite changes numerics."""
    x, kernel, shortcut = _bhl_inputs()
    expected = fftconv2d_fp32_bhl(x, kernel, shortcut)

    out = _compile_with_lowering(fftconv2d_fp32_bhl, x, kernel, shortcut, allow_reduced_precision=False)

    assert lowering_stats().get("rewritten") == 1
    assert_l2_close(out, expected, torch.float32)


def test_no_fft_in_graph_is_untouched():
    """A graph with no irfft2 exits early without touching anything."""

    def plain(x):
        return x * 2 + 1

    x = torch.randn(4, 4, device="cuda")
    out = _compile_with_lowering(plain, x)

    assert lowering_stats() == {}
    torch.testing.assert_close(out, plain(x), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Registration plumbing
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_options_route_does_not_touch_global_config(self):
        """The ``options=`` patch is scoped to one callable, unlike the context manager."""
        from torch._inductor import config as inductor_config

        assert inductor_config.pre_grad_custom_pass is None
        x, kernel, shortcut = _bhl_inputs()

        torch.compile(fftconv2d_fp32_bhl, fullgraph=True, options=fused_fftconv2d_options())(x, kernel, shortcut)

        assert lowering_stats().get("rewritten") == 1
        assert inductor_config.pre_grad_custom_pass is None

    def test_context_manager_route_also_rewrites(self):
        """The global-config route stays supported for framework-owned compiles."""
        x, kernel, shortcut = _bhl_inputs()
        expected = fftconv2d_fp32_bhl(x, kernel, shortcut)

        with fused_fftconv2d_lowering():
            out = torch.compile(fftconv2d_fp32_bhl, fullgraph=True)(x, kernel, shortcut)

        assert lowering_stats().get("rewritten") == 1
        assert_l2_close(out, expected, torch.float32)

    def test_options_carries_the_precision_flag(self):
        options = fused_fftconv2d_options(allow_reduced_precision=False)
        assert options["pre_grad_custom_pass"].allow_reduced_precision is False

    def test_restores_previous_pass_on_exit(self):
        from torch._inductor import config as inductor_config

        sentinel = FusedFFTConv2dLowering()
        inductor_config.pre_grad_custom_pass = sentinel
        try:
            with fused_fftconv2d_lowering():
                assert inductor_config.pre_grad_custom_pass is not sentinel
            assert inductor_config.pre_grad_custom_pass is sentinel
        finally:
            inductor_config.pre_grad_custom_pass = None

    def test_restores_on_exception(self):
        from torch._inductor import config as inductor_config

        assert inductor_config.pre_grad_custom_pass is None
        with pytest.raises(RuntimeError, match="boom"):
            with fused_fftconv2d_lowering():
                raise RuntimeError("boom")
        assert inductor_config.pre_grad_custom_pass is None

    def test_uuid_is_stable_and_flag_dependent(self):
        """Inductor mixes uuid() into its cache key, so it must track behaviour."""
        assert FusedFFTConv2dLowering(True).uuid() == FusedFFTConv2dLowering(True).uuid()
        assert FusedFFTConv2dLowering(True).uuid() != FusedFFTConv2dLowering(False).uuid()

    def test_not_installed_means_no_rewrite(self):
        """Without the context manager, torch.compile leaves the chain as-is."""
        x, kernel, shortcut = _bhl_inputs()
        expected = fftconv2d_fp32_bhl(x, kernel, shortcut)

        out = torch.compile(fftconv2d_fp32_bhl, fullgraph=True)(x, kernel, shortcut)

        assert lowering_stats() == {}
        assert_l2_close(out, expected, torch.float32)
