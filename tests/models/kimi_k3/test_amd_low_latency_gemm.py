# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `_build_plan` (AMD Kimi-K3 low-latency GEMM planning)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.models.kimi_k3.amd.low_latency_gemm import (
    KimiK3LowLatencyEmbeddingMethod,
    KimiK3LowLatencyLinearMethod,
    _build_plan,
    _run_plan,
    enable_kimi_k3_low_latency_gemm,
    try_low_latency_gemm,
)
from vllm.platforms.rocm import on_gfx950
from vllm.utils.platform_utils import num_compute_units


# Fixed test knobs, independent from the boundary being exercised.
# OUTPUT_SIZE is kept small enough that wvsplitkrc_dispatch(...).fits is True
# across num_tokens in [10, 128] and k in [769, 1024] at CU_COUNT below;
# larger output sizes exceed the CU budget for large num_tokens/k.
OUTPUT_SIZE = 2048  # divisible by 16, >= 8448 threshold not tested here
K = 1024  # > 768
CU_COUNT = 304


@pytest.mark.parametrize(
    "num_tokens,expected_backend",
    [
        (1, "wvsplitk"),
        (4, "wvsplitk"),
        (5, "wvsplitk"),
        (6, None),
        (9, None),
        (10, "wvsplitkrc"),
        (128, "wvsplitkrc"),
        (129, None),
    ],
)
def test_build_plan_backend_boundaries(num_tokens, expected_backend):
    plan = _build_plan(OUTPUT_SIZE, K, CU_COUNT)
    assert plan.get(num_tokens) == expected_backend


@pytest.mark.parametrize(
    "output_size,expected_backend",
    [
        (8447, "wvsplitk"),
        (8448, None),
    ],
)
def test_build_plan_wvsplitk_output_size_regression_edge(
    output_size, expected_backend
):
    plan = _build_plan(output_size, K, CU_COUNT)
    assert plan.get(5) == expected_backend


@pytest.mark.parametrize(
    "k,expected_backend",
    [
        (768, None),
        (769, "wvsplitkrc"),
    ],
)
def test_build_plan_wvsplitkrc_k_regression_edge(k, expected_backend):
    plan = _build_plan(OUTPUT_SIZE, k, CU_COUNT)
    assert plan.get(64) == expected_backend


def test_build_plan_wvsplitkrc_requires_output_size_multiple_of_16():
    plan = _build_plan(OUTPUT_SIZE + 1, K, CU_COUNT)
    assert plan.get(64) is None


def test_build_plan_omits_entry_when_wvsplitkrc_dispatch_does_not_fit():
    with patch(
        "vllm.models.kimi_k3.amd.low_latency_gemm.wvsplitkrc_dispatch",
        return_value=(None, False),
    ):
        plan = _build_plan(OUTPUT_SIZE, K, CU_COUNT)
    assert plan.get(64) is None


def test_build_plan_omits_entry_when_cu_budget_too_small():
    # CuNeeded for (num_tokens=128, k=K, output_size=OUTPUT_SIZE) is 256
    # (see wvsplitkrc_dispatch); a CU count well below that must fail the
    # real (unmocked) dispatch fit check, not just the mocked one above.
    plan = _build_plan(OUTPUT_SIZE, K, cu_count=8)
    assert plan.get(128) is None
    assert plan.get(64) is None


@pytest.mark.skipif(
    not (torch.cuda.is_available() and on_gfx950()),
    reason="try_low_latency_gemm requires a gfx950 CUDA/ROCm device",
)
def test_build_plan_matches_try_low_latency_gemm_decision():
    """`_build_plan` must pick the same backend try_low_latency_gemm would,
    for every num_tokens in its supported range, given the same statics."""
    cu_count = num_compute_units()
    plan = _build_plan(OUTPUT_SIZE, K, cu_count)

    device = "cuda"
    dtype = torch.bfloat16
    weight = torch.randn(OUTPUT_SIZE, K, dtype=dtype, device=device)

    for num_tokens in range(1, 129):
        x = torch.randn(num_tokens, K, dtype=dtype, device=device)
        wvsplitk_out = torch.zeros(
            num_tokens, OUTPUT_SIZE, dtype=dtype, device=device
        )
        wvsplitkrc_out = torch.zeros(
            num_tokens, OUTPUT_SIZE, dtype=dtype, device=device
        )
        with (
            patch(
                "vllm.models.kimi_k3.amd.low_latency_gemm.ops.wvSplitK",
                MagicMock(return_value=wvsplitk_out),
            ) as mock_wvsplitk,
            patch(
                "vllm.models.kimi_k3.amd.low_latency_gemm.ops.wvSplitKrc",
                MagicMock(return_value=wvsplitkrc_out),
            ) as mock_wvsplitkrc,
        ):
            try_low_latency_gemm(x, weight)

        if mock_wvsplitk.called:
            actual_backend = "wvsplitk"
        elif mock_wvsplitkrc.called:
            actual_backend = "wvsplitkrc"
        else:
            actual_backend = None

        assert plan.get(num_tokens) == actual_backend, (
            f"num_tokens={num_tokens}: plan says "
            f"{plan.get(num_tokens)!r}, try_low_latency_gemm used "
            f"{actual_backend!r}"
        )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and on_gfx950()),
    reason="_run_plan requires a gfx950 CUDA/ROCm device",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 5, 10, 64, 128])
def test_run_plan_matches_reference_linear(num_tokens):
    """The real kernels dispatched by `_run_plan` must match a plain
    `torch.nn.functional.linear` reference within BF16 tolerance, for every
    boundary num_tokens the plan claims to cover."""
    cu_count = num_compute_units()
    plan = _build_plan(OUTPUT_SIZE, K, cu_count)
    assert plan.get(num_tokens) is not None, (
        f"num_tokens={num_tokens} expected to be covered by the plan"
    )

    device = "cuda"
    dtype = torch.bfloat16
    weight = torch.randn(OUTPUT_SIZE, K, dtype=dtype, device=device)
    x = torch.randn(num_tokens, K, dtype=dtype, device=device)

    actual = _run_plan(plan, x, weight)
    # fp32 accumulation reference: isolates the kernel's own numerical error
    # from the BF16-rounding error already inherent in a BF16 `linear` call.
    expected = torch.nn.functional.linear(x.float(), weight.float())

    assert actual is not None
    torch.testing.assert_close(
        actual.float(), expected, rtol=1e-2, atol=1e-2
    )
    relative_l2_error = torch.linalg.norm(
        actual.float() - expected
    ) / torch.linalg.norm(expected)
    assert relative_l2_error < 1e-2


@pytest.mark.skipif(
    not (torch.cuda.is_available() and on_gfx950()),
    reason="apply() with a real weight requires a gfx950 CUDA/ROCm device",
)
def test_apply_with_bias_falls_back_to_correct_base_output():
    """When bias is not None, `_KimiK3LowLatencyApply.apply` must defer to
    the base (super) method rather than the plan, and that fallback output
    must itself be numerically correct (not just "taken")."""
    cu_count = num_compute_units()
    plan = _build_plan(OUTPUT_SIZE, K, cu_count)
    assert plan.get(4) is not None  # would be dispatched if bias were None

    device = "cuda"
    dtype = torch.bfloat16
    weight = torch.randn(OUTPUT_SIZE, K, dtype=dtype, device=device)
    bias = torch.randn(OUTPUT_SIZE, dtype=dtype, device=device)
    x = torch.randn(4, K, dtype=dtype, device=device)
    layer = SimpleNamespace(weight=weight)

    # Patch our own dispatch function rather than ops.wvSplitK: the base
    # (super) gemm impl may itself legitimately use wvSplitK internally for
    # bias-including small-token-count calls on gfx950, so asserting on the
    # shared kernel would conflate "our plan fired" with "the baseline op
    # made its own independent choice to use the same kernel."
    with patch(
        "vllm.models.kimi_k3.amd.low_latency_gemm._run_plan"
    ) as mock_run_plan:
        method = KimiK3LowLatencyLinearMethod(plan)
        actual = method.apply(layer, x, bias)

    mock_run_plan.assert_not_called()
    expected = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)


def _make_replicated_linear(output_size: int, input_size: int) -> ReplicatedLinear:
    # BasevLLMParameter.__init__ reads the TP rank/world size unconditionally,
    # even for disable_tp=True layers, so a process group must be mocked out
    # for construction outside of a distributed test environment.
    with (
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        return ReplicatedLinear(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            disable_tp=True,
        )


def test_enable_kimi_k3_low_latency_gemm_swaps_linear_when_plan_nonempty():
    linear = _make_replicated_linear(OUTPUT_SIZE, K)
    module = nn.Sequential(linear)

    with (
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.on_gfx950",
            return_value=True,
        ),
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.num_compute_units",
            return_value=CU_COUNT,
        ),
    ):
        enable_kimi_k3_low_latency_gemm(module, torch.bfloat16)

    assert isinstance(linear.quant_method, KimiK3LowLatencyLinearMethod)
    assert linear.quant_method._plan == _build_plan(OUTPUT_SIZE, K, CU_COUNT)


def test_enable_kimi_k3_low_latency_gemm_leaves_linear_when_plan_empty():
    # An output_size not divisible by 16 fails wvsplitkrc's n % 16 == 0
    # requirement for every num_tokens in [10, 128]; with wvSplitK also
    # unavailable, _build_plan has nothing to offer for any num_tokens.
    linear = _make_replicated_linear(OUTPUT_SIZE + 1, K)
    original_method = linear.quant_method
    module = nn.Sequential(linear)

    class _OpsWithoutWvSplitK:
        wvSplitKrc = staticmethod(lambda *args, **kwargs: None)

    with (
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.on_gfx950",
            return_value=True,
        ),
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.num_compute_units",
            return_value=CU_COUNT,
        ),
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.ops",
            _OpsWithoutWvSplitK(),
        ),
    ):
        enable_kimi_k3_low_latency_gemm(module, torch.bfloat16)

    assert linear.quant_method is original_method


def test_enable_kimi_k3_low_latency_gemm_noop_when_dtype_not_bfloat16():
    linear = _make_replicated_linear(OUTPUT_SIZE, K)
    original_method = linear.quant_method
    module = nn.Sequential(linear)

    with (
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.on_gfx950",
            return_value=True,
        ),
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.num_compute_units",
            return_value=CU_COUNT,
        ),
    ):
        enable_kimi_k3_low_latency_gemm(module, torch.float16)

    assert linear.quant_method is original_method


def test_enable_kimi_k3_low_latency_gemm_noop_when_not_gfx950():
    linear = _make_replicated_linear(OUTPUT_SIZE, K)
    original_method = linear.quant_method
    module = nn.Sequential(linear)

    with (
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.on_gfx950",
            return_value=False,
        ),
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.num_compute_units",
            return_value=CU_COUNT,
        ),
    ):
        enable_kimi_k3_low_latency_gemm(module, torch.bfloat16)

    assert linear.quant_method is original_method


def test_enable_kimi_k3_low_latency_gemm_leaves_non_unquantized_method_untouched():
    linear = _make_replicated_linear(OUTPUT_SIZE, K)
    quantized_method = MagicMock()
    linear.quant_method = quantized_method
    module = nn.Sequential(linear)

    with (
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.on_gfx950",
            return_value=True,
        ),
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.num_compute_units",
            return_value=CU_COUNT,
        ),
    ):
        enable_kimi_k3_low_latency_gemm(module, torch.bfloat16)

    assert linear.quant_method is quantized_method


def test_enable_kimi_k3_low_latency_gemm_swaps_lm_head_when_plan_nonempty():
    with (
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        lm_head = ParallelLMHead(
            num_embeddings=OUTPUT_SIZE,
            embedding_dim=K,
            disable_tp=True,
        )
    module = nn.Sequential(lm_head)

    with (
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.on_gfx950",
            return_value=True,
        ),
        patch(
            "vllm.models.kimi_k3.amd.low_latency_gemm.num_compute_units",
            return_value=CU_COUNT,
        ),
    ):
        enable_kimi_k3_low_latency_gemm(module, torch.bfloat16)

    assert isinstance(lm_head.quant_method, KimiK3LowLatencyEmbeddingMethod)
    assert lm_head.quant_method._plan == _build_plan(OUTPUT_SIZE, K, CU_COUNT)
