# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AITER fused top-k gating fast path in ``fused_topk_bias``.

The fast path calls ``aiter.ops.flydsl.moe_sorting.fused_topk_gating`` when
enabled via ``AITER_USE_FUSED_TOPK_MOE_SORT`` and the routing shape matches
Kimi-K3's decode profile (M<=16, 896 routed experts, top-16, softmax
scoring). Everything outside that gate must fall back to the existing
baseline path, unmodified.
"""

import sys
import types
from unittest import mock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)

NUM_EXPERTS = 896
TOPK = 16
MODEL_DIM = 16

# The baseline (non-fast-path) code in fused_topk_bias calls CUDA/ROCm-only
# custom ops (ops.topk_softmax etc.), so fallback-path tests need a real
# accelerator; the fast-path-fires tests mock fused_topk_gating entirely and
# don't need one.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
requires_accelerator = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="baseline fused_topk_bias path requires a CUDA/ROCm device",
)


def _real_aiter_fused_topk_gating_available() -> bool:
    """Whether the installed aiter actually has the gating-only split
    function (not just whatever gets mocked in sys.modules by other tests
    in this file). See tasks/fused_topk_softmax_moe_sort/check_aiter_fused_topk.py
    for the fuller standalone version of this check."""
    try:
        from aiter.ops.flydsl.moe_sorting import fused_topk_gating  # noqa: F401

        return True
    except Exception:
        return False


requires_real_aiter_fused_topk_gating = pytest.mark.skipif(
    not (torch.cuda.is_available() and _real_aiter_fused_topk_gating_available()),
    reason="requires a CUDA/ROCm device and an aiter install with "
    "aiter.ops.flydsl.moe_sorting.fused_topk_gating (the split gating-only "
    "function); see check_aiter_fused_topk.py to diagnose a stale/broken "
    "aiter install",
)


def _make_inputs(m: int, num_experts: int = NUM_EXPERTS, device: str = DEVICE):
    hidden_states = torch.randn(m, MODEL_DIM, device=device)
    gating_output = torch.randn(m, num_experts, device=device)
    return hidden_states, gating_output


def _install_fake_fused_topk_gating(monkeypatch, fn):
    """Inject a fake aiter.ops.flydsl.moe_sorting.fused_topk_gating so the
    test doesn't depend on aiter actually being installed."""
    moe_sorting_mod = types.ModuleType("aiter.ops.flydsl.moe_sorting")
    moe_sorting_mod.fused_topk_gating = fn
    flydsl_mod = types.ModuleType("aiter.ops.flydsl")
    flydsl_mod.moe_sorting = moe_sorting_mod
    ops_mod = types.ModuleType("aiter.ops")
    ops_mod.flydsl = flydsl_mod
    aiter_mod = types.ModuleType("aiter")
    aiter_mod.ops = ops_mod

    monkeypatch.setitem(sys.modules, "aiter", aiter_mod)
    monkeypatch.setitem(sys.modules, "aiter.ops", ops_mod)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", flydsl_mod)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.moe_sorting", moe_sorting_mod)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
def test_fast_path_fires_and_uses_return_value(monkeypatch, m):
    monkeypatch.setenv("AITER_USE_FUSED_TOPK_MOE_SORT", "1")
    hidden_states, gating_output = _make_inputs(m)

    expected_weights = torch.rand(m, TOPK)
    expected_indices = torch.randint(0, NUM_EXPERTS, (m, TOPK), dtype=torch.int32)
    fake_fn = mock.Mock(return_value=(expected_weights, expected_indices))
    _install_fake_fused_topk_gating(monkeypatch, fake_fn)

    topk_weights, topk_indices = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="softmax",
        e_score_correction_bias=None,
        topk=TOPK,
        renormalize=True,
    )

    fake_fn.assert_called_once()
    args, kwargs = fake_fn.call_args
    assert args[0] is gating_output
    assert kwargs["topk"] == TOPK
    assert kwargs["scoring_func"] == "softmax"
    assert kwargs["need_renorm"] is True

    torch.testing.assert_close(topk_weights, expected_weights)
    assert torch.equal(topk_indices, expected_indices.to(torch.int32))


def test_fast_path_applies_routed_scaling_factor(monkeypatch):
    monkeypatch.setenv("AITER_USE_FUSED_TOPK_MOE_SORT", "1")
    m = 4
    hidden_states, gating_output = _make_inputs(m)

    raw_weights = torch.rand(m, TOPK)
    raw_indices = torch.randint(0, NUM_EXPERTS, (m, TOPK), dtype=torch.int32)
    fake_fn = mock.Mock(return_value=(raw_weights, raw_indices))
    _install_fake_fused_topk_gating(monkeypatch, fake_fn)

    scale = 2.5
    topk_weights, _ = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="softmax",
        e_score_correction_bias=None,
        topk=TOPK,
        renormalize=True,
        routed_scaling_factor=scale,
    )

    torch.testing.assert_close(topk_weights, raw_weights * scale)


def _baseline_fallback_matches(monkeypatch, **overrides):
    """Call fused_topk_bias once with a fake fused_topk_gating that would
    raise if called, and once with the env var off, and assert both match
    the same `torch.manual_seed`-fixed baseline result."""
    kwargs = dict(
        scoring_func="softmax",
        e_score_correction_bias=None,
        topk=TOPK,
        renormalize=True,
    )
    kwargs.update(overrides)

    def run():
        torch.manual_seed(0)
        m = kwargs.pop("_m", 4)
        num_experts = kwargs.pop("_num_experts", NUM_EXPERTS)
        hidden_states, gating_output = _make_inputs(m, num_experts=num_experts)
        return fused_topk_bias(
            hidden_states=hidden_states, gating_output=gating_output, **kwargs
        )

    # Env var set, but a poison fused_topk_gating that must not be reached
    # for out-of-gate cases (assert it's never called).
    monkeypatch.setenv("AITER_USE_FUSED_TOPK_MOE_SORT", "1")
    poison = mock.Mock(side_effect=AssertionError("fast path should not fire"))
    _install_fake_fused_topk_gating(monkeypatch, poison)
    kwargs_copy = dict(kwargs)
    out_with_gate_on = run()
    poison.assert_not_called()

    # Env var unset entirely -- same result expected.
    monkeypatch.delenv("AITER_USE_FUSED_TOPK_MOE_SORT", raising=False)
    kwargs.clear()
    kwargs.update(kwargs_copy)
    out_with_gate_off = run()

    torch.testing.assert_close(out_with_gate_on[0], out_with_gate_off[0])
    assert torch.equal(out_with_gate_on[1], out_with_gate_off[1])


@requires_accelerator
def test_fallback_for_prefill_m32(monkeypatch):
    _baseline_fallback_matches(monkeypatch, _m=32)


@requires_accelerator
def test_fallback_for_wrong_num_experts(monkeypatch):
    _baseline_fallback_matches(monkeypatch, _m=4, _num_experts=256)


@requires_accelerator
def test_fallback_when_gate_disabled(monkeypatch):
    monkeypatch.delenv("AITER_USE_FUSED_TOPK_MOE_SORT", raising=False)
    hidden_states, gating_output = _make_inputs(4)
    # Should simply run the baseline path without error; no mocking needed
    # since the gate's first condition (env var) already fails closed.
    topk_weights, topk_indices = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="softmax",
        e_score_correction_bias=None,
        topk=TOPK,
        renormalize=True,
    )
    assert topk_weights.shape == (4, TOPK)
    assert topk_indices.shape == (4, TOPK)


@requires_accelerator
def test_fallback_when_bias_vl_set(monkeypatch):
    monkeypatch.setenv("AITER_USE_FUSED_TOPK_MOE_SORT", "1")
    m = 4
    hidden_states, gating_output = _make_inputs(m)
    input_tokens = torch.zeros(m, dtype=torch.long, device=DEVICE)
    bias_vl = torch.zeros(NUM_EXPERTS, device=DEVICE)

    poison = mock.Mock(side_effect=AssertionError("fast path should not fire"))
    _install_fake_fused_topk_gating(monkeypatch, poison)

    fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="softmax",
        e_score_correction_bias=None,
        topk=TOPK,
        renormalize=True,
        input_tokens=input_tokens,
        bias_vl=bias_vl,
        image_sentinel_lo=1,
    )
    poison.assert_not_called()


@requires_accelerator
def test_fallback_when_hash_indices_table_set(monkeypatch):
    monkeypatch.setenv("AITER_USE_FUSED_TOPK_MOE_SORT", "1")
    m = 4
    hidden_states, gating_output = _make_inputs(m)
    input_tokens = torch.zeros(m, dtype=torch.long, device=DEVICE)
    hash_indices_table = torch.zeros(10, TOPK, dtype=torch.long, device=DEVICE)

    poison = mock.Mock(side_effect=AssertionError("fast path should not fire"))
    _install_fake_fused_topk_gating(monkeypatch, poison)

    fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="softmax",
        e_score_correction_bias=None,
        topk=TOPK,
        renormalize=True,
        input_tokens=input_tokens,
        hash_indices_table=hash_indices_table,
    )
    poison.assert_not_called()


@requires_real_aiter_fused_topk_gating
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
def test_fast_path_matches_baseline_against_real_aiter(monkeypatch, m):
    """End-to-end check against the real installed aiter (not mocked): the
    fast path's output should match the baseline softmax+topk path within
    float tolerance for weights and exactly for indices (same top-k set).

    This is the one test in this file that actually exercises the installed
    aiter build -- everything else mocks fused_topk_gating out. If this test
    is skipped, the aiter install doesn't have the split function at all
    (stale/broken build); if it's collected but fails, the real
    fused_topk_gating disagrees with vLLM's own topk_softmax baseline.
    """
    monkeypatch.delenv("AITER_USE_FUSED_TOPK_MOE_SORT", raising=False)
    torch.manual_seed(0)
    hidden_states, gating_output = _make_inputs(m)

    baseline_weights, baseline_indices = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="softmax",
        e_score_correction_bias=None,
        topk=TOPK,
        renormalize=True,
    )

    monkeypatch.setenv("AITER_USE_FUSED_TOPK_MOE_SORT", "1")
    fused_weights, fused_indices = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="softmax",
        e_score_correction_bias=None,
        topk=TOPK,
        renormalize=True,
    )

    # Same set of selected experts per token, order may legitimately differ.
    baseline_sorted, _ = torch.sort(baseline_indices, dim=-1)
    fused_sorted, _ = torch.sort(fused_indices, dim=-1)
    assert torch.equal(baseline_sorted, fused_sorted)

    baseline_by_expert = torch.zeros(m, NUM_EXPERTS, device=DEVICE)
    baseline_by_expert.scatter_(1, baseline_indices.long(), baseline_weights)
    fused_by_expert = torch.zeros(m, NUM_EXPERTS, device=DEVICE)
    fused_by_expert.scatter_(1, fused_indices.long(), fused_weights)
    torch.testing.assert_close(
        fused_by_expert, baseline_by_expert, atol=1e-3, rtol=1e-3
    )
