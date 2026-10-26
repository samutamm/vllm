# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing of QuickReduce to AITER's FlyDSL INT4 schedules
(VLLM_ROCM_QUICK_REDUCE_USE_FLYDSL). The AITER engines are stand-ins: these
tests check what vLLM sends where, not the kernels."""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm.envs as envs
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm-only quick-reduce tests",
)

_BACKENDS = "aiter.dist.device_communicators.flydsl_backends"


def _make_qr(*, fly_ar=None, fly_rms=None, disabled=False):
    from vllm.distributed.device_communicators.quick_all_reduce import (
        QuickAllReduce,
    )

    qr = QuickAllReduce.__new__(QuickAllReduce)
    qr.disabled = disabled
    qr._fly_ar = fly_ar
    qr._fly_rms = fly_rms
    return qr


def _fake_backends(*, ar_disabled=False, rms_disabled=False):
    """A stand-in for AITER's flydsl_backends module, and its two engines."""
    ar = MagicMock(disabled=ar_disabled)
    rms = MagicMock(disabled=rms_disabled)
    module = types.ModuleType(_BACKENDS)
    module.FlyDSLQuickAllReduce = MagicMock(return_value=ar)
    module.FlyDSLAllReduceRMSNorm = MagicMock(return_value=rms)
    return module, ar, rms


def test_use_flydsl_env_default(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_QUICK_REDUCE_USE_FLYDSL", raising=False)
    assert importlib.reload(envs).VLLM_ROCM_QUICK_REDUCE_USE_FLYDSL is False


@pytest.mark.parametrize("value", ["1", "true", "True"])
def test_use_flydsl_env_enabled(monkeypatch, value):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_USE_FLYDSL", value)
    assert importlib.reload(envs).VLLM_ROCM_QUICK_REDUCE_USE_FLYDSL is True


def test_hip_path_does_not_use_flydsl():
    assert not _make_qr().uses_flydsl


def test_plain_allreduce_routes_to_flydsl():
    fly_ar = MagicMock()
    fly_ar.should_fly_ar.return_value = True
    qr = _make_qr(fly_ar=fly_ar)
    inp = torch.empty(4, 8, dtype=torch.bfloat16)
    out = torch.empty_like(inp)

    assert qr.should_quick_allreduce(inp)
    fly_ar.should_fly_ar.assert_called_once_with(inp)
    qr.quick_all_reduce(inp, out=out)
    fly_ar.fly_ar.assert_called_once_with(inp, out=out)


def test_plain_allreduce_declined_by_flydsl_policy():
    fly_ar = MagicMock()
    fly_ar.should_fly_ar.return_value = False
    qr = _make_qr(fly_ar=fly_ar)
    assert not qr.should_quick_allreduce(torch.empty(4, 8, dtype=torch.bfloat16))


def test_plain_allreduce_declined_with_only_fused_engine():
    # The HIP checks must not run: there is no HIP communicator behind them.
    qr = _make_qr(fly_rms=MagicMock())
    assert not qr.should_quick_allreduce(torch.empty(4, 8, dtype=torch.bfloat16))


def test_fused_rmsnorm_routes_to_flydsl():
    fly_rms = MagicMock()
    fly_rms.should_fly_fused_ar_rms.return_value = True
    fly_rms.fly_fused_ar_rms.return_value = ("out", "residual_out")
    qr = _make_qr(fly_rms=fly_rms)
    inp, res, w = (MagicMock() for _ in range(3))

    assert qr.should_fused_allreduce_rmsnorm(inp, res, w)
    fly_rms.should_fly_fused_ar_rms.assert_called_once_with(inp, res, w)
    assert qr.fused_allreduce_rmsnorm(inp, res, w, 1e-6) == ("out", "residual_out")
    fly_rms.fly_fused_ar_rms.assert_called_once_with(inp, res, w, 1e-6)


def test_fused_rmsnorm_declines_gemma_norm():
    fly_rms = MagicMock()
    fly_rms.should_fly_fused_ar_rms.return_value = True
    qr = _make_qr(fly_rms=fly_rms)
    assert not qr.should_fused_allreduce_rmsnorm(
        MagicMock(), MagicMock(), MagicMock(), gemma_norm=True
    )


def test_fused_rmsnorm_declines_without_flydsl():
    assert not _make_qr().should_fused_allreduce_rmsnorm(
        MagicMock(), MagicMock(), MagicMock()
    )


def test_close_releases_flydsl_engines():
    fly_ar, fly_rms = MagicMock(), MagicMock()
    qr = _make_qr(fly_ar=fly_ar, fly_rms=fly_rms)
    qr.close()
    fly_ar.close.assert_called_once()
    fly_rms.close.assert_called_once()
    assert qr.disabled
    assert not qr.uses_flydsl
    assert not qr.should_quick_allreduce(torch.empty(4, 8, dtype=torch.bfloat16))


def _init_flydsl(monkeypatch, module, quantization="INT4"):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", quantization)
    importlib.reload(envs)
    qr = _make_qr(disabled=True)
    qr.group = MagicMock()
    qr.device = torch.device("cuda", 0)
    with (
        patch.dict(sys.modules, {_BACKENDS: module}),
        patch(
            "vllm.distributed.device_communicators.quick_all_reduce."
            "get_current_vllm_config_or_none",
            return_value=None,
        ),
    ):
        return qr, qr._init_flydsl()


def test_init_flydsl_builds_both_engines(monkeypatch):
    module, ar, rms = _fake_backends()
    qr, took_over = _init_flydsl(monkeypatch, module)

    assert took_over
    assert not qr.disabled
    assert qr._fly_ar is ar and qr._fly_rms is rms
    module.FlyDSLQuickAllReduce.assert_called_once_with(
        qr.group, qr.device, quantization="INT4"
    )
    module.FlyDSLAllReduceRMSNorm.assert_called_once_with(
        qr.group, qr.device, quantized=True, hiddens=()
    )


def test_init_flydsl_keeps_the_engine_aiter_enabled(monkeypatch):
    module, ar, rms = _fake_backends(ar_disabled=True)
    qr, took_over = _init_flydsl(monkeypatch, module)

    assert took_over
    assert qr._fly_ar is None and qr._fly_rms is rms
    ar.close.assert_called_once()


def test_init_flydsl_falls_back_when_aiter_declines(monkeypatch):
    module, ar, rms = _fake_backends(ar_disabled=True, rms_disabled=True)
    qr, took_over = _init_flydsl(monkeypatch, module)

    assert not took_over
    assert qr.disabled
    assert not qr.uses_flydsl
    ar.close.assert_called_once()
    rms.close.assert_called_once()


@pytest.mark.parametrize("quantization", ["FP", "INT8", "INT6"])
def test_init_flydsl_needs_int4(monkeypatch, quantization):
    module, _, _ = _fake_backends()
    qr, took_over = _init_flydsl(monkeypatch, module, quantization)

    assert not took_over
    module.FlyDSLQuickAllReduce.assert_not_called()
    module.FlyDSLAllReduceRMSNorm.assert_not_called()


def _patch_tp_qr_comm(qr_comm):
    device_comm = types.SimpleNamespace(qr_comm=qr_comm)
    tp_group = types.SimpleNamespace(device_communicator=device_comm)
    return patch("vllm.distributed.get_tp_group", return_value=tp_group)


def test_fused_group_quant_op_runs_flydsl_then_quant():
    from vllm import _aiter_ops

    qr_comm = MagicMock(uses_flydsl=True)
    qr_comm.should_fused_allreduce_rmsnorm.return_value = True
    qr_comm.fused_allreduce_rmsnorm.return_value = ("norm", "residual_out")
    inp, res, w = (MagicMock() for _ in range(3))

    with (
        _patch_tp_qr_comm(qr_comm),
        patch.object(
            _aiter_ops.rocm_aiter_ops, "get_aiter_allreduce", return_value=MagicMock()
        ),
        patch.object(
            _aiter_ops,
            "_rocm_aiter_group_fp8_quant_impl",
            return_value=("quant", "scale"),
        ) as quant,
    ):
        got = _aiter_ops._rocm_aiter_fused_allreduce_rmsnorm_quant_per_group_impl(
            inp, res, w, 1e-6, 128
        )
        got_bf16 = _aiter_ops._rocm_aiter_fused_allreduce_rmsnorm_quant_per_group_with_bf16_norm_impl(  # noqa: E501
            inp, res, w, 1e-6, 128
        )

    assert got == ("quant", "residual_out", "scale")
    assert got_bf16 == ("quant", "residual_out", "scale", "norm")
    quant.assert_called_with("norm", 128)


def test_fused_group_quant_needs_group_size_128():
    from vllm import _aiter_ops

    qr_comm = MagicMock(uses_flydsl=True)
    qr_comm.should_fused_allreduce_rmsnorm.return_value = True
    with _patch_tp_qr_comm(qr_comm):
        assert (
            _aiter_ops._flydsl_fused_allreduce_rmsnorm_group_quant(
                MagicMock(), MagicMock(), MagicMock(), 1e-6, 64
            )
            is None
        )
    qr_comm.fused_allreduce_rmsnorm.assert_not_called()


def test_fused_rmsnorm_op_routes_to_flydsl():
    from vllm import _aiter_ops

    qr_comm = MagicMock(uses_flydsl=True)
    qr_comm.should_fused_allreduce_rmsnorm.return_value = True
    qr_comm.fused_allreduce_rmsnorm.return_value = ("out", "residual_out")
    aiter_ar = MagicMock()
    inp, res, w = (MagicMock() for _ in range(3))

    with (
        _patch_tp_qr_comm(qr_comm),
        patch.object(
            _aiter_ops.rocm_aiter_ops, "get_aiter_allreduce", return_value=aiter_ar
        ),
    ):
        got = _aiter_ops._rocm_aiter_fused_allreduce_rmsnorm_impl(
            inp, res, w, 1e-6, False
        )

    assert got == ("out", "residual_out")
    qr_comm.should_fused_allreduce_rmsnorm.assert_called_once_with(inp, res, w, False)
    aiter_ar.aiter_ca.custom_fused_ar_rms.assert_not_called()
