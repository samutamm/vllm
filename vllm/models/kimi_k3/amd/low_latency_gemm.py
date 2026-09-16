# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 decode GEMM selection for unquantized BF16 on AMD ROCm gfx950"""

from typing import Literal

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.utils import wvsplitkrc_dispatch
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms.rocm import on_gfx950
from vllm.utils.platform_utils import num_compute_units

Backend = Literal["wvsplitk", "wvsplitkrc"]


def _build_plan(n: int, k: int, cu_count: int) -> dict[int, Backend]:
    """Precompute the num_tokens -> backend plan for a linear of shape
    (n, k), given the process-wide compute unit count.
    """
    plan: dict[int, Backend] = {}
    for num_tokens in range(1, 129):
        if hasattr(ops, "wvSplitK") and (
            1 <= num_tokens <= 4 or (num_tokens == 5 and n < 8448)
        ):
            plan[num_tokens] = "wvsplitk"
        elif (
            hasattr(ops, "wvSplitKrc")
            and 10 <= num_tokens <= 128
            and k > 768
            and n % 16 == 0
        ):
            _, fits = wvsplitkrc_dispatch(num_tokens, k, n, cu_count)
            if fits:
                plan[num_tokens] = "wvsplitkrc"
    return plan


def _run_plan(
    plan: dict[int, Backend], x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    """Dispatch by the precomputed plan, or return None to fall back."""
    backend = plan.get(x.shape[0])
    if backend is None:
        return None
    cu_count = num_compute_units()
    output_size = weight.shape[0]
    x_view = x.reshape(-1, x.shape[-1]).contiguous()
    if backend == "wvsplitk":
        output = ops.wvSplitK(weight, x_view, cu_count, None)
    else:
        output = ops.wvSplitKrc(x_view, weight, cu_count, None)
    return output.reshape(*x.shape[:-1], output_size)


class _KimiK3LowLatencyApply:
    """Mixin: try the precomputed plan, else defer to the base method."""

    def __init__(self, plan: dict[int, Backend]) -> None:
        super().__init__()
        self._plan = plan

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            output = _run_plan(self._plan, x, layer.weight)
            if output is not None:
                return output
        return super().apply(layer, x, bias)  # type: ignore[misc]


class KimiK3LowLatencyLinearMethod(_KimiK3LowLatencyApply, UnquantizedLinearMethod):
    pass


class KimiK3LowLatencyEmbeddingMethod(
    _KimiK3LowLatencyApply, UnquantizedEmbeddingMethod
):
    pass


def enable_kimi_k3_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    """Install shape-selected low-latency GEMMs for gfx950 unquantized BF16.

    The plan is a pure function of the local ``(N, K)`` shape and the
    process-wide compute unit count, so no measured per-shape table is
    needed.
    """
    if dtype != torch.bfloat16 or not on_gfx950():
        return
    if envs.VLLM_DISABLE_KIMI_K3_LOW_LATENCY_GEMM:
        return

    cu_count = num_compute_units()
    for child in module.modules():
        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        # ParallelLMHead is a VocabParallelEmbedding subclass; embed_tokens is
        # the parent type, so isinstance already excludes it.
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        plan = _build_plan(weight.shape[0], weight.shape[1], cu_count)
        if not plan:
            continue
        if is_linear:
            child.quant_method = KimiK3LowLatencyLinearMethod(plan)
        else:
            child.quant_method = KimiK3LowLatencyEmbeddingMethod(plan)


def try_low_latency_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Run the first supported ROCm skinny-GEMM case, or return None."""

    if (
        envs.VLLM_BATCH_INVARIANT
        or residual is not None
        or not on_gfx950()
        or x.dim() != 2
        or weight.dim() != 2
        or x.dtype not in (torch.float16, torch.bfloat16)
        or weight.dtype != x.dtype
        or not x.is_cuda
        or not weight.is_cuda
        or x.device != weight.device
        or x.shape[1] != weight.shape[1]
        or weight.shape[1] % 8 != 0
        or not weight.is_contiguous()
    ):
        return None

    num_tokens = x.shape[0]
    output_size = weight.shape[0]
    k = weight.shape[1]
    cu_count = num_compute_units()
    x_view = x.reshape(-1, x.shape[-1]).contiguous()

    # wvSplitK wins for very small token counts. It regresses badly at
    # num_tokens == 5 once output_size grows large (measured on MI350X).
    if hasattr(ops, "wvSplitK") and (
        1 <= num_tokens <= 4 or (num_tokens == 5 and output_size < 8448)
    ):
        output = ops.wvSplitK(weight, x_view, cu_count, None)
        return output.reshape(*x.shape[:-1], output_size)

    # wvSplitKrc's kernel only supports next-power-of-2(num_tokens) in
    # {16,32,64,128}, i.e. num_tokens in [9,128]; mirror
    # rocm_unquantized_gemm_impl()'s gfx950 dispatch bound of 10 to stay clear
    # of the boundary. k > 768 excludes a measured regression at k == 768
    # (0.89-0.90x on MI350X) where the kernel's overhead isn't worth it.
    if (
        hasattr(ops, "wvSplitKrc")
        and 10 <= num_tokens <= 128
        and k > 768
        and output_size % 16 == 0
    ):
        _, fits = wvsplitkrc_dispatch(num_tokens, k, output_size, cu_count)
        if fits:
            output = ops.wvSplitKrc(x_view, weight, cu_count, None)
            return output.reshape(*x.shape[:-1], output_size)

    return None
