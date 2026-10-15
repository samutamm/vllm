#!/usr/bin/env python3
"""Offline/one-time check: does AITER_USE_FUSED_TOPK_MOE_SORT=1 vs 0 actually
switch which kernel executes inside fused_topk_bias, not just produce the
same numeric result through some other path?

Wraps the real aiter.ops.flydsl.moe_sorting.fused_topk_gating and vLLM's own
vllm_topk_softmax (the CUDA/ROCm custom-op wrapper used by the baseline
softmax path) with call-count spies, and asserts that exactly one of them
fires per env var setting:
  - AITER_USE_FUSED_TOPK_MOE_SORT=1 -> fused_topk_gating called,
    vllm_topk_softmax NOT called.
  - AITER_USE_FUSED_TOPK_MOE_SORT=0 (or unset) -> vllm_topk_softmax called,
    fused_topk_gating NOT called.

Run inside the target docker image, with GPU device access, e.g.:
  docker run --rm --entrypoint '' --device /dev/dri --device /dev/kfd \
    --security-opt seccomp=unconfined --group-add video \
    <image> python3 /path/to/this/script.py
"""

import os
import sys
from unittest import mock

import torch

NUM_EXPERTS = 896
TOPK = 16
MODEL_DIM = 16

CHECKS_PASSED: list[str] = []
CHECKS_FAILED: list[str] = []


def _ok(msg: str) -> None:
    CHECKS_PASSED.append(msg)
    print(f"[PASS] {msg}")


def _fail(msg: str) -> None:
    CHECKS_FAILED.append(msg)
    print(f"[FAIL] {msg}")


def main() -> int:
    if not torch.cuda.is_available():
        _fail("No CUDA/ROCm device visible -- cannot run this check.")
        _summarize()
        return 1

    import aiter.ops.flydsl.moe_sorting as moe_sorting_mod
    import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as router_mod

    real_fused_topk_gating = moe_sorting_mod.fused_topk_gating
    real_vllm_topk_softmax = router_mod.vllm_topk_softmax

    device = "cuda"
    torch.manual_seed(0)
    hidden_states = torch.randn(4, MODEL_DIM, device=device)
    gating_output = torch.randn(4, NUM_EXPERTS, device=device)

    def _run(env_value: str | None):
        if env_value is None:
            os.environ.pop("AITER_USE_FUSED_TOPK_MOE_SORT", None)
        else:
            os.environ["AITER_USE_FUSED_TOPK_MOE_SORT"] = env_value

        gating_spy = mock.Mock(wraps=real_fused_topk_gating)
        softmax_spy = mock.Mock(wraps=real_vllm_topk_softmax)
        with (
            mock.patch.object(moe_sorting_mod, "fused_topk_gating", gating_spy),
            mock.patch.object(router_mod, "vllm_topk_softmax", softmax_spy),
        ):
            router_mod.fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=gating_output,
                scoring_func="softmax",
                e_score_correction_bias=None,
                topk=TOPK,
                renormalize=True,
            )
        return gating_spy.call_count, softmax_spy.call_count

    gating_calls_on, softmax_calls_on = _run("1")
    if gating_calls_on == 1 and softmax_calls_on == 0:
        _ok(
            "AITER_USE_FUSED_TOPK_MOE_SORT=1 -> fused_topk_gating executed "
            "(1 call), vllm_topk_softmax NOT executed (0 calls)"
        )
    else:
        _fail(
            "AITER_USE_FUSED_TOPK_MOE_SORT=1 did not switch kernels as "
            f"expected: fused_topk_gating calls={gating_calls_on}, "
            f"vllm_topk_softmax calls={softmax_calls_on}"
        )

    gating_calls_off, softmax_calls_off = _run("0")
    if gating_calls_off == 0 and softmax_calls_off == 1:
        _ok(
            "AITER_USE_FUSED_TOPK_MOE_SORT=0 -> vllm_topk_softmax executed "
            "(1 call), fused_topk_gating NOT executed (0 calls)"
        )
    else:
        _fail(
            "AITER_USE_FUSED_TOPK_MOE_SORT=0 did not switch kernels as "
            f"expected: fused_topk_gating calls={gating_calls_off}, "
            f"vllm_topk_softmax calls={softmax_calls_off}"
        )

    gating_calls_unset, softmax_calls_unset = _run(None)
    if gating_calls_unset == 0 and softmax_calls_unset == 1:
        _ok(
            "AITER_USE_FUSED_TOPK_MOE_SORT unset -> vllm_topk_softmax "
            "executed (1 call), fused_topk_gating NOT executed (0 calls)"
        )
    else:
        _fail(
            "AITER_USE_FUSED_TOPK_MOE_SORT unset did not fall back as "
            f"expected: fused_topk_gating calls={gating_calls_unset}, "
            f"vllm_topk_softmax calls={softmax_calls_unset}"
        )

    _summarize()
    return 0 if not CHECKS_FAILED else 1


def _summarize() -> None:
    print()
    print(f"=== {len(CHECKS_PASSED)} passed, {len(CHECKS_FAILED)} failed ===")


if __name__ == "__main__":
    sys.exit(main())
