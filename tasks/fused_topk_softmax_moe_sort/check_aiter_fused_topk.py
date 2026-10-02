#!/usr/bin/env python3
"""Offline/one-time check: does the installed aiter actually have the fused
top-k gating + MoE sort split functions this task depends on?

Run this inside whatever environment (bare metal or docker image) is about
to be used for unit tests or e2e benchmarking of
AITER_USE_FUSED_TOPK_MOE_SORT, *before* trusting any results from it. A
docker image tag claiming a specific aiter commit (e.g.
"..._aiter_2ed2f032a") is not proof the build actually picked up that
source -- this script checks the installed package directly instead of
trusting the tag.

Checks, in order:
  1. aiter is importable at all.
  2. aiter.ops.flydsl.moe_sorting.fused_topk_gating exists (the gating-only
     split function vLLM's router patch calls).
  3. Its signature matches what vLLM's patch assumes: positional/keyword
     params (gating_output, bias, topk, scoring_func, need_renorm) and,
     importantly, NO routed_scaling_factor parameter (vLLM applies that
     scaling itself after the call -- if aiter ever adds it, the call
     signature in fused_topk_bias_router.py needs revisiting).
  4. aiter.ops.flydsl.moe_sorting.flydsl_fused_topk_moe_sort exists (the
     sort-only split function, wired into aiter's own moe_sorting() --
     not called from vLLM directly, but its absence means the sort-side
     fusion isn't actually active even if gating is).
  5. The env var gate flag aiter reads, _USE_FUSED_TOPK_MOE_SORT, is parsed
     from AITER_USE_FUSED_TOPK_MOE_SORT as expected.
  6. aiter/fused_moe.py::moe_sorting references the fused dispatch (i.e.
     the sort-side wiring in step 4 is actually reachable from the real
     dispatch call site, not just defined standalone).

Exits 0 if everything is present and shaped as expected, 1 otherwise, with
a clear PASS/FAIL line per check so a CI/bench harness can gate on it.
"""

import inspect
import sys

CHECKS_PASSED: list[str] = []
CHECKS_FAILED: list[str] = []


def _ok(msg: str) -> None:
    CHECKS_PASSED.append(msg)
    print(f"[PASS] {msg}")


def _fail(msg: str) -> None:
    CHECKS_FAILED.append(msg)
    print(f"[FAIL] {msg}")


def main() -> int:
    try:
        import aiter

        _ok(f"aiter importable (file: {aiter.__file__})")
    except Exception as e:
        _fail(f"aiter not importable: {e!r}")
        _summarize()
        return 1

    try:
        import aiter.ops.flydsl.moe_sorting as moe_sorting_mod

        _ok(f"aiter.ops.flydsl.moe_sorting importable (file: {moe_sorting_mod.__file__})")
    except Exception as e:
        _fail(f"aiter.ops.flydsl.moe_sorting not importable: {e!r}")
        _summarize()
        return 1

    fused_topk_gating = getattr(moe_sorting_mod, "fused_topk_gating", None)
    if fused_topk_gating is None:
        _fail(
            "aiter.ops.flydsl.moe_sorting.fused_topk_gating is MISSING -- "
            "vLLM's fused_topk_bias fast path will silently no-op (falls "
            "back via its try/except) even with "
            "AITER_USE_FUSED_TOPK_MOE_SORT=1. This installed aiter does not "
            "have the split-gating commit, regardless of what the image "
            "tag claims."
        )
    else:
        _ok("aiter.ops.flydsl.moe_sorting.fused_topk_gating exists")
        sig = inspect.signature(fused_topk_gating)
        params = list(sig.parameters)
        expected_prefix = ["gating_output", "bias", "topk", "scoring_func", "need_renorm"]
        if params[: len(expected_prefix)] == expected_prefix:
            _ok(f"fused_topk_gating signature matches expected: {sig}")
        else:
            _fail(
                f"fused_topk_gating signature {sig} does not match expected "
                f"param order {expected_prefix} -- vLLM's call site in "
                "fused_topk_bias_router.py may break or silently mis-bind "
                "args."
            )
        if "routed_scaling_factor" in params:
            _fail(
                "fused_topk_gating now has a routed_scaling_factor param -- "
                "vLLM's router patch applies scaling manually after the "
                "call and does NOT pass this through; double-scaling risk. "
                "Update vllm/model_executor/layers/fused_moe/router/"
                "fused_topk_bias_router.py if this is intentional upstream."
            )
        else:
            _ok("fused_topk_gating has no routed_scaling_factor param (as expected)")

    flydsl_fused_topk_moe_sort = getattr(moe_sorting_mod, "flydsl_fused_topk_moe_sort", None)
    if flydsl_fused_topk_moe_sort is None:
        _fail(
            "aiter.ops.flydsl.moe_sorting.flydsl_fused_topk_moe_sort is "
            "MISSING -- the sort-side fusion (wired into aiter's own "
            "moe_sorting()) is not active. Gating alone may still work, but "
            "the full fused pipeline's perf benefit will not."
        )
    else:
        _ok("aiter.ops.flydsl.moe_sorting.flydsl_fused_topk_moe_sort exists")

    use_flag = getattr(moe_sorting_mod, "_USE_FUSED_TOPK_MOE_SORT", None)
    if use_flag is None:
        _fail(
            "aiter.ops.flydsl.moe_sorting._USE_FUSED_TOPK_MOE_SORT gate flag "
            "not found -- AITER_USE_FUSED_TOPK_MOE_SORT may not do anything "
            "in this build."
        )
    else:
        _ok(f"_USE_FUSED_TOPK_MOE_SORT gate flag present (current value: {use_flag})")

    try:
        import aiter.fused_moe as fused_moe_mod

        source = inspect.getsource(fused_moe_mod.moe_sorting)
        if "_flydsl_fused_topk_moe_sorting" in source or "fused_topk" in source:
            _ok(
                "aiter/fused_moe.py::moe_sorting references the fused-topk "
                "sort dispatch -- sort-side wiring is reachable from the "
                "real dispatch call site."
            )
        else:
            _fail(
                "aiter/fused_moe.py::moe_sorting has no reference to the "
                "fused-topk sort dispatch -- even if "
                "flydsl_fused_topk_moe_sort exists as a standalone function, "
                "nothing calls it from the real MoE forward path."
            )
    except Exception as e:
        _fail(f"Could not inspect aiter.fused_moe.moe_sorting: {e!r}")

    _summarize()
    return 0 if not CHECKS_FAILED else 1


def _summarize() -> None:
    print()
    print(f"=== {len(CHECKS_PASSED)} passed, {len(CHECKS_FAILED)} failed ===")
    if CHECKS_FAILED:
        print(
            "This aiter install is NOT ready for "
            "AITER_USE_FUSED_TOPK_MOE_SORT=1 e2e/unit testing. Do not trust "
            "benchmark numbers from this environment until these are fixed."
        )


if __name__ == "__main__":
    sys.exit(main())
