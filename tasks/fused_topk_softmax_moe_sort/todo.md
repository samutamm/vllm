# topk_gating_softmax_moe_sort_fusion

## Details
This is an implementation task.
- branch: topk_gating_softmax_moe_sort_fusion
- testing node: gbt350-odcdh2-c05-1.png-odc.dcgpu. 
  - there are other users of the node, check first with rocm-smi before running. Ask me to verify if another run is about to start.
- testing docker image:
  - vllm/vllm-openai-rocm:nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516
- example docker run command:
  - workdir/low_latency_gemm/start_server_low_latency_gemm_e2e.sh
  - for iterative testing, feel free to modify files inside docker file on the node

## Implementation guide — vLLM side

Wire a single env-gated interception point into vLLM's router, right before it calls the baseline softmax/sigmoid+topk path — no changes to the MoE GEMM/expert-dispatch logic itself are needed.

1. Patch target: vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py, inside fused_topk_bias(...) — insert the guard immediately after n_routed_experts = gating_output.shape[-1] is computed, before the existing softmax/sigmoid scoring block.
2. Guard condition: only take the fast path when scoring_func == "softmax", AITER_USE_FUSED_TOPK_MOE_SORT is truthy, hidden_states.size(0) <= 16 (decode), n_routed_experts == 896, topk == 16, and there's no vision bias (bias_vl is None) or hash-indices table (hash_indices_table is None) — these are the Kimi-K3-specific, non-fast-path-eligible router variants.
3. Call out to AITER: from aiter.ops.flydsl.fused_topk_moe_sorting import fused_topk_gating, invoke with (gating_output, e_score_correction_bias, topk=topk, scoring_func="softmax", need_renorm=renormalize, routed_scaling_factor=routed_scaling_factor), cast topk_indices to indices_type or torch.int32, and return (topk_weights, topk_indices) directly — bypassing the rest of the function.
4. Wrap the import/call in try/except Exception: pass so any unexpected AITER-side failure (missing FlyDSL, shape mismatch) falls through to the existing baseline code unmodified below.
5. Sorting side is untouched directly — it's AITER's moe_sorting/flydsl_moe_sorting_fwd call inside fused_flydsl_moe.py that gets intercepted on the AITER side (see AITER guide); vLLM only needs the router-side patch above since the sorter is invoked via the AITER wrapper, not reimplemented in vLLM.
6. No new vLLM config/flags — reuse the existing AITER_USE_FUSED_TOPK_MOE_SORT env var for consistency with the AITER-side gate; don't introduce a parallel vLLM-only flag.

Validate with the integration test (test_framework_dispatch_fused_topk_bias) across M ∈ {1,2,4,8,16} and confirm fallback paths trigger correctly for M=32 (prefill), E=256 (wrong model), and gate-disabled cases.

The interface

vLLM → AITER call (router side):
fused_topk_gating(
    gating_output: torch.Tensor,        # [M, 896] float32 router logits
    bias: Optional[torch.Tensor],       # [896] float32 e_score_correction_bias, or None
    topk: int = 16,
    scoring_func: str = "softmax",
    need_renorm: bool = True,
    routed_scaling_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]   # (topk_weights [M,16] f32, topk_indices [M,16] int32)

AITER sorter call (invoked from fused_flydsl_moe.py's moe_sorting, AITER-internal):
flydsl_compact_moe_sorting(
    topk_ids: torch.Tensor,             # [M, 16] int32
    topk_weights: torch.Tensor,         # [M, 16] float32
    num_experts: int = 896,
    block_size: int = 32,
    moebuf_dtype: torch.dtype = torch.bfloat16,
    model_dim: int = 7168,
    accumulate: bool = True,
    output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    # (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids[2], moe_buf)

Combined end-to-end entrypoint (used when both stages dispatch together, e.g. for custom call sites not going through vLLM's two-step router/sorter split):
fused_topk_softmax_moe_sort(
    gating_logits: torch.Tensor,        # [M, 896] float32
    bias: Optional[torch.Tensor] = None,
    topk: int = 16, unit_size: int = 32,
    scoring_func: str = "softmax", need_renorm: bool = True,
    sorted_ids/sorted_weights/sorted_expert_ids/num_valid_ids/moe_buf: Optional[torch.Tensor] = None,  # pre-allocation hooks
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

Gate: AITER_USE_FUSED_TOPK_MOE_SORT ∈ {"1","true","yes"} (default off) — checked independently on both the vLLM router patch and the AITER aiter/fused_moe.py::moe_sorting patch, each with its own try/except fallback to the respective baseline (fused_topk_bias softmax/sigmoid path in vLLM; topk_gating+flydsl_moe_sorting_fwd in AITER).

See the reference implementation in /home/stammine/repos/brains/Arete/export/topk_gating_softmax_moe_sort_fusion/ and use that more authorative, than the high level summary above.

## Root cause of the "no speedup" e2e result (2026-10-02) — FIXED on aiter side

The A/B sweep in
`silo-tiger-oob-benchmark-configs/runs/mi355x/kimi_k3/sweep/2026-10-01_1300/summary.md`
showed only noise-level deltas (~0.5-1%, inconsistent sign), not the Arete
campaign's +1.6%. Cause: Kimi-K3's router always passes
`e_score_correction_bias` as `bias` into `fused_topk_gating` (this file's
step 3). AITER's `fused_topk_gating` gated its fast path with
`not has_bias`, so every real call fell straight back to the sequential
`topk_gating` HIP path — both arms of the sweep ran identical code, hence
noise.

Fixed in aiter (`aiter/ops/flydsl/moe_sorting.py`,
see `tasks/topk_gating_softmax_moe_sort_fusion/todo.md` in the aiter repo for
details): added a bias-aware fast path
(`_fused_topk_gating_biased`) matching `topk_gating_kernels.cuh`'s HIP bias
semantics (bias shifts selection, not the reported weight). The `has_bias`
exclusion is gone from `use_fast_path`. No vLLM-side change needed — this
file's router patch already passes `e_score_correction_bias` through
unconditionally; it will now actually reach the fast path on the real model.

Next step: rebuild the test image with the updated aiter commit and re-run
the A/B e2e sweep to get a real (non-noise) measurement before concluding
anything about Arete's reported 1.6%.