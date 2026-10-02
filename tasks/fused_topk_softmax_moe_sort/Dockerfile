# Overlays this branch's vLLM changes (commit d0bc7710d4, "Add AITER fused
# top-k gating fast path for decode", now also including the one-line fix
# for the KDA chunk_intra.py autotune `key` bug -- see
# runs/mi355x/kimi_k3/sweep/2026-10-01_1300/bug_report_kda_chunk_intra_autotune_key.md
# in silo-tiger-oob-benchmark-configs) onto an image that already has aiter's
# topk_gating_softmax_moe_sort_fusion branch installed -- see
# aiter/tasks/topk_gating_softmax_moe_sort_fusion/Dockerfile for the
# aiter-side half of this.
#
# Build (from the vLLM repo root, so the COPY paths below resolve):
#   docker build -f tasks/fused_topk_softmax_moe_sort/Dockerfile \
#     --build-arg BASE_IMAGE=vllm/vllm-openai-rocm:nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516_aiter_2ed2f032a \
#     -t vllm/vllm-openai-rocm:nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516_aiter_2ed2f032a_vllm_d0bc7710d4 \
#     .
#
# BASE_IMAGE must already have aiter's split fused_topk_gating /
# flydsl_fused_topk_moe_sort installed (verify with
# tasks/fused_topk_softmax_moe_sort/check_aiter_fused_topk.py before trusting
# any tag). The COPY below only touches vLLM files, so rebuilding after an
# aiter-side change reuses the aiter install layer unchanged, and vice versa.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# vLLM's installed location in this base image (see check in
# check_aiter_fused_topk.py's companion investigation): a regular
# (non-editable) pip install under dist-packages, not a git checkout.
ARG VLLM_SITE_PACKAGES=/usr/local/lib/python3.12/dist-packages/vllm

# Router patch: AITER_USE_FUSED_TOPK_MOE_SORT fast path in fused_topk_bias.
COPY vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py \
    ${VLLM_SITE_PACKAGES}/model_executor/layers/fused_moe/router/fused_topk_bias_router.py

# Unit tests for the above, placed alongside the source (not inside
# dist-packages/vllm, which isn't a test tree) so they can be run against
# the installed package with:
#   python3 -m pytest /workspace/vllm_tests/test_fused_topk_bias_router.py \
#     --confcutdir=/workspace/vllm_tests -p no:cacheprovider
COPY tests/model_executor/layers/test_fused_topk_bias_router.py \
    /workspace/vllm_tests/test_fused_topk_bias_router.py

# Unrelated bugfix picked up on this branch: a stale Triton autotune `key`
# entry ("NC", a call-site-only local var, not a kernel arg) crashes Kimi-K3
# model import before AITER_USE_FUSED_TOPK_MOE_SORT is ever read. See
# bug_report_kda_chunk_intra_autotune_key.md.
COPY vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py \
    ${VLLM_SITE_PACKAGES}/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py
