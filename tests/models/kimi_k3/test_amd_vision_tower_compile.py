# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.rotary_embedding.packed_qk_rope import (
    packed_qk_rope_,
)
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    KimiVisionPatchMergerProjector,
    MoonVision3dPatchEmbed,
    MoonViTEncoderLayer,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_k3 import KimiK3VisionConfig

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 Vision Tower compile tests require ROCm",
)


@pytest.fixture(scope="module")
def vllm_compile_config():
    """VllmConfig with compile_mm_encoder enabled and initialized dist env."""
    cfg = VllmConfig(compilation_config=CompilationConfig(compile_mm_encoder=True))
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    with set_current_vllm_config(cfg):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        yield cfg
        destroy_model_parallel()
        destroy_distributed_environment()
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def _init_weights(module: nn.Module):
    """Initialize empty parameter tensors to avoid NaN in random tests."""
    for p in module.parameters():
        if p.data.numel() > 0:
            if p.data.ndim >= 2:
                p.data.normal_(0.0, 0.02)
            else:
                p.data.fill_(0.01)


def test_packed_qk_rope_custom_op_compile():
    """Verify packed_qk_rope custom op compiles cleanly without Dynamo breaks."""
    seqlen = 64
    nheads = 16
    headdim = 64
    rotary_dim = headdim

    xqkv_eager = torch.randn(
        seqlen, 3, nheads, headdim, device="cuda", dtype=torch.bfloat16
    )
    xqkv_comp = xqkv_eager.clone()

    angles = torch.randn(seqlen, rotary_dim // 2, device="cuda", dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles).contiguous()

    # Eager execution
    packed_qk_rope_(xqkv_eager, freqs)

    # Compiled execution
    @torch.compile(dynamic=True)
    def fn(x, f):
        packed_qk_rope_(x, f)
        return x

    res_comp = fn(xqkv_comp, freqs)
    torch.testing.assert_close(res_comp, xqkv_eager, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("seqlen", [128, 256])
def test_moonvit_encoder_layer_compile_parity(vllm_compile_config, seqlen: int):
    """Verify MoonViTEncoderLayer forward pass compiles and matches eager."""
    hidden_dim = 256
    num_heads = 4
    mlp_dim = 512
    device = torch.device("cuda")

    torch.manual_seed(42)
    layer = (
        MoonViTEncoderLayer(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            prefix="test_encoder_layer",
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    _init_weights(layer)

    x = torch.randn(seqlen, hidden_dim, device=device, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
    head_dim = hidden_dim // num_heads
    angles = torch.randn(seqlen, head_dim // 2, device=device, dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles).contiguous()
    max_seqlen = torch.tensor(seqlen, device=device, dtype=torch.int32)

    with torch.inference_mode():
        eager_out = layer(x, cu_seqlens, freqs, max_seqlen=max_seqlen)

    compiled_layer = torch.compile(layer, mode="default", dynamic=True)
    with torch.inference_mode():
        comp_out = compiled_layer(x, cu_seqlens, freqs, max_seqlen=max_seqlen)

    cos_sim = F.cosine_similarity(
        eager_out.float().flatten(1), comp_out.float().flatten(1), dim=-1
    ).mean()
    assert cos_sim.item() > 0.999, f"Cosine similarity {cos_sim.item()} <= 0.999"


@pytest.mark.parametrize("num_tokens", [64, 128])
def test_kimik25_projector_compile_parity(vllm_compile_config, num_tokens: int):
    """Verify KimiK25MultiModalProjector compiles and matches eager."""
    v_cfg = KimiK3VisionConfig()
    out_dim = 512
    device = torch.device("cuda")

    torch.manual_seed(42)
    proj = (
        KimiK25MultiModalProjector(
            config=v_cfg,
            out_hidden_size=out_dim,
            use_data_parallel=False,
            prefix="test_proj",
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    _init_weights(proj)

    proj_in_dim = 4 * v_cfg.hidden_size
    features = torch.randn(num_tokens, proj_in_dim, device=device, dtype=torch.bfloat16)

    with torch.inference_mode():
        eager_out = proj(features)

    compiled_proj = torch.compile(proj, mode="default", dynamic=True)
    with torch.inference_mode():
        comp_out = compiled_proj(features)

    torch.testing.assert_close(comp_out, eager_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("num_patches", [144, 288])
def test_moonvision3d_patchembed_compile_parity(vllm_compile_config, num_patches: int):
    """Verify MoonVision3dPatchEmbed compiles into 1 graph without breaks."""
    v_cfg = KimiK3VisionConfig()
    device = torch.device("cuda")

    torch.manual_seed(42)
    pe = (
        MoonVision3dPatchEmbed(
            out_dim=v_cfg.hidden_size,
            patch_size=v_cfg.patch_size,
            pos_emb_height=14,
            pos_emb_width=14,
            pos_emb_time=4,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    _init_weights(pe)

    x = torch.randn(
        num_patches,
        3,
        v_cfg.patch_size,
        v_cfg.patch_size,
        device=device,
        dtype=torch.bfloat16,
    )
    pos_embeds = torch.randn(
        num_patches, v_cfg.hidden_size, device=device, dtype=torch.bfloat16
    )

    with torch.inference_mode():
        eager_out = pe(x, None, pos_embeds=pos_embeds)

    compiled_pe = torch.compile(pe, mode="default", dynamic=True)
    with torch.inference_mode():
        comp_out = compiled_pe(x, None, pos_embeds=pos_embeds)

    torch.testing.assert_close(comp_out, eager_out, atol=1e-3, rtol=1e-3)

    # Verify 0 TorchDynamo graph breaks
    explanation = torch._dynamo.explain(pe)(x, None, pos_embeds=pos_embeds)
    assert explanation.graph_break_count == 0, (
        f"Graph breaks found: {explanation.break_reasons}"
    )
    assert len(explanation.graphs) == 1, (
        f"Expected 1 graph, got {len(explanation.graphs)}"
    )


@pytest.mark.parametrize("num_tokens", [36, 72])
def test_kimi_vision_patch_merger_projector_compile_parity(
    vllm_compile_config, num_tokens: int
):
    """Verify KimiVisionPatchMergerProjector compiles into 1 graph."""
    v_cfg = KimiK3VisionConfig()
    out_dim = 512
    device = torch.device("cuda")

    torch.manual_seed(42)
    proj = (
        KimiK25MultiModalProjector(
            config=v_cfg,
            out_hidden_size=out_dim,
            use_data_parallel=False,
            prefix="test_proj_merger",
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    _init_weights(proj)

    merger_and_proj = (
        KimiVisionPatchMergerProjector(proj)
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )

    num_patches = num_tokens * 4
    hidden_states = torch.randn(
        num_patches, v_cfg.hidden_size, device=device, dtype=torch.bfloat16
    )
    merge_gather_idx = torch.randint(
        0, num_patches, (num_tokens, 4), device=device, dtype=torch.int64
    )

    with torch.inference_mode():
        eager_out = merger_and_proj(hidden_states, merge_gather_idx)

    compiled_merger_proj = torch.compile(merger_and_proj, mode="default", dynamic=True)
    with torch.inference_mode():
        comp_out = compiled_merger_proj(hidden_states, merge_gather_idx)

    torch.testing.assert_close(comp_out, eager_out, atol=1e-3, rtol=1e-3)

    # Verify 0 TorchDynamo graph breaks
    explanation = torch._dynamo.explain(merger_and_proj)(
        hidden_states, merge_gather_idx
    )
    assert explanation.graph_break_count == 0, (
        f"Graph breaks found: {explanation.break_reasons}"
    )
    assert len(explanation.graphs) == 1, (
        f"Expected 1 graph, got {len(explanation.graphs)}"
    )


def test_kimi_k3_amd_supports_encoder_cudagraph():
    """Verify KimiK3 on AMD implements SupportsEncoderCudaGraph."""
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
    from vllm.models.kimi_k3.amd.model import KimiK3ForConditionalGeneration

    assert SupportsEncoderCudaGraph in KimiK3ForConditionalGeneration.__bases__
    assert supports_encoder_cudagraph(KimiK3ForConditionalGeneration)
    assert (
        getattr(KimiK3ForConditionalGeneration, "supports_encoder_cudagraph", False)
        is True
    )
