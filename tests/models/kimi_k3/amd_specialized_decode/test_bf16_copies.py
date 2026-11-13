# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's specialized decode on online-FP8 linears (CPU): the
`specialized_decode_bf16_copies` switch, the linears that keep a bf16 copy, the
copy itself, and the weight refusal with and without it."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.amd.specialized_decode.hooks import SpecializedDecodeMixin
from vllm.models.kimi_k3.amd.specialized_decode.runtime import (
    BF16_COPIES,
    FFN,
    WHOLE,
    DecodeConfig,
    bf16_copies,
    dense_weight,
    keep_bf16_copy,
    kernel_linears,
    specialized_decode_config,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime.guards import weight_reasons

from .test_runtime import (  # noqa: F401
    _k3_text_config,
    _vllm_config,
    fp4_bmm,
    on_mi355x,
)


def _online_method():
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PtpcOnlineLinearMethod,
    )

    method = object.__new__(Fp8PtpcOnlineLinearMethod)
    method.requantization_source = None
    return method


def _linear(**attrs) -> SimpleNamespace:
    return SimpleNamespace(
        weight=torch.empty(0), quant_method=_online_method(), **attrs
    )


def _wrapped(linear) -> bool:
    return "process_weights_after_loading" in vars(linear.quant_method)


def _layer() -> SimpleNamespace:
    return SimpleNamespace(
        self_attn=SimpleNamespace(
            o_proj=_linear(),
            in_proj_qkvgfab=_linear(),
            f_b_proj=_linear(),
        ),
        mlp=SimpleNamespace(
            routed_expert_down_proj=_linear(),
            routed_expert_up_proj=_linear(),
            shared_experts=SimpleNamespace(gate_up_proj=_linear(), down_proj=_linear()),
        ),
    )


@pytest.mark.parametrize(
    "extra, want",
    [
        (None, False),
        ({}, False),
        ({BF16_COPIES: False}, False),
        ({BF16_COPIES: True}, True),
    ],
)
def test_switch(extra, want):
    assert bf16_copies(SimpleNamespace(additional_config=extra)) is want


@pytest.mark.parametrize("value", [1, "true", None])
def test_switch_must_be_a_bool(value):
    with pytest.raises(ValueError, match=BF16_COPIES):
        bf16_copies(SimpleNamespace(additional_config={BF16_COPIES: value}))


def test_config_carries_the_switch(on_mi355x):  # noqa: F811
    config = _vllm_config(speculative_config=None)
    assert specialized_decode_config(config).bf16_copies is False
    config.additional_config[BF16_COPIES] = True
    assert specialized_decode_config(config).bf16_copies is True


def test_kernel_linears_are_the_ones_the_kernels_read():
    layer = _layer()
    whole, ffn = kernel_linears(layer, WHOLE), kernel_linears(layer, FFN)
    attn, mlp = layer.self_attn, layer.mlp
    assert [id(x) for x in ffn] == [
        id(x)
        for x in (
            attn.o_proj,
            mlp.routed_expert_down_proj,
            mlp.shared_experts.gate_up_proj,
            mlp.shared_experts.down_proj,
        )
    ]
    assert [id(x) for x in whole] == [
        id(x) for x in ffn + [attn.in_proj_qkvgfab, attn.f_b_proj]
    ]
    assert all(lin is not mlp.routed_expert_up_proj for lin in whole)


def test_dense_weight_prefers_the_copy():
    copy = torch.ones(2, 2, dtype=torch.bfloat16)
    assert dense_weight(_linear(unquantized_weight=copy)) is copy
    linear = _linear()
    assert dense_weight(linear) is linear.weight


def test_keep_kernel_weight_copies_marks_the_layers_in_this_pp_stage():
    layers = [None] + [_layer() for _ in range(3)]
    model = SimpleNamespace(
        config=_k3_text_config(), layers=layers, start_layer=2, end_layer=3
    )
    host = SpecializedDecodeMixin()
    host.model = model
    host.init_specialized_decode(DecodeConfig(q=1, bf16_copies=True))
    host.keep_kernel_weight_copies()
    assert not any(_wrapped(lin) for lin in kernel_linears(layers[1], WHOLE))
    assert not any(_wrapped(lin) for lin in kernel_linears(layers[3], WHOLE))
    assert all(_wrapped(lin) for lin in kernel_linears(layers[2], WHOLE))
    assert not _wrapped(layers[2].mlp.routed_expert_up_proj)


def test_keep_kernel_weight_copies_is_off_by_default():
    layers = [None, _layer()]
    host = SpecializedDecodeMixin()
    host.model = SimpleNamespace(
        config=_k3_text_config(), layers=layers, start_layer=0, end_layer=2
    )
    for config in (None, DecodeConfig(q=1)):
        host.init_specialized_decode(config)
        host.keep_kernel_weight_copies()
    assert not any(_wrapped(lin) for lin in kernel_linears(layers[1], WHOLE))


def test_other_linears_are_left_alone():
    linear = SimpleNamespace(weight=torch.empty(0), quant_method=object())
    keep_bf16_copy(linear)
    keep_bf16_copy(SimpleNamespace(weight=torch.empty(0)))
    assert not hasattr(linear.quant_method, "process_weights_after_loading")


def _quantize(layer) -> None:
    """Stands in for the method's own quantization: the weight becomes FP8."""
    layer.weight = torch.nn.Parameter(
        layer.weight.to(torch.float8_e4m3fn), requires_grad=False
    )
    layer._already_called_process_weights_after_loading = True


@pytest.mark.parametrize("kept", [True, False])
def test_online_fp8_keeps_the_loaded_bf16_weight(kept):
    weight = torch.randn(64, 32, dtype=torch.bfloat16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.quant_method = _online_method()
    layer.quant_method.process_weights_after_loading = _quantize
    if kept:
        keep_bf16_copy(layer)

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    if kept:
        assert layer.unquantized_weight.dtype == torch.bfloat16
        assert torch.equal(layer.unquantized_weight, weight)
        assert dense_weight(layer) is layer.unquantized_weight
    else:
        assert not hasattr(layer, "unquantized_weight")


def test_a_second_pass_keeps_the_first_copy():
    weight = torch.randn(8, 8, dtype=torch.bfloat16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.quant_method = _online_method()
    layer.quant_method.process_weights_after_loading = _quantize
    keep_bf16_copy(layer)
    layer.quant_method.process_weights_after_loading(layer)
    copy = layer.unquantized_weight
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.unquantized_weight is copy


def _kernel_layer(fp8_weights: bool, copies: bool) -> SimpleNamespace:
    """Layer 1 (a KDA layer) with every weight the kernels read at its kernel
    shape on the meta device; the guarded linears' weights are FP8 (as online
    quantization leaves them), with or without a bf16 copy."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import back, kda
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import moe

    bf, f32, H = torch.bfloat16, torch.float32, moe.HIDDEN
    wdtype = torch.float8_e4m3fn if fp8_weights else bf

    def t(*shape, dtype=bf):
        return torch.empty(*shape, dtype=dtype, device="meta")

    def lin(*shape):
        # FP8 weights are stored transposed; the kernels never read those
        weight = t(*shape[::-1], dtype=wdtype).t() if fp8_weights else t(*shape)
        attrs = {"unquantized_weight": t(*shape)} if fp8_weights and copies else {}
        return SimpleNamespace(weight=weight, **attrs)

    experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_BF16, is_k3_situ_aiter=True
        ),
        w13_weight=t(1),
        w13_weight_scale=t(1),
        w2_weight=t(1),
        w2_weight_scale=t(1),
    )
    mlp = SimpleNamespace(
        use_latent_moe=True,
        routed_scaling_factor=1.0,
        experts=experts,
        gate=SimpleNamespace(
            weight=t(moe.E, H), e_score_correction_bias=t(moe.E, dtype=f32)
        ),
        routed_expert_down_proj=lin(moe.LAT, H),
        routed_expert_norm=SimpleNamespace(weight=t(moe.LAT)),
        shared_experts=SimpleNamespace(
            gate_up_proj=lin(2 * moe.SI, H), down_proj=lin(H, moe.SI)
        ),
        routed_output_transform=SimpleNamespace(row_sharded=False),
        routed_expert_up_proj=SimpleNamespace(weight=t(H, moe.LAT)),
    )
    attn = SimpleNamespace(
        local_num_heads=kda.NH,
        head_dim=kda.HD,
        conv_size=4,
        o_proj=lin(H, back.OK),
        in_proj_qkvgfab=lin(kda.NPROJ, H),
        f_b_proj=lin(kda.PROJ, kda.HD),
        o_norm=SimpleNamespace(weight=t(kda.HD)),
        conv1d=SimpleNamespace(weight=t(kda.QKV, 1, 4, dtype=f32)),
        A_log=t(kda.NH, dtype=f32),
        dt_bias=t(kda.PROJ, dtype=f32),
    )
    norm = lambda: SimpleNamespace(weight=t(H))  # noqa: E731
    proj = lambda: SimpleNamespace(weight=t(1, H))  # noqa: E731
    return SimpleNamespace(
        mlp=mlp,
        self_attn=attn,
        mlp_res_norm=norm(),
        mlp_res_proj=proj(),
        post_attention_layernorm=norm(),
        input_layernorm=norm(),
        self_attention_res_norm=norm(),
        self_attention_res_proj=proj(),
    )


def _reasons(layer) -> list[str]:
    model = SimpleNamespace(
        config=_k3_text_config(), layers=[None, layer], start_layer=1, end_layer=2
    )
    return weight_reasons(model)


def test_bf16_weights_are_accepted():
    pytest.importorskip("flydsl")
    assert _reasons(_kernel_layer(fp8_weights=False, copies=False)) == []


def test_fp8_weights_without_copies_are_refused_with_the_switch_named():
    pytest.importorskip("flydsl")
    reasons = _reasons(_kernel_layer(fp8_weights=True, copies=False))
    for name in ("latent down", "shared gate_up", "shared down", "o_proj", "in_proj"):
        assert any(r.startswith(f"layer 1 {name}: ") for r in reasons), name
    assert any(r.startswith("layer 1 f_b_proj: ") for r in reasons)
    assert any(BF16_COPIES in r for r in reasons)


def test_fp8_weights_with_copies_are_accepted():
    pytest.importorskip("flydsl")
    assert _reasons(_kernel_layer(fp8_weights=True, copies=True)) == []


def test_a_copy_of_the_wrong_shape_is_refused():
    pytest.importorskip("flydsl")
    layer = _kernel_layer(fp8_weights=True, copies=True)
    layer.self_attn.o_proj.unquantized_weight = torch.empty(3, 3, dtype=torch.bfloat16)
    assert [r for r in _reasons(layer) if "o_proj" in r]
