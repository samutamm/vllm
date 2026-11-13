# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's hooks on `KimiLinearForCausalLM`, kept here so the
model only calls them: `init_specialized_decode` in `__init__` (after the config
check, which runs before any module allocates weights), `plan_specialized_decode`
in `forward` and `after_specialized_decode_step` in `compute_logits`. The
model state (`model_state.KimiK3ModelState`) builds the executor after load."""

from typing import TYPE_CHECKING

import torch

from vllm.models.kimi_k3.amd.specialized_decode.runtime import (
    DecodeConfig,
    keep_bf16_copy,
    kernel_linears,
    layer_kinds,
)

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.decode import (
        SpecializedDecode,
        Step,
    )
    from vllm.v1.worker.gpu.model_states.interface import ModelState


class SpecializedDecodeMixin:
    """`self.model` is the `KimiLinearModel` the executor runs over."""

    model: torch.nn.Module
    specialized_decode_config: DecodeConfig | None
    specialized_decode: "SpecializedDecode | None"

    def init_specialized_decode(self, config: DecodeConfig | None) -> None:
        """`config`: `specialized_decode_config(vllm_config)`, None for the
        stock decode."""
        self.specialized_decode_config = config
        self.specialized_decode = None

    def keep_kernel_weight_copies(self) -> None:
        """Called once `self.model` exists: the online-FP8 linears the layer
        kernels read keep a bf16 copy of their loaded weight
        (`DecodeConfig.bf16_copies`; `runtime.dense_weight` reads it)."""
        config = self.specialized_decode_config
        if config is None or not config.bf16_copies:
            return
        model = self.model
        for i, kind in layer_kinds(model.config).items():
            if model.start_layer <= i < model.end_layer:
                for linear in kernel_linears(model.layers[i], kind):
                    keep_bf16_copy(linear)

    def build_specialized_decode(self) -> None:
        from vllm.models.kimi_k3.amd.specialized_decode.runtime.decode import (
            SpecializedDecode,
        )

        assert self.specialized_decode_config is not None
        self.specialized_decode = SpecializedDecode(
            self.model, self.specialized_decode_config
        )

    def get_model_state_cls(self) -> "type[ModelState]":
        from vllm.models.kimi_k3.amd.specialized_decode.model_state import (
            KimiK3ModelState,
        )

        return KimiK3ModelState

    def plan_specialized_decode(
        self, positions: torch.Tensor, intermediate_tensors
    ) -> "Step | None":
        """This step's plan (`Step.layer(i)` per decoder layer), None for the
        stock forward."""
        decode = self.specialized_decode
        return None if decode is None else decode.plan(positions, intermediate_tensors)

    def after_specialized_decode_step(self) -> None:
        if self.specialized_decode is not None:
            self.specialized_decode.after_step()
