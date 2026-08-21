# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.interface import supports_step_execution
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import Wan22I2VPipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v import Wan22S2VPipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace import Wan22VACEPipeline
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _pipeline() -> Wan22Pipeline:
    pipeline = object.__new__(Wan22Pipeline)
    pipeline.expand_timesteps = False
    pipeline.interrupt = False
    pipeline.transformer = object()
    pipeline.transformer_2 = object()
    pipeline.transformer_config = SimpleNamespace(patch_size=(1, 2, 2))
    return pipeline


def _state(
    request_id: str,
    *,
    step_index: int,
    latents: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None = None,
) -> StepRequestState:
    state = StepRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(),
        prompt="prompt",
    )
    state.step_index = step_index
    state.timesteps = torch.tensor([900.0, 100.0])
    state.latents = latents
    state.prompt_embeds = torch.ones(latents.shape[0], 2, 3)
    state.negative_prompt_embeds = negative_prompt_embeds
    state.extra.update(
        {
            "attention_kwargs": {},
            "boundary_timestep": 500.0,
            "dtype": torch.float32,
            "guidance_high": 6.0,
            "guidance_low": 4.0,
        }
    )
    return state


def test_base_wan_declares_step_execution_but_vace_does_not() -> None:
    assert supports_step_execution(object.__new__(Wan22Pipeline))
    assert not supports_step_execution(object.__new__(Wan22I2VPipeline))
    assert not supports_step_execution(object.__new__(Wan22S2VPipeline))
    assert not supports_step_execution(object.__new__(Wan22VACEPipeline))


def test_denoise_step_groups_by_current_wan_stage_and_preserves_request_order() -> None:
    pipeline = _pipeline()
    captured: list[tuple[object, float]] = []

    def fake_predict_noise_maybe_with_cfg(
        *,
        do_true_cfg,
        true_cfg_scale,
        positive_kwargs,
        negative_kwargs,
        cfg_normalize,
    ):
        assert do_true_cfg
        assert negative_kwargs is not None
        assert not cfg_normalize
        current_model = positive_kwargs["current_model"]
        captured.append((current_model, true_cfg_scale))
        offset = 10.0 if current_model is pipeline.transformer else 20.0
        return positive_kwargs["hidden_states"] + offset

    pipeline.predict_noise_maybe_with_cfg = fake_predict_noise_maybe_with_cfg
    high_noise = _state(
        "high",
        step_index=0,
        latents=torch.tensor([[1.0]]),
        negative_prompt_embeds=torch.zeros(1, 2, 3),
    )
    low_noise = _state(
        "low",
        step_index=1,
        latents=torch.tensor([[2.0]]),
        negative_prompt_embeds=torch.zeros(1, 2, 3),
    )

    output = pipeline.denoise_step(object(), states=[high_noise, low_noise])

    torch.testing.assert_close(output, torch.tensor([[11.0], [22.0]]))
    assert captured == [(pipeline.transformer, 4.0), (pipeline.transformer_2, 6.0)]


def test_denoise_step_allows_pipeline_parallel_non_output_rank() -> None:
    pipeline = _pipeline()
    pipeline.predict_noise_maybe_with_cfg = lambda **_: None
    state = _state(
        "pp-first-rank",
        step_index=0,
        latents=torch.tensor([[1.0]]),
        negative_prompt_embeds=torch.zeros(1, 2, 3),
    )

    assert pipeline.denoise_step(object(), states=[state]) is None


def test_step_scheduler_uses_current_stage_cfg_flag() -> None:
    pipeline = _pipeline()
    state = _state(
        "low-no-cfg",
        step_index=1,
        latents=torch.tensor([[2.0]]),
        negative_prompt_embeds=torch.zeros(1, 1, 1),
    )
    state.extra["guidance_high"] = 1.0
    state.scheduler = object()
    captured = {}

    def fake_scheduler_step_maybe_with_cfg(noise_pred, timestep, latents, do_true_cfg, *, per_request_scheduler):
        captured["do_true_cfg"] = do_true_cfg
        captured["timestep"] = timestep
        captured["per_request_scheduler"] = per_request_scheduler
        return latents + noise_pred

    pipeline.scheduler_step_maybe_with_cfg = fake_scheduler_step_maybe_with_cfg

    pipeline.step_scheduler(state, torch.tensor([[3.0]]))

    assert captured["do_true_cfg"] is False
    assert captured["per_request_scheduler"] is state.scheduler
    torch.testing.assert_close(state.latents, torch.tensor([[5.0]]))
    assert state.step_index == 2
