# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.wan2_2.conftest import StubScheduler, StubTransformer, StubVAE, noop_progress_bar
from vllm_omni.diffusion.models.wan2_2.easycache import WanEasyCacheState
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import (
    Wan22I2VPipeline,
    get_wan22_i2v_post_process_func,
    get_wan22_i2v_pre_process_func,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_wan22_i2v_postprocess_honors_request_output_type() -> None:
    video = torch.zeros(1, 4, 1, 2, 2)

    output = get_wan22_i2v_post_process_func(SimpleNamespace())(
        video,
        sampling_params=SimpleNamespace(output_type="latent"),
    )

    assert output is video


def _make_i2v_pipeline(*, expand_timesteps: bool) -> Wan22I2VPipeline:
    pipeline = object.__new__(Wan22I2VPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = StubTransformer(name="high", in_channels=8, out_channels=4)
    pipeline.transformer_2 = StubTransformer(name="low", in_channels=8, out_channels=4)
    pipeline.vae = StubVAE(z_dim=4)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.expand_timesteps = expand_timesteps
    pipeline.progress_bar = noop_progress_bar
    return pipeline


def _make_i2v_sampling(**overrides):
    values = {
        "height": 16,
        "width": 16,
        "num_frames": 5,
        "num_inference_steps": 1,
        "guidance_scale_provided": True,
        "guidance_scale": 1.0,
        "guidance_scale_2": None,
        "guidance_scale_2_provided": False,
        "boundary_ratio": None,
        "generator": None,
        "seed": None,
        "num_outputs_per_prompt": 1,
        "max_sequence_length": 8,
        "latents": None,
        "output_type": "latent",
        "extra_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_i2v_preprocess_requires_image_and_resizes_to_480p_aspect() -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())
    request = SimpleNamespace(
        prompt={"prompt": "p", "multi_modal_data": {"image": Image.new("RGB", (320, 160), "red")}},
        sampling_params=SimpleNamespace(height=None, width=None),
    )

    result = preprocess(request)
    prompt = result.prompt

    assert result.sampling_params.height == 432
    assert result.sampling_params.width == 880
    assert prompt["multi_modal_data"]["image"].size == (880, 432)

    missing_image = SimpleNamespace(
        prompt={"prompt": "p", "multi_modal_data": {}},
        sampling_params=SimpleNamespace(height=None, width=None),
    )
    with pytest.raises(ValueError, match="No image is provided"):
        preprocess(missing_image)


def _make_i2v_preprocess_request(last_image):
    return SimpleNamespace(
        prompt={
            "prompt": "p",
            "multi_modal_data": {
                "image": Image.new("RGB", (320, 160), "red"),
                "last_image": last_image,
            },
        },
        sampling_params=SimpleNamespace(height=16, width=16),
    )


def test_i2v_preprocess_treats_empty_last_image_list_as_absent() -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())

    result = preprocess(_make_i2v_preprocess_request([]))

    assert result.prompt["multi_modal_data"]["last_image"] is None
    assert result.batch_compatibility_key == ("wan22_i2v_last_image", False)


def test_i2v_preprocess_unwraps_single_last_image_list() -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())
    last_image = Image.new("RGB", (16, 16), "blue")

    result = preprocess(_make_i2v_preprocess_request([last_image]))

    assert result.prompt["multi_modal_data"]["last_image"] is last_image
    assert result.batch_compatibility_key == ("wan22_i2v_last_image", True)


@pytest.mark.parametrize(
    "last_image",
    [
        Image.new("RGB", (16, 16), "blue"),
        torch.zeros(3, 16, 16),
    ],
)
def test_i2v_preprocess_preserves_supported_last_image(last_image) -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())

    result = preprocess(_make_i2v_preprocess_request(last_image))

    assert result.prompt["multi_modal_data"]["last_image"] is last_image
    assert result.batch_compatibility_key == ("wan22_i2v_last_image", True)


def test_i2v_preprocess_loads_last_image_path(tmp_path) -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())
    path = tmp_path / "last.png"
    Image.new("RGB", (16, 16), "blue").save(path)

    result = preprocess(_make_i2v_preprocess_request(str(path)))

    last_image = result.prompt["multi_modal_data"]["last_image"]
    assert isinstance(last_image, Image.Image)
    assert last_image.mode == "RGB"
    assert result.batch_compatibility_key == ("wan22_i2v_last_image", True)


def test_i2v_preprocess_rejects_multiple_last_images() -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())

    with pytest.raises(ValueError, match="at most one last_image"):
        preprocess(
            _make_i2v_preprocess_request(
                [
                    Image.new("RGB", (16, 16), "blue"),
                    Image.new("RGB", (16, 16), "green"),
                ]
            )
        )


def test_i2v_preprocess_rejects_unsupported_last_image_type() -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())

    with pytest.raises(TypeError, match="Unsupported last_image format"):
        preprocess(_make_i2v_preprocess_request({"not": "an image"}))


def test_i2v_diffuse_selects_stage_guidance_and_expands_timesteps() -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    latents = torch.zeros(1, 4, 2, 4, 4)
    condition = torch.ones_like(latents)
    first_frame_mask = torch.ones(1, 1, 2, 4, 4)
    first_frame_mask[:, :, 0] = 0
    timesteps = torch.tensor([900, 100])

    calls = []

    def fake_predict_noise_maybe_with_cfg(**kwargs):
        positive = kwargs["positive_kwargs"]
        calls.append(
            {
                "model": positive["current_model"].name,
                "scale": kwargs["true_cfg_scale"],
                "timestep_shape": tuple(positive["timestep"].shape),
                "timestep_values": positive["timestep"].clone(),
                "hidden_states": positive["hidden_states"].clone(),
            }
        )
        return torch.ones_like(latents)

    pipeline.predict_noise_maybe_with_cfg = fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]
    pipeline.scheduler_step_maybe_with_cfg = lambda noise, t, current, cfg: current + noise  # type: ignore[method-assign]

    result = pipeline.diffuse(
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=torch.zeros(1, 2, 3),
        negative_prompt_embeds=None,
        image_embeds=None,
        guidance_low=1.0,
        guidance_high=2.0,
        boundary_timestep=500.0,
        dtype=torch.float32,
        attention_kwargs={},
        condition=condition,
        first_frame_mask=first_frame_mask,
    )

    assert [call["model"] for call in calls] == ["high", "low"]
    assert [call["scale"] for call in calls] == [1.0, 2.0]
    assert calls[0]["timestep_shape"] == (1, 8)
    timestep_dtype = calls[0]["timestep_values"].dtype
    torch.testing.assert_close(calls[0]["timestep_values"][0, :4], torch.zeros(4, dtype=timestep_dtype))
    torch.testing.assert_close(calls[0]["timestep_values"][0, 4:], torch.full((4,), 900, dtype=timestep_dtype))
    torch.testing.assert_close(
        calls[0]["hidden_states"][:, :, 0],
        torch.ones_like(calls[0]["hidden_states"][:, :, 0]),
    )
    torch.testing.assert_close(result, torch.full_like(latents, 2.0))


class _FixedRiskPredictor(nn.Module):
    horizon = 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.new_tensor([[0.01, 0.02]])


def _make_easycache_state(num_steps: int = 5) -> WanEasyCacheState:
    return WanEasyCacheState(
        predictor=_FixedRiskPredictor(),
        calibration_offsets=torch.zeros(2),
        threshold=1.0,
        warmup_steps=0,
        num_steps=num_steps,
    )


def test_i2v_easycache_reuses_branch_caches_before_cfg_combine() -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    pipeline._easycache_state = _make_easycache_state()
    pipeline._current_timestep = torch.tensor(900)
    raw = torch.ones(1, 4, 1, 2, 2)
    calls: list[str] = []

    def fake_predict_noise(**kwargs):
        branch = kwargs["branch"]
        calls.append(branch)
        residual = 2.0 if branch == "cond" else 0.5
        return raw + residual

    pipeline.predict_noise = fake_predict_noise  # type: ignore[method-assign]
    positive_kwargs = {"current_model": pipeline.transformer, "branch": "cond"}
    negative_kwargs = {"current_model": pipeline.transformer, "branch": "uncond"}

    first = pipeline.predict_noise_maybe_with_easycache(
        do_true_cfg=True,
        true_cfg_scale=3.0,
        positive_kwargs=positive_kwargs,
        negative_kwargs=negative_kwargs,
        cfg_normalize=False,
        raw_input=raw,
        step_idx=0,
    )
    torch.testing.assert_close(first, (raw + 0.5) + 3.0 * ((raw + 2.0) - (raw + 0.5)))

    raw2 = torch.full_like(raw, 10.0)
    second = pipeline.predict_noise_maybe_with_easycache(
        do_true_cfg=True,
        true_cfg_scale=3.0,
        positive_kwargs=positive_kwargs,
        negative_kwargs=negative_kwargs,
        cfg_normalize=False,
        raw_input=raw2,
        step_idx=1,
    )

    torch.testing.assert_close(second, (raw2 + 0.5) + 3.0 * ((raw2 + 2.0) - (raw2 + 0.5)))
    assert calls == ["cond", "uncond"]
    assert pipeline._easycache_state.stats.skip_pairs == 1
    assert pipeline._easycache_state.stats.skip_forwards == 2


def test_i2v_easycache_separates_patch_and_transformer_state() -> None:
    state = _make_easycache_state()
    raw = torch.ones(1, 4, 1, 2, 2)
    state.update_branch(key=("transformer", 0, "cond"), raw_input=raw, output=raw + 1)
    state.update_branch(key=("transformer", 0, "uncond"), raw_input=raw, output=raw + 2)

    assert state.should_skip_pair(
        pair_key=("transformer", 0),
        raw_input=raw + 1,
        timestep_value=500.0,
        step_idx=1,
        do_true_cfg=True,
    )
    assert not state.should_skip_pair(
        pair_key=("transformer", 1),
        raw_input=raw + 1,
        timestep_value=500.0,
        step_idx=1,
        do_true_cfg=True,
    )
    assert not state.should_skip_pair(
        pair_key=("transformer_2", 0),
        raw_input=raw + 1,
        timestep_value=500.0,
        step_idx=1,
        do_true_cfg=True,
    )


def test_i2v_easycache_warmup_covers_pipefusion_warmup(monkeypatch) -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v.load_horizon_predictor",
        lambda checkpoint_path, device, threshold_override: (_FixedRiskPredictor(), torch.zeros(2), 1.0, {}),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v.is_pipefusion_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v.get_pipefusion_runtime",
        lambda: SimpleNamespace(warmup_steps=5),
    )
    pipeline._safe_pipeline_parallel_world_size = lambda: 1  # type: ignore[method-assign]
    pipeline._safe_cfg_parallel_world_size = lambda: 1  # type: ignore[method-assign]

    pipeline._configure_easycache_for_request(
        [SimpleNamespace(extra_args={"lazy_enabled": True, "lazy_ckpt": "ckpt", "lazy_warmup_steps": 2})],
        num_steps=10,
    )

    assert pipeline._easycache_state.warmup_steps == 5


def test_i2v_prepare_latents_builds_expand_condition_and_first_frame_mask() -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    latents, condition, first_frame_mask = pipeline.prepare_latents(
        image=torch.zeros(1, 3, 16, 16),
        batch_size=1,
        num_channels_latents=4,
        height=16,
        width=16,
        num_frames=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(0),
    )

    assert latents.shape == (1, 4, 2, 2, 2)
    assert condition.shape == (1, 4, 1, 2, 2)
    assert first_frame_mask.shape == (1, 1, 2, 2, 2)
    assert first_frame_mask[:, :, 0].sum() == 0
    assert first_frame_mask[:, :, 1].sum() == 4


def test_i2v_prepare_latents_preserves_batched_image_conditions() -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    generators = [
        torch.Generator(device="cpu").manual_seed(1),
        torch.Generator(device="cpu").manual_seed(2),
    ]

    latents, condition, first_frame_mask = pipeline.prepare_latents(
        image=torch.zeros(2, 3, 16, 16),
        batch_size=2,
        num_channels_latents=4,
        height=16,
        width=16,
        num_frames=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=generators,
    )

    assert latents.shape == (2, 4, 2, 2, 2)
    assert condition.shape == (2, 4, 1, 2, 2)
    assert first_frame_mask.shape == (2, 1, 2, 2, 2)


def test_i2v_forward_batches_conditions_random_inputs_and_outputs(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v.current_omni_platform",
        SimpleNamespace(is_available=lambda: False),
    )
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    pipeline.scheduler = StubScheduler([9])
    pipeline.od_config = SimpleNamespace(flow_shift=5.0)
    pipeline._sample_solver = "unipc"
    pipeline._flow_shift = 5.0
    pipeline.boundary_ratio = 0.875
    pipeline.has_image_encoder = False
    pipeline._guidance_scale = None
    pipeline._guidance_scale_2 = None
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.check_inputs = lambda **kwargs: None
    prepare_call = {}

    def _fake_encode_prompt(**kwargs):
        batch_size = len(kwargs["prompt"])
        n = batch_size * kwargs["num_videos_per_prompt"]
        return torch.zeros(n, 2, 3), None

    def _fake_prepare_latents(**kwargs):
        prepare_call.update(kwargs)
        batch_size = kwargs["batch_size"]
        return (
            kwargs["latents"],
            torch.zeros(batch_size, 4, 1, 2, 2),
            torch.ones(batch_size, 1, 2, 2, 2),
        )

    pipeline.encode_prompt = _fake_encode_prompt  # type: ignore[method-assign]
    pipeline.prepare_latents = _fake_prepare_latents  # type: ignore[method-assign]
    pipeline.diffuse = lambda **kwargs: kwargs["latents"]  # type: ignore[method-assign]

    gen_a = torch.Generator(device="cpu").manual_seed(1)
    gen_b = torch.Generator(device="cpu").manual_seed(2)
    latents_a = torch.zeros(2, 4, 2, 2, 2)
    latents_b = torch.ones(2, 4, 2, 2, 2)
    image_a = torch.zeros(1, 3, 16, 16)
    image_b = torch.ones(1, 3, 16, 16)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={"prompt": "first", "multi_modal_data": {"image": image_a}},
                sampling_params=_make_i2v_sampling(
                    generator=gen_a,
                    latents=latents_a,
                    num_outputs_per_prompt=2,
                ),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt": "second", "multi_modal_data": {"image": image_b}},
                sampling_params=_make_i2v_sampling(
                    generator=gen_b,
                    latents=latents_b,
                    num_outputs_per_prompt=2,
                ),
            ),
        ]
    )

    outputs = pipeline.forward(batch)

    assert prepare_call["batch_size"] == 4
    assert prepare_call["generator"] == [gen_a, gen_a, gen_b, gen_b]
    torch.testing.assert_close(prepare_call["latents"], torch.cat([latents_a, latents_b]))
    torch.testing.assert_close(
        prepare_call["image"],
        torch.cat([image_a, image_a, image_b, image_b]),
    )
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0].output, latents_a)
    torch.testing.assert_close(outputs[1].output, latents_b)


def test_i2v_forward_rejects_mismatched_tensor_condition_shapes(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v.current_omni_platform",
        SimpleNamespace(is_available=lambda: False),
    )
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    pipeline.scheduler = StubScheduler([9])
    pipeline.od_config = SimpleNamespace(flow_shift=5.0)
    pipeline._sample_solver = "unipc"
    pipeline._flow_shift = 5.0
    pipeline.boundary_ratio = 0.875
    pipeline.has_image_encoder = False
    pipeline._guidance_scale = None
    pipeline._guidance_scale_2 = None
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.check_inputs = lambda **kwargs: None
    pipeline.encode_prompt = lambda **kwargs: (torch.zeros(2, 2, 3), None)  # type: ignore[method-assign]
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={"prompt": "first", "multi_modal_data": {"image": torch.zeros(1, 3, 16, 16)}},
                sampling_params=_make_i2v_sampling(),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt": "second", "multi_modal_data": {"image": torch.zeros(1, 3, 16, 8)}},
                sampling_params=_make_i2v_sampling(),
            ),
        ]
    )

    with pytest.raises(ValueError, match="image condition"):
        pipeline.forward(batch)


def test_i2v_forward_rejects_mixed_last_image_presence() -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    image = torch.zeros(1, 3, 16, 16)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={
                    "prompt": "first",
                    "multi_modal_data": {"image": image, "last_image": image},
                },
                sampling_params=_make_i2v_sampling(),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt": "second", "multi_modal_data": {"image": image}},
                sampling_params=_make_i2v_sampling(),
            ),
        ]
    )

    with pytest.raises(ValueError, match="mix of provided and missing last_image"):
        pipeline.forward(batch)
