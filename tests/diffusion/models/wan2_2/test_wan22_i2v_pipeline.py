# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.wan2_2.conftest import StubScheduler, StubTransformer, StubVAE, noop_progress_bar
from vllm_omni.diffusion.models.wan2_2.easycache import (
    FINAL_FULL_STEPS,
    WanEasyCacheConfig,
    WanEasyCacheState,
    rotational_comm_unsafe_patches,
    rotational_nonlast_predicted_patches,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import _WAN_TEXT_ENCODER_OFFLOAD_PLAN, build_wan_scheduler
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


def test_i2v_pipeline_declares_text_encoder_offload_blocks() -> None:
    assert Wan22I2VPipeline._offload_plan is _WAN_TEXT_ENCODER_OFFLOAD_PLAN


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
    pipeline._init_easycache_state()
    return pipeline


def _make_i2v_sampling(**overrides):
    values: dict[str, object] = {
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


def _make_easycache_state(num_steps: int = 5) -> WanEasyCacheState:
    return WanEasyCacheState(threshold=1.0, warmup_steps=0, num_steps=num_steps)


def test_i2v_easycache_reuses_branch_caches_before_cfg_combine() -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    pipeline._easycache_state = _make_easycache_state()
    pipeline._current_timestep = torch.tensor(900)
    raw = torch.ones(1, 4, 1, 2, 2)
    raw2 = torch.full_like(raw, 1.01)
    raw3 = torch.full_like(raw, 1.02)
    calls: list[str] = []

    def fake_predict_noise(**kwargs):
        calls.append(kwargs["branch"])
        return kwargs["hidden_states"] * (1.5 if kwargs["branch"] == "cond" else 0.9)

    pipeline.predict_noise = fake_predict_noise  # type: ignore[method-assign]
    positive_kwargs = {"current_model": pipeline.transformer, "branch": "cond", "hidden_states": raw}
    negative_kwargs = {"current_model": pipeline.transformer, "branch": "uncond", "hidden_states": raw}

    def run(raw_input, step_idx):
        positive_kwargs["hidden_states"] = raw_input
        negative_kwargs["hidden_states"] = raw_input
        return pipeline.predict_noise_maybe_with_easycache(
            do_true_cfg=True,
            true_cfg_scale=3.0,
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
            raw_input=raw_input,
            step_idx=step_idx,
        )

    first = run(raw, 0)
    first_pos, first_neg = 1.5 * raw, 0.9 * raw
    torch.testing.assert_close(first, first_neg + 3.0 * (first_pos - first_neg))

    second = run(raw2, 1)
    second_pos, second_neg = 1.5 * raw2, 0.9 * raw2
    torch.testing.assert_close(second, second_neg + 3.0 * (second_pos - second_neg))
    assert pipeline._easycache_last_call_skipped is False

    third = run(raw3, 2)
    # Steps 0 and 1 compute (warmup history), step 2 skips and reconstructs with
    # the residual-delta correction.
    assert pipeline._easycache_last_call_skipped is True
    cache_pos = (1.5 * raw2) - raw2
    cache_pos_prev = (1.5 * raw) - raw
    cache_neg = (0.9 * raw2) - raw2
    cache_neg_prev = (0.9 * raw) - raw
    expected_pos = raw3 + cache_pos + (cache_pos - cache_pos_prev)
    expected_neg = raw3 + cache_neg + (cache_neg - cache_neg_prev)
    torch.testing.assert_close(third, expected_neg + 3.0 * (expected_pos - expected_neg))

    assert calls == ["cond", "uncond", "cond", "uncond"]
    stats = pipeline._easycache_state.stats
    assert stats.skip_pairs == 1
    assert stats.skip_forwards == 2
    assert stats.calc_pairs == 2
    assert stats.calc_forwards == 4


def test_i2v_easycache_separates_patch_and_transformer_state() -> None:
    state = _make_easycache_state()
    raw = torch.ones(1, 4, 1, 2, 2)
    state.update_branch(key=("transformer", 0, "cond"), raw_input=raw, output=1.1 * raw)
    state.update_branch(key=("transformer", 0, "cond"), raw_input=1.01 * raw, output=1.11 * raw)
    state.update_branch(key=("transformer", 0, "uncond"), raw_input=raw, output=0.9 * raw)

    assert state.should_skip_pair(
        pair_key=("transformer", 0),
        raw_input=1.02 * raw,
        timestep_value=500.0,
        step_idx=1,
        do_true_cfg=True,
    )
    assert not state.should_skip_pair(
        pair_key=("transformer", 1),
        raw_input=1.02 * raw,
        timestep_value=500.0,
        step_idx=1,
        do_true_cfg=True,
    )
    assert not state.should_skip_pair(
        pair_key=("transformer_2", 0),
        raw_input=1.02 * raw,
        timestep_value=500.0,
        step_idx=1,
        do_true_cfg=True,
    )


def test_i2v_easycache_warmup_covers_pipefusion_warmup(monkeypatch) -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.easycache.is_pipefusion_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.easycache.get_pipefusion_runtime",
        lambda: SimpleNamespace(warmup_steps=5),
    )
    pipeline._safe_pipeline_parallel_world_size = lambda: 1  # type: ignore[method-assign]
    pipeline._safe_cfg_parallel_world_size = lambda: 1  # type: ignore[method-assign]

    pipeline._configure_easycache_for_request(
        [SimpleNamespace(extra_args={"d2cache_enabled": True, "d2cache_warmup_steps": 2})],
        num_steps=10,
    )

    assert pipeline._easycache_state.warmup_steps == 5


def _async_entered_patches(*, pp_size: int, num_patch: int, step_i: int, num_async_steps: int) -> set[int]:
    """Patches a non-last stage really calls predict() for.

    Independent mirror of ``PipeFusionPipelineMixin._async_pipeline``: rotation
    seeds each rank's order with ``get_initial_patch_indices()`` and rolls it
    right once per step; ``skip_patch`` drops that rank's own last position, and
    the final async step truncates positions ``>= rank + 1``.
    """
    if step_i <= 0:
        return set(range(num_patch))
    entered: set[int] = set()
    for rank in range(pp_size - 1):
        order = list(range(num_patch))
        shift = rank % num_patch
        order = order[-shift:] + order[:-shift]
        for _ in range(step_i - 1):
            order = order[-1:] + order[:-1]
        for ip, pidx in enumerate(order):
            skip_patch = ip == num_patch - 1 or (step_i == num_async_steps - 1 and ip >= rank + 1)
            if not skip_patch:
                entered.add(pidx)
    return entered


def _last_stage_anchor(*, pp_size: int, num_patch: int, step_i: int) -> int:
    """``get_last_stage_patch_indices(step_i - 1)[-1]`` for the same config."""
    order = list(range(num_patch))
    shift = (pp_size - 1) % num_patch
    order = order[-shift:] + order[:-shift]
    for _ in range(step_i - 1):
        order = order[-1:] + order[:-1]
    return order[-1]


@pytest.mark.parametrize("pp_size,num_patch", [(2, 2), (2, 3), (2, 4), (3, 6), (4, 4), (4, 8)])
@pytest.mark.parametrize("step_i", [0, 1, 2, 7, 44])
def test_rotational_predicted_patches_supersets_every_entered_patch(pp_size: int, num_patch: int, step_i: int) -> None:
    # The plan handshake is dropped whenever this set is a subset of the patches
    # the last stage is forced to compute, so an under-report here would let a
    # rank keep sending into a channel its peer decided not to receive on.
    entered = _async_entered_patches(pp_size=pp_size, num_patch=num_patch, step_i=step_i, num_async_steps=45)
    predicted = rotational_nonlast_predicted_patches(pp_size=pp_size, num_patch=num_patch, step_i=step_i)
    assert entered <= predicted, f"pp={pp_size} N={num_patch} step_i={step_i}: {sorted(entered - predicted)}"


@pytest.mark.parametrize("pp_size,num_patch", [(2, 2), (2, 4), (4, 4), (4, 8)])
def test_plan_handshake_elision_never_hides_a_skippable_patch(pp_size: int, num_patch: int) -> None:
    """The plan broadcast may only be dropped for patches the last stage cannot skip.

    ``plan_easycache_step`` elides the handshake when every patch a non-last
    stage may enter is a proven calc on the last stage: inside the protected
    region ``should_skip_pair`` never returns True, and the anchor and
    comm-unsafe patches are rolled back to calc. Any other patch stays
    skippable and must still be broadcast.
    """
    num_steps, pipefusion_warmup, easycache_warmup = 50, 5, 7
    num_async = num_steps - pipefusion_warmup
    for step_i in range(num_async):
        step_idx = pipefusion_warmup + step_i
        guaranteed = (
            set(range(num_patch)) if step_idx < easycache_warmup or step_idx >= num_steps - FINAL_FULL_STEPS else set()
        )
        if step_i > 0:
            guaranteed.add(_last_stage_anchor(pp_size=pp_size, num_patch=num_patch, step_i=step_i))
            guaranteed |= rotational_comm_unsafe_patches(
                pp_size=pp_size, num_patch=num_patch, num_async_steps=num_async, step_i=step_i
            )
        predicted = rotational_nonlast_predicted_patches(pp_size=pp_size, num_patch=num_patch, step_i=step_i)
        if predicted <= guaranteed:
            entered = _async_entered_patches(
                pp_size=pp_size, num_patch=num_patch, step_i=step_i, num_async_steps=num_async
            )
            assert entered <= guaranteed, (
                f"pp={pp_size} N={num_patch} step_i={step_i}: elides while {sorted(entered - guaranteed)} "
                "could still be skipped by the last stage"
            )


@pytest.mark.parametrize("pp_size,num_patch", [(2, 2), (2, 4), (4, 4), (4, 8)])
def test_anchor_force_off_never_elides_the_handshake(pp_size: int, num_patch: int) -> None:
    """With the anchor force off, the anchor is genuinely skippable.

    The elide proof is "every patch a non-last stage may run is a proven calc".
    At pp=2/num_patch=2 the only patch the first stage runs is the anchor, so
    dropping the anchor from ``guaranteed_calc`` must collapse the subset test
    and restore the per-step broadcast -- otherwise the first stage would keep
    its all-zero flags while the last stage skips the anchor, orphaning an
    isend on that patch's comm-id.
    """
    num_steps, pipefusion_warmup, easycache_warmup = 50, 5, 7
    num_async = num_steps - pipefusion_warmup
    for step_i in range(1, num_async):
        step_idx = pipefusion_warmup + step_i
        protected = step_idx < easycache_warmup or step_idx >= num_steps - FINAL_FULL_STEPS
        anchor = _last_stage_anchor(pp_size=pp_size, num_patch=num_patch, step_i=step_i)
        unsafe = rotational_comm_unsafe_patches(
            pp_size=pp_size, num_patch=num_patch, num_async_steps=num_async, step_i=step_i
        )
        predicted = rotational_nonlast_predicted_patches(pp_size=pp_size, num_patch=num_patch, step_i=step_i)
        base = set(range(num_patch)) if protected else set()
        entered = _async_entered_patches(pp_size=pp_size, num_patch=num_patch, step_i=step_i, num_async_steps=num_async)

        on = base | {anchor} | unsafe
        off = base | unsafe
        # Dropping the anchor force only ever removes proven-calc entries, so it
        # can turn the elide off but must never turn it on where it was unsafe.
        assert not (predicted <= off) or predicted <= on
        if predicted <= off and not protected:
            assert entered <= off, (
                f"pp={pp_size} N={num_patch} step_i={step_i}: elides with the anchor force off "
                f"while {sorted(entered - off)} could still be skipped"
            )


def test_force_anchor_calc_defaults_on_and_is_overridable() -> None:
    """The anchor force is opt-out: absent means forced (previous behaviour)."""
    assert WanEasyCacheConfig().force_anchor_calc is True

    pipeline = _make_i2v_pipeline(expand_timesteps=False)
    params = SimpleNamespace(
        extra_args={
            "d2cache_enabled": True,
            "d2cache_threshold": 0.05,
            "d2cache_warmup_steps": 7,
            "d2cache_epsilon": 1e-8,
            "d2cache_log_stats": True,
        }
    )
    config = pipeline._resolve_easycache_config([params])
    assert config.force_anchor_calc is True

    for falsy in (False, "false", "0", "no"):
        params.extra_args["d2cache_force_anchor_calc"] = falsy
        assert pipeline._resolve_easycache_config([params]).force_anchor_calc is False
    for truthy in (True, "true", "1", "yes"):
        params.extra_args["d2cache_force_anchor_calc"] = truthy
        assert pipeline._resolve_easycache_config([params]).force_anchor_calc is True


def test_it_residual_cache_is_off_until_requested_on_a_multi_stage_pipeline(monkeypatch) -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=False)
    params = SimpleNamespace(
        extra_args={
            "d2cache_enabled": True,
            "it_residual_enabled": True,
            "it_residual_threshold": 0.2,
            "it_residual_second_order": True,
        }
    )
    config = pipeline._resolve_easycache_config([params])
    assert config.it_residual_enabled is True
    assert config.it_residual_threshold == 0.2
    assert config.it_residual_second_order is True
    assert WanEasyCacheConfig().it_residual_enabled is False

    pipeline._configure_easycache_for_request([params], num_steps=10)
    assert pipeline._easycache_it_cache is None

    monkeypatch.setattr(pipeline, "_safe_pipeline_parallel_world_size", lambda: 2)
    pipeline._configure_easycache_for_request([params], num_steps=10)
    cache = pipeline._easycache_it_cache
    assert cache is not None
    assert cache.threshold == 0.2
    assert cache.second_order is True
    assert cache.tensors_per_entry() == 3


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


@pytest.mark.parametrize("solver", ["unipc", "euler"])
@pytest.mark.parametrize("shift", [3.0, 5.0, 12.0])
def test_i2v_forward_batches_conditions_random_inputs_and_outputs(monkeypatch, solver: str, shift: float) -> None:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v.current_omni_platform",
        SimpleNamespace(is_available=lambda: False),
    )
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    pipeline.scheduler = build_wan_scheduler(solver, shift)
    pipeline.od_config = SimpleNamespace(flow_shift=shift)
    pipeline._sample_solver = solver
    pipeline._flow_shift = shift
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
                    num_inference_steps=50,
                    extra_args={"sample_solver": solver, "flow_shift": shift},
                ),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt": "second", "multi_modal_data": {"image": image_b}},
                sampling_params=_make_i2v_sampling(
                    generator=gen_b,
                    latents=latents_b,
                    num_outputs_per_prompt=2,
                    num_inference_steps=50,
                    extra_args={"sample_solver": solver, "flow_shift": shift},
                ),
            ),
        ]
    )

    outputs = pipeline.forward(batch)
    if solver == "unipc":
        sigmas = np.linspace(float(np.float32(0.999)), 0.0, 51)[:-1]
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        torch.testing.assert_close(
            pipeline.scheduler.sigmas, torch.tensor(np.append(sigmas, 0.0), dtype=torch.float32), rtol=0, atol=0
        )
        assert pipeline.scheduler.timesteps.tolist() == (sigmas * 1000).astype(np.int64).tolist()
    else:
        reference = build_wan_scheduler("euler", shift)
        reference.set_timesteps(50, device="cpu")
        torch.testing.assert_close(pipeline.scheduler.sigmas, reference.sigmas, rtol=0, atol=0)

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
