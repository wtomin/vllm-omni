# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""D2Cache output-residual cache for Wan2.2 T2V/I2V, including PipeFusion PP skip sync.

Keeps the EasyCache skip schedule (accumulated-error threshold) and adds the
D2Cache residual-delta correction: every skip reconstructs
``raw_input + cache + correction_scale * residual_delta`` where ``residual_delta``
is the change of the output residual since the last full compute.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_world_size,
    get_pp_group,
    is_pipeline_last_stage,
)
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_runtime import (
    get_pipefusion_runtime,
    is_pipefusion_initialized,
)

logger = logging.getLogger(__name__)

DEFAULT_D2CACHE_THRESHOLD = 0.05
DEFAULT_D2CACHE_WARMUP_STEPS = 7
DEFAULT_D2CACHE_EPSILON = 1e-8
# Force full computation for the last N steps of a request.
FINAL_FULL_STEPS = 2

EasyCacheBranch = Literal["cond", "uncond"]
EasyCacheKey = tuple[str, int, EasyCacheBranch]
EasyCachePairKey = tuple[str, int]


@dataclass(frozen=True)
class WanEasyCacheConfig:
    enabled: bool = False
    threshold: float = DEFAULT_D2CACHE_THRESHOLD
    warmup_steps: int = DEFAULT_D2CACHE_WARMUP_STEPS
    epsilon: float = DEFAULT_D2CACHE_EPSILON
    log_stats: bool = False


@dataclass
class WanEasyCacheStats:
    calc_pairs: int = 0
    skip_pairs: int = 0
    calc_forwards: int = 0
    skip_forwards: int = 0
    forced_calc_pairs: int = 0
    correction_scale_sum: float = 0.0
    correction_scale_count: int = 0

    @property
    def mean_correction_scale(self) -> float:
        if self.correction_scale_count == 0:
            return 0.0
        return self.correction_scale_sum / self.correction_scale_count


@dataclass
class _BranchState:
    previous_raw_input: torch.Tensor | None = None
    previous_raw_output: torch.Tensor | None = None
    last_full_input: torch.Tensor | None = None
    cache: torch.Tensor | None = None
    residual_delta: torch.Tensor | None = None


@dataclass
class _PairState:
    accumulated_error: float = 0.0
    previous_error_score: float = 0.0
    correction_scale: float = 1.0
    transformation_rate: float | None = None
    last_decision: str = "calc"
    last_reason: str = "uninitialized"


def _tensor_l1_mean_scalar(tensor: torch.Tensor, other: torch.Tensor | None = None) -> torch.Tensor:
    if other is not None:
        if tensor.shape != other.shape:
            raise RuntimeError(f"EasyCache tensor shape mismatch: {tuple(tensor.shape)} != {tuple(other.shape)}")
        tensor = tensor - other
    if tensor.numel() == 0:
        raise RuntimeError("cannot compute a mean over an empty tensor")
    return tensor.float().abs().mean()


class WanEasyCacheState:
    def __init__(
        self,
        *,
        threshold: float,
        warmup_steps: int,
        num_steps: int,
        epsilon: float = DEFAULT_D2CACHE_EPSILON,
    ) -> None:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("d2cache_threshold must be finite and positive")
        if warmup_steps < 0:
            raise ValueError("d2cache_warmup_steps must be non-negative")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("d2cache_epsilon must be finite and positive")
        self.threshold = float(threshold)
        self.epsilon = float(epsilon)
        self.warmup_steps = int(warmup_steps)
        self.num_steps = int(num_steps)
        self.branch_states: dict[EasyCacheKey, _BranchState] = {}
        self.pair_states: dict[EasyCachePairKey, _PairState] = {}
        self.stats = WanEasyCacheStats()

    def _branch_state(self, key: EasyCacheKey) -> _BranchState:
        state = self.branch_states.get(key)
        if state is None:
            state = _BranchState()
            self.branch_states[key] = state
        return state

    def _pair_state(self, key: EasyCachePairKey) -> _PairState:
        state = self.pair_states.get(key)
        if state is None:
            state = _PairState()
            self.pair_states[key] = state
        return state

    def _history_ready(self, pair_key: EasyCachePairKey, do_true_cfg: bool) -> bool:
        cond = self._branch_state((pair_key[0], pair_key[1], "cond"))
        if cond.previous_raw_input is None or cond.previous_raw_output is None or cond.cache is None:
            return False
        if do_true_cfg:
            uncond = self._branch_state((pair_key[0], pair_key[1], "uncond"))
            return uncond.cache is not None
        return True

    def _pair_history_matches(self, pair_key: EasyCachePairKey, raw_input: torch.Tensor) -> bool:
        branches: tuple[EasyCacheBranch, ...] = ("cond", "uncond")
        for branch in branches:
            state = self.branch_states.get((pair_key[0], pair_key[1], branch))
            if state is None:
                continue
            stored_tensors = (
                state.previous_raw_input,
                state.previous_raw_output,
                state.last_full_input,
                state.cache,
                state.residual_delta,
            )
            if any(tensor is not None and tensor.shape != raw_input.shape for tensor in stored_tensors):
                return False
        return True

    def _reset_pair(self, pair_key: EasyCachePairKey) -> None:
        """Drop pair history after a latent-shape discontinuity.

        PipeFusion warmup runs full latents through key (transformer, 0) while
        the async phase reuses the same key for patch latents; the old history
        cannot be compared against or reused for the new shapes.
        """
        branches: tuple[EasyCacheBranch, ...] = ("cond", "uncond")
        for branch in branches:
            self.branch_states.pop((pair_key[0], pair_key[1], branch), None)
        self.pair_states.pop(pair_key, None)

    def _observe_conditional(self, key: EasyCacheKey, raw_input: torch.Tensor, output: torch.Tensor) -> None:
        branch = self._branch_state(key)
        if branch.previous_raw_output is not None and branch.last_full_input is not None:
            output_change, input_change = torch.stack(
                (
                    _tensor_l1_mean_scalar(output, branch.previous_raw_output),
                    _tensor_l1_mean_scalar(raw_input, branch.last_full_input),
                )
            ).tolist()
            # The rate spans full-compute-to-full-compute pairs, so it is shared
            # by both branches of the (transformer, patch) pair.
            self._pair_state((key[0], key[1])).transformation_rate = output_change / (input_change + self.epsilon)
        branch.last_full_input = raw_input.detach().clone()

    def should_skip_pair(
        self,
        *,
        pair_key: EasyCachePairKey,
        raw_input: torch.Tensor,
        timestep_value: float | torch.Tensor,
        step_idx: int,
        do_true_cfg: bool,
    ) -> bool:
        del timestep_value
        if not self._pair_history_matches(pair_key, raw_input):
            self._reset_pair(pair_key)
        pair = self._pair_state(pair_key)
        protected = step_idx < self.warmup_steps or step_idx >= self.num_steps - FINAL_FULL_STEPS

        if protected:
            # Protected region: bank the assumed error and reset, never skip.
            pair.previous_error_score = pair.accumulated_error
            pair.accumulated_error = 0.0
            pair.last_decision = "calc"
            pair.last_reason = "warmup_or_final"
        elif not self._history_ready(pair_key, do_true_cfg):
            pair.last_decision = "calc"
            pair.last_reason = "history_not_ready"
        elif pair.transformation_rate is None:
            pair.last_decision = "calc"
            pair.last_reason = "rate_unavailable"
        else:
            cond = self._branch_state((pair_key[0], pair_key[1], "cond"))
            assert cond.previous_raw_input is not None
            assert cond.previous_raw_output is not None
            input_change, output_norm = torch.stack(
                (
                    _tensor_l1_mean_scalar(raw_input, cond.previous_raw_input),
                    _tensor_l1_mean_scalar(cond.previous_raw_output),
                )
            ).tolist()
            predicted_change = pair.transformation_rate * input_change / (output_norm + self.epsilon)
            pair.accumulated_error += predicted_change
            if pair.accumulated_error < self.threshold:
                denominator = pair.previous_error_score if pair.previous_error_score != 0.0 else pair.accumulated_error
                pair.correction_scale = pair.accumulated_error / denominator if denominator else 1.0
                self.stats.correction_scale_sum += pair.correction_scale
                self.stats.correction_scale_count += 1
                pair.last_decision = "skip"
                pair.last_reason = "within_threshold"
            else:
                pair.previous_error_score = pair.accumulated_error
                pair.accumulated_error = 0.0
                pair.correction_scale = 1.0
                pair.last_decision = "calc"
                pair.last_reason = "threshold_exceeded"

        if pair.last_decision == "skip":
            self.stats.skip_pairs += 1
            return True
        self.stats.calc_pairs += 1
        return False

    def force_pair_calc(self, pair_key: EasyCachePairKey) -> None:
        """Roll back a skip decision when a full compute is forced for this pair.

        Used for the Rotational PipeFusion anchor patch, which must always run so
        the next step's intermediate-tensor capture is fresh. Mirrors the
        protected-region calc branch: bank the assumed error and reset.
        """
        pair = self._pair_state(pair_key)
        if pair.last_decision != "skip":
            return
        pair.previous_error_score = pair.accumulated_error
        pair.accumulated_error = 0.0
        pair.last_decision = "calc"
        pair.last_reason = "anchor_forced"
        self.stats.skip_pairs -= 1
        self.stats.calc_pairs += 1
        self.stats.forced_calc_pairs += 1
        if self.stats.correction_scale_count > 0:
            self.stats.correction_scale_sum -= pair.correction_scale
            self.stats.correction_scale_count -= 1

    def get_cached_output(
        self,
        *,
        key: EasyCacheKey,
        raw_input: torch.Tensor,
    ) -> torch.Tensor:
        state = self._branch_state(key)
        if state.cache is None:
            raise RuntimeError(f"EasyCache branch cache is missing for {key}")
        self.stats.skip_forwards += 1
        if key[2] == "cond":
            # A skipped pair still advances the conditional input history, which
            # feeds the step-to-step input change of the next decision.
            state.previous_raw_input = raw_input.detach().clone()
        pair = self._pair_state((key[0], key[1]))
        residual = state.cache
        if state.residual_delta is not None:
            residual = residual + pair.correction_scale * state.residual_delta
        return raw_input + residual.to(device=raw_input.device)

    def update_branch(
        self,
        *,
        key: EasyCacheKey,
        raw_input: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if raw_input.shape != output.shape:
            raise RuntimeError(
                f"EasyCache raw_input/output shape mismatch: {tuple(raw_input.shape)} != {tuple(output.shape)}"
            )
        pair_key = (key[0], key[1])
        if not self._pair_history_matches(pair_key, raw_input):
            self._reset_pair(pair_key)
        state = self._branch_state(key)
        if key[2] == "cond":
            self._observe_conditional(key, raw_input, output)
        # Subtraction already returns fresh storage. Cloning that result again
        # adds a full-latent allocation and copy on every computed branch.
        new_cache = output.detach() - raw_input.detach()
        if state.cache is not None:
            state.residual_delta = new_cache - state.cache
        state.previous_raw_input = raw_input.detach().clone()
        state.previous_raw_output = output.detach().clone()
        state.cache = new_cache
        self.stats.calc_forwards += 1


class WanEasyCacheMixin:
    """D2Cache EasyCache for Wan2.2 T2V/I2V, including PipeFusion PP skip sync."""

    def _init_easycache_state(self) -> None:
        self._easycache_config = WanEasyCacheConfig()
        self._easycache_state: WanEasyCacheState | None = None
        self._easycache_step_plan: dict[int, bool] | None = None

    def plan_easycache_step(
        self,
        raw_inputs: dict[int, torch.Tensor],
        step_idx: int,
        do_true_cfg: bool,
        transformer_id: str = "transformer",
    ) -> None:
        """Precompute per-patch skip decisions in fixed patch order, broadcast once per step.

        Under Rotational PipeFusion each PP stage iterates patches in a different
        rotated order and the non-last stages process a different number of
        patches per step, so the per-pair skip broadcast inside
        predict_noise_maybe_with_easycache() would misalign across PP ranks.
        Call this once per denoising step before the patch loop; the resulting
        plan is then consulted by patch id inside the loop.
        """
        state = getattr(self, "_easycache_state", None)
        pp_size = self._safe_pipeline_parallel_world_size()
        self._easycache_step_plan = None
        if state is None or pp_size <= 1:
            return
        patch_ids = sorted(raw_inputs.keys())
        flags = torch.zeros(len(patch_ids), dtype=torch.int32, device=self.device)
        if is_pipeline_last_stage():
            timestep = self._current_timestep
            # should_skip_pair accepts a float or a tensor timestep.
            timestep_value = 0.0 if timestep is None else timestep
            for k, pidx in enumerate(patch_ids):
                if state.should_skip_pair(
                    pair_key=(transformer_id, pidx),
                    raw_input=raw_inputs[pidx],
                    timestep_value=timestep_value,
                    step_idx=step_idx,
                    do_true_cfg=do_true_cfg,
                ):
                    flags[k] = 1
            # Rotational PipeFusion feeds the last stage's last-patch intermediate
            # tensors (captured while computing this step's anchor patch) into the
            # next step's first patch. If EasyCache skipped the anchor patch, the
            # capture would be empty/stale and the next step would reuse mismatched
            # tensors, so the anchor patch is forced to always calc.
            runtime = get_pipefusion_runtime()
            step_i = step_idx - runtime.warmup_steps
            if runtime.use_rotational_pipefusion and step_i > 0:
                anchor_pidx = runtime.get_last_stage_patch_indices(step_i - 1)[-1]
                anchor_index = patch_ids.index(anchor_pidx)
                if flags[anchor_index] == 1:
                    flags[anchor_index] = 0
                    state.force_pair_calc((transformer_id, anchor_pidx))
        get_pp_group().broadcast(flags, src=pp_size - 1)
        self._easycache_step_plan = {pidx: bool(flags[k]) for k, pidx in enumerate(patch_ids)}

    @staticmethod
    def _truthy_extra_arg(value: object, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _safe_pipeline_parallel_world_size() -> int:
        try:
            return int(get_pipeline_parallel_world_size())
        except AssertionError:
            return 1

    @staticmethod
    def _safe_cfg_parallel_world_size() -> int:
        try:
            return int(get_classifier_free_guidance_world_size())
        except AssertionError:
            return 1

    @staticmethod
    def _easycache_relevant_extra_args(sampling_params: Any) -> dict[str, Any]:
        extra_args = getattr(sampling_params, "extra_args", None) or {}
        if not isinstance(extra_args, Mapping):
            raise TypeError("Wan2.2 EasyCache expects sampling_params.extra_args to be a mapping.")
        return {
            "d2cache_enabled": extra_args.get("d2cache_enabled"),
            "d2cache_threshold": extra_args.get("d2cache_threshold"),
            "d2cache_warmup_steps": extra_args.get("d2cache_warmup_steps"),
            "d2cache_epsilon": extra_args.get("d2cache_epsilon"),
            "d2cache_log_stats": extra_args.get("d2cache_log_stats"),
        }

    def _resolve_easycache_config(self, sampling_params_list: list[Any]) -> WanEasyCacheConfig:
        first_extra = self._easycache_relevant_extra_args(sampling_params_list[0])
        for sampling_params in sampling_params_list[1:]:
            if self._easycache_relevant_extra_args(sampling_params) != first_extra:
                raise ValueError("Batched Wan2.2 requests must use identical EasyCache extra_args.")

        enabled = self._truthy_extra_arg(first_extra["d2cache_enabled"])
        if not enabled:
            return WanEasyCacheConfig()

        threshold = DEFAULT_D2CACHE_THRESHOLD
        if first_extra["d2cache_threshold"] is not None:
            threshold = float(first_extra["d2cache_threshold"])
            if not math.isfinite(threshold) or threshold <= 0:
                raise ValueError("Wan2.2 EasyCache d2cache_threshold must be finite and positive.")

        warmup_steps = (
            DEFAULT_D2CACHE_WARMUP_STEPS
            if first_extra["d2cache_warmup_steps"] is None
            else int(first_extra["d2cache_warmup_steps"])
        )
        if warmup_steps < 0:
            raise ValueError("Wan2.2 EasyCache d2cache_warmup_steps must be non-negative.")

        epsilon = DEFAULT_D2CACHE_EPSILON
        if first_extra["d2cache_epsilon"] is not None:
            epsilon = float(first_extra["d2cache_epsilon"])
            if not math.isfinite(epsilon) or epsilon <= 0:
                raise ValueError("Wan2.2 EasyCache d2cache_epsilon must be finite and positive.")

        return WanEasyCacheConfig(
            enabled=True,
            threshold=threshold,
            warmup_steps=warmup_steps,
            epsilon=epsilon,
            log_stats=self._truthy_extra_arg(first_extra["d2cache_log_stats"]),
        )

    def _configure_easycache_for_request(self, sampling_params_list: list[Any], num_steps: int) -> None:
        config = self._resolve_easycache_config(sampling_params_list)
        self._easycache_config = config
        self._easycache_state = None
        self._easycache_step_plan = None
        if not config.enabled:
            return

        if self._safe_cfg_parallel_world_size() > 1:
            raise NotImplementedError("Wan2.2 EasyCache does not yet support CFG parallel.")

        effective_warmup_steps = config.warmup_steps
        if is_pipefusion_initialized():
            effective_warmup_steps = max(effective_warmup_steps, get_pipefusion_runtime().warmup_steps)
        self._easycache_state = WanEasyCacheState(
            threshold=config.threshold,
            epsilon=config.epsilon,
            warmup_steps=effective_warmup_steps,
            num_steps=num_steps,
        )

    def _easycache_transformer_id(self, positive_kwargs: dict[str, Any]) -> str:
        current_model = positive_kwargs.get("current_model")
        if current_model is not None and current_model is getattr(self, "transformer_2", None):
            return "transformer_2"
        return "transformer"

    @staticmethod
    def _easycache_patch_id() -> int:
        if is_pipefusion_initialized():
            runtime = get_pipefusion_runtime()
            if runtime.patch_mode:
                return int(runtime.pipeline_patch_idx)
        return 0

    def _log_easycache_stats(self) -> None:
        state = getattr(self, "_easycache_state", None)
        config = getattr(self, "_easycache_config", WanEasyCacheConfig())
        if state is None or not config.log_stats:
            return
        if self._safe_pipeline_parallel_world_size() > 1 and not is_pipeline_last_stage():
            return
        stats = state.stats
        total_pairs = stats.calc_pairs + stats.skip_pairs
        logger.info(
            "Wan2.2 EasyCache statistics: calc_pairs=%d, skip_pairs=%d, skip_ratio=%.2f%%, "
            "calc_forwards=%d, skip_forwards=%d, forced_calc_pairs=%d, mean_correction_scale=%.4f",
            stats.calc_pairs,
            stats.skip_pairs,
            100.0 * stats.skip_pairs / max(total_pairs, 1),
            stats.calc_forwards,
            stats.skip_forwards,
            stats.forced_calc_pairs,
            stats.mean_correction_scale,
        )

    def _release_easycache_request_state(self) -> None:
        """Release per-request histories and plans."""
        self._easycache_state = None
        self._easycache_step_plan = None

    def predict_noise_maybe_with_easycache(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any] | None,
        cfg_normalize: bool = True,
        output_slice: int | None = None,
        raw_input: torch.Tensor | None = None,
        step_idx: int | None = None,
        skip_sync: bool = False,
        inter_comm_ids: list[str] | None = None,
        intermediate_tensors: list | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        # Signals to PipeFusion (e.g. last-stage IT capture bookkeeping) whether this
        # call reused cached outputs without running predict_noise.
        self._easycache_last_call_skipped = False
        state = getattr(self, "_easycache_state", None)
        if state is None:
            return self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=cfg_normalize,
                output_slice=output_slice,
                skip_sync=skip_sync,
                inter_comm_ids=inter_comm_ids,
                intermediate_tensors=intermediate_tensors,
            )
        if raw_input is None:
            raise ValueError("Wan2.2 EasyCache requires raw denoising latents for cache decisions.")
        if do_true_cfg and negative_kwargs is None:
            raise ValueError("Wan2.2 EasyCache requires negative_kwargs when CFG is enabled.")

        transformer_id = self._easycache_transformer_id(positive_kwargs)
        patch_id = self._easycache_patch_id()
        step_idx = 0 if step_idx is None else step_idx
        pair_key = (transformer_id, patch_id)
        timestep = self._current_timestep
        timestep_value = 0.0 if timestep is None else timestep
        pp_size = self._safe_pipeline_parallel_world_size()
        decide_locally = pp_size == 1 or is_pipeline_last_stage()
        step_plan = getattr(self, "_easycache_step_plan", None)
        if step_plan is not None:
            # Skip decisions for this step were precomputed in fixed patch order
            # by plan_easycache_step() (required under Rotational PipeFusion,
            # where stages iterate patches in different rotated orders).
            should_skip = bool(step_plan.get(patch_id, False))
        else:
            should_skip = False
            if decide_locally:
                should_skip = state.should_skip_pair(
                    pair_key=pair_key,
                    raw_input=raw_input,
                    timestep_value=timestep_value,
                    step_idx=step_idx,
                    do_true_cfg=do_true_cfg,
                )
            if pp_size > 1:
                skip_flag = torch.zeros(1, dtype=torch.int32, device=self.device)
                if decide_locally:
                    skip_flag[0] = int(should_skip)
                get_pp_group().broadcast(skip_flag, src=pp_size - 1)
                should_skip = bool(skip_flag.item())

        if should_skip:
            self._easycache_last_call_skipped = True
            if pp_size > 1 and not is_pipeline_last_stage():
                return None
            positive_noise_pred = state.get_cached_output(
                key=(transformer_id, patch_id, "cond"),
                raw_input=raw_input,
            )
            if do_true_cfg:
                negative_noise_pred = state.get_cached_output(
                    key=(transformer_id, patch_id, "uncond"),
                    raw_input=raw_input,
                )
                if output_slice is not None:
                    positive_noise_pred = positive_noise_pred[:, :output_slice]
                    negative_noise_pred = negative_noise_pred[:, :output_slice]
                return self.combine_cfg_noise(
                    positive_noise_pred,
                    negative_noise_pred,
                    true_cfg_scale,
                    cfg_normalize,
                )
            if output_slice is not None:
                positive_noise_pred = positive_noise_pred[:, :output_slice]
            return positive_noise_pred

        if pp_size > 1:
            result = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=cfg_normalize,
                output_slice=output_slice,
                skip_sync=skip_sync,
                inter_comm_ids=inter_comm_ids,
                intermediate_tensors=intermediate_tensors,
                return_uncombined=True,
            )
            if result is None:
                return None
            combined, positive_noise_pred, negative_noise_pred = result
            state.update_branch(
                key=(transformer_id, patch_id, "cond"),
                raw_input=raw_input,
                output=positive_noise_pred,
            )
            if do_true_cfg:
                assert negative_noise_pred is not None
                state.update_branch(
                    key=(transformer_id, patch_id, "uncond"),
                    raw_input=raw_input,
                    output=negative_noise_pred,
                )
            return combined

        positive_noise_pred = self.predict_noise(**positive_kwargs)
        if isinstance(positive_noise_pred, IntermediateTensors):
            raise NotImplementedError("Wan2.2 EasyCache does not support pipeline-parallel intermediate tensors yet.")
        state.update_branch(
            key=(transformer_id, patch_id, "cond"),
            raw_input=raw_input,
            output=positive_noise_pred,
        )
        if do_true_cfg:
            assert negative_kwargs is not None
            negative_noise_pred = self.predict_noise(**negative_kwargs)
            if isinstance(negative_noise_pred, IntermediateTensors):
                raise NotImplementedError(
                    "Wan2.2 EasyCache does not support pipeline-parallel intermediate tensors yet."
                )
            state.update_branch(
                key=(transformer_id, patch_id, "uncond"),
                raw_input=raw_input,
                output=negative_noise_pred,
            )
            if output_slice is not None:
                positive_noise_pred = positive_noise_pred[:, :output_slice]
                negative_noise_pred = negative_noise_pred[:, :output_slice]
            return self.combine_cfg_noise(
                positive_noise_pred,
                negative_noise_pred,
                true_cfg_scale,
                cfg_normalize,
            )

        if output_slice is not None:
            positive_noise_pred = positive_noise_pred[:, :output_slice]
        return positive_noise_pred
