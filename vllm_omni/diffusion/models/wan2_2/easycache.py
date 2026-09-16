# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
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

FEATURE_DIM = 8
EXPECTED_TARGET_CONTRACT = "cumulative_sum_of_local_cache_errors"
EXPECTED_OUTPUT_CONTRACT = "monotonic_cumulative_risk"

EasyCacheBranch = Literal["cond", "uncond"]
EasyCacheKey = tuple[str, int, EasyCacheBranch]
EasyCachePairKey = tuple[str, int]


@dataclass(frozen=True)
class WanEasyCacheConfig:
    enabled: bool = False
    checkpoint_path: str | None = None
    threshold_override: float | None = None
    warmup_steps: int = 7
    log_stats: bool = False

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.checkpoint_path,
            self.threshold_override,
            self.warmup_steps,
        )


@dataclass
class WanEasyCacheStats:
    prediction_count: int = 0
    plan_count: int = 0
    zero_prefix_count: int = 0
    calc_pairs: int = 0
    skip_pairs: int = 0
    calc_forwards: int = 0
    skip_forwards: int = 0
    prefix_histogram: list[int] = field(default_factory=list)


@dataclass
class _BranchState:
    previous_raw_input: torch.Tensor | None = None
    prev_prev_raw_input: torch.Tensor | None = None
    previous_raw_output: torch.Tensor | None = None
    prev_prev_raw_output: torch.Tensor | None = None
    cache: torch.Tensor | None = None


@dataclass
class _PairState:
    lazy_skip_remaining: int = 0
    lazy_refresh_pending: bool = False
    last_decision: str = "calc"
    last_reason: str = "uninitialized"


class LazyHorizonPredictor(nn.Module):
    """Predict monotonic cumulative cache risk over a future horizon."""

    FEATURE_DIM = FEATURE_DIM

    def __init__(
        self,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        horizon: int = 4,
        initial_prediction: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if initial_prediction <= 0:
            raise ValueError("initial_prediction must be positive")

        self.horizon = int(horizon)
        layers: list[nn.Module] = []
        in_dim = self.FEATURE_DIM
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.horizon))
        self.network = nn.Sequential(*layers)

        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        final_layer = self.network[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, math.log(math.expm1(initial_prediction)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        increments = F.softplus(self.network(features))
        return torch.cumsum(increments, dim=-1)


def _safe_torch_load(path: str, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError("EasyCache checkpoint must contain a dictionary")
    return payload


def load_horizon_predictor(
    checkpoint_path: str,
    device: torch.device,
    threshold_override: float | None,
) -> tuple[LazyHorizonPredictor, torch.Tensor, float, dict[str, Any]]:
    payload = _safe_torch_load(checkpoint_path, device)
    required_keys = {
        "raw_input_state_dict",
        "horizon",
        "threshold",
        "calibration_offsets",
        "target_contract",
        "model_output_contract",
        "args",
    }
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise RuntimeError(
            f"Use lazy_horizon_predictor_full.pt produced by train_lazy_cumulative_risk.py; missing keys: {missing}"
        )

    if payload["target_contract"] != EXPECTED_TARGET_CONTRACT:
        raise RuntimeError(f"Checkpoint target contract mismatch: {payload['target_contract']!r}")
    if payload["model_output_contract"] != EXPECTED_OUTPUT_CONTRACT:
        raise RuntimeError(f"Checkpoint output contract mismatch: {payload['model_output_contract']!r}")
    if int(payload.get("feature_dim", FEATURE_DIM)) != FEATURE_DIM:
        raise RuntimeError("Checkpoint feature dimension must be 8")

    training_args = payload["args"]
    if not isinstance(training_args, dict):
        raise RuntimeError("Checkpoint args metadata must be a dictionary")

    horizon = int(payload["horizon"])
    predictor = LazyHorizonPredictor(
        hidden_dim=int(training_args.get("hidden_dim", 128)),
        num_hidden_layers=int(training_args.get("num_hidden_layers", 3)),
        horizon=horizon,
        initial_prediction=float(training_args.get("initial_prediction", 0.01)),
    ).to(device)
    predictor.load_state_dict(payload["raw_input_state_dict"], strict=True)
    predictor.eval()
    predictor.requires_grad_(False)

    calibration_offsets = torch.as_tensor(payload["calibration_offsets"], dtype=torch.float32, device=device).flatten()
    if calibration_offsets.shape != (horizon,):
        raise RuntimeError(f"calibration_offsets must have shape [{horizon}], got {tuple(calibration_offsets.shape)}")
    if not torch.isfinite(calibration_offsets).all():
        raise RuntimeError("calibration_offsets contain NaN or Inf")
    if (calibration_offsets < 0).any():
        raise RuntimeError("calibration_offsets must be non-negative")
    calibration_offsets = torch.cummax(calibration_offsets, dim=0).values

    threshold = float(payload["threshold"]) if threshold_override is None else float(threshold_override)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("lazy_threshold must be finite and positive")

    return predictor, calibration_offsets, threshold, payload


def _tensor_l1_mean_scalar(tensor: torch.Tensor, other: torch.Tensor | None = None) -> torch.Tensor:
    if other is not None:
        if tensor.shape != other.shape:
            raise RuntimeError(f"EasyCache tensor shape mismatch: {tuple(tensor.shape)} != {tuple(other.shape)}")
        tensor = tensor - other
    if tensor.numel() == 0:
        raise RuntimeError("cannot compute a mean over an empty tensor")
    return tensor.float().abs().mean()


def tensor_l1_mean(tensor: torch.Tensor, other: torch.Tensor | None = None) -> float:
    return float(_tensor_l1_mean_scalar(tensor, other).item())


def longest_safe_prefix(cumulative_risk: torch.Tensor, threshold: float) -> int:
    if cumulative_risk.ndim != 1:
        raise ValueError("cumulative_risk must be one-dimensional")
    # Preserve "stop at the first unsafe value" even for non-monotonic input,
    # while reducing the device-to-host synchronization count to one.
    safe_prefix_mask = torch.cumprod((cumulative_risk < threshold).to(torch.int32), dim=0)
    return int(safe_prefix_mask.sum().item())


class WanEasyCacheState:
    def __init__(
        self,
        *,
        predictor: LazyHorizonPredictor,
        calibration_offsets: torch.Tensor,
        threshold: float,
        warmup_steps: int,
        num_steps: int,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("lazy_warmup_steps must be non-negative")
        self.predictor = predictor
        self.calibration_offsets = calibration_offsets
        self.threshold = threshold
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.branch_states: dict[EasyCacheKey, _BranchState] = {}
        self.pair_states: dict[EasyCachePairKey, _PairState] = {}
        self.stats = WanEasyCacheStats(prefix_histogram=[0 for _ in range(predictor.horizon + 1)])

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

    def _build_features(
        self,
        *,
        raw_input: torch.Tensor,
        timestep_value: float | torch.Tensor,
        step_idx: int,
        pair_key: EasyCachePairKey,
    ) -> torch.Tensor:
        cond = self._branch_state((pair_key[0], pair_key[1], "cond"))
        assert cond.previous_raw_input is not None
        assert cond.previous_raw_output is not None
        assert cond.cache is not None

        zero = torch.zeros((), dtype=torch.float32, device=raw_input.device)
        timestep_scalar = (
            timestep_value.flatten()[0].to(device=raw_input.device, dtype=torch.float32)
            if torch.is_tensor(timestep_value)
            else torch.tensor(timestep_value, dtype=torch.float32, device=raw_input.device)
        )
        if cond.prev_prev_raw_input is None:
            input_change_prev_scalar = zero
            input_norm_prev_prev_scalar = zero
        else:
            input_change_prev_scalar = _tensor_l1_mean_scalar(cond.previous_raw_input, cond.prev_prev_raw_input)
            input_norm_prev_prev_scalar = _tensor_l1_mean_scalar(cond.prev_prev_raw_input)
        if cond.prev_prev_raw_output is None:
            output_change_prev_scalar = zero
            output_norm_prev_prev_scalar = zero
        else:
            output_change_prev_scalar = _tensor_l1_mean_scalar(cond.previous_raw_output, cond.prev_prev_raw_output)
            output_norm_prev_prev_scalar = _tensor_l1_mean_scalar(cond.prev_prev_raw_output)

        (
            timestep_float,
            input_change_curr,
            input_norm_prev,
            input_change_prev,
            input_norm_prev_prev,
            input_mean_curr,
            output_change_prev,
            output_norm_prev_prev,
            cache_norm,
        ) = torch.stack(
            (
                timestep_scalar,
                _tensor_l1_mean_scalar(raw_input, cond.previous_raw_input),
                _tensor_l1_mean_scalar(cond.previous_raw_input),
                input_change_prev_scalar,
                input_norm_prev_prev_scalar,
                _tensor_l1_mean_scalar(raw_input),
                output_change_prev_scalar,
                output_norm_prev_prev_scalar,
                _tensor_l1_mean_scalar(cond.cache),
            )
        ).tolist()

        input_change_curr_rel = input_change_curr / (input_norm_prev + 1e-8)
        input_change_prev_rel = (
            input_change_prev / (input_norm_prev_prev + 1e-8) if cond.prev_prev_raw_input is not None else 0.0
        )
        input_mean_prev = input_norm_prev
        output_change_prev_rel = (
            output_change_prev / (output_norm_prev_prev + 1e-8) if cond.prev_prev_raw_output is not None else 0.0
        )
        residual_norm = cache_norm / (input_mean_curr + 1e-8)

        return torch.tensor(
            [
                timestep_float / 1000.0,
                input_change_curr_rel,
                input_change_prev_rel,
                input_mean_curr,
                input_mean_prev,
                output_change_prev_rel,
                residual_norm,
                step_idx / max(self.num_steps, 1),
            ],
            dtype=torch.float32,
            device=self.calibration_offsets.device,
        )

    def should_skip_pair(
        self,
        *,
        pair_key: EasyCachePairKey,
        raw_input: torch.Tensor,
        timestep_value: float | torch.Tensor,
        step_idx: int,
        do_true_cfg: bool,
    ) -> bool:
        pair = self._pair_state(pair_key)
        history_ready = self._history_ready(pair_key, do_true_cfg)
        force_full_region = step_idx < self.warmup_steps or step_idx >= self.num_steps - 2

        if force_full_region:
            pair.lazy_skip_remaining = 0
            pair.lazy_refresh_pending = False
            pair.last_decision = "calc"
            pair.last_reason = "warmup_or_final"
        elif pair.lazy_skip_remaining > 0:
            if not history_ready:
                raise RuntimeError("A planned EasyCache skip cannot run because branch caches are missing")
            pair.lazy_skip_remaining -= 1
            pair.last_decision = "skip"
            pair.last_reason = "planned_prefix"
        elif pair.lazy_refresh_pending:
            pair.lazy_refresh_pending = False
            pair.last_decision = "calc"
            pair.last_reason = "planned_refresh"
        elif history_ready:
            features = self._build_features(
                raw_input=raw_input,
                timestep_value=timestep_value,
                step_idx=step_idx,
                pair_key=pair_key,
            )
            with torch.no_grad():
                raw_risk = self.predictor(features.unsqueeze(0))[0]
                calibrated_risk = torch.cummax(raw_risk + self.calibration_offsets, dim=0).values

            planned_prefix = longest_safe_prefix(calibrated_risk, self.threshold)
            self.stats.prediction_count += 1
            self.stats.prefix_histogram[planned_prefix] += 1
            if planned_prefix > 0:
                pair.lazy_skip_remaining = planned_prefix - 1
                pair.lazy_refresh_pending = True
                pair.last_decision = "skip"
                pair.last_reason = "new_safe_prefix"
                self.stats.plan_count += 1
            else:
                pair.lazy_skip_remaining = 0
                pair.lazy_refresh_pending = False
                pair.last_decision = "calc"
                pair.last_reason = "zero_safe_prefix"
                self.stats.zero_prefix_count += 1
        else:
            pair.lazy_skip_remaining = 0
            pair.lazy_refresh_pending = False
            pair.last_decision = "calc"
            pair.last_reason = "history_not_ready"

        if pair.last_decision == "skip":
            self.stats.skip_pairs += 1
            return True
        self.stats.calc_pairs += 1
        return False

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
        return raw_input + state.cache.to(device=raw_input.device)

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
        state = self._branch_state(key)
        state.prev_prev_raw_input = state.previous_raw_input
        state.previous_raw_input = raw_input.detach().clone()
        state.prev_prev_raw_output = state.previous_raw_output
        state.previous_raw_output = output.detach().clone()
        # Subtraction already returns fresh storage. Cloning that result again
        # adds a full-latent allocation and copy on every computed branch.
        state.cache = output.detach() - raw_input.detach()
        self.stats.calc_forwards += 1


class WanEasyCacheMixin:
    """Lazy-horizon EasyCache for Wan2.2 T2V/I2V, including PipeFusion PP skip sync."""

    def _init_easycache_state(self) -> None:
        self._easycache_config = WanEasyCacheConfig()
        self._easycache_loaded_signature: tuple[object, ...] | None = None
        self._easycache_predictor = None
        self._easycache_calibration_offsets = None
        self._easycache_threshold: float | None = None
        self._easycache_state: WanEasyCacheState | None = None

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
            "lazy_enabled": extra_args.get("lazy_enabled"),
            "lazy_ckpt": extra_args.get("lazy_ckpt"),
            "lazy_threshold": extra_args.get("lazy_threshold"),
            "lazy_warmup_steps": extra_args.get("lazy_warmup_steps"),
            "lazy_log_stats": extra_args.get("lazy_log_stats"),
        }

    def _resolve_easycache_config(self, sampling_params_list: list[Any]) -> WanEasyCacheConfig:
        first_extra = self._easycache_relevant_extra_args(sampling_params_list[0])
        for sampling_params in sampling_params_list[1:]:
            if self._easycache_relevant_extra_args(sampling_params) != first_extra:
                raise ValueError("Batched Wan2.2 requests must use identical EasyCache extra_args.")

        enabled = self._truthy_extra_arg(first_extra["lazy_enabled"])
        if not enabled:
            return WanEasyCacheConfig()

        checkpoint_path = first_extra["lazy_ckpt"]
        if not checkpoint_path:
            raise ValueError("Wan2.2 EasyCache requires extra_args['lazy_ckpt'] when lazy_enabled is true.")
        if not isinstance(checkpoint_path, str):
            raise TypeError("Wan2.2 EasyCache lazy_ckpt must be a string path.")

        threshold_override = first_extra["lazy_threshold"]
        if threshold_override is not None:
            threshold_override = float(threshold_override)
            if not np.isfinite(threshold_override) or threshold_override <= 0:
                raise ValueError("Wan2.2 EasyCache lazy_threshold must be finite and positive.")

        warmup_steps = 7 if first_extra["lazy_warmup_steps"] is None else int(first_extra["lazy_warmup_steps"])
        if warmup_steps < 0:
            raise ValueError("Wan2.2 EasyCache lazy_warmup_steps must be non-negative.")

        return WanEasyCacheConfig(
            enabled=True,
            checkpoint_path=checkpoint_path,
            threshold_override=threshold_override,
            warmup_steps=warmup_steps,
            log_stats=self._truthy_extra_arg(first_extra["lazy_log_stats"]),
        )

    def _configure_easycache_for_request(self, sampling_params_list: list[Any], num_steps: int) -> None:
        config = self._resolve_easycache_config(sampling_params_list)
        self._easycache_config = config
        self._easycache_state = None
        if not config.enabled:
            return

        if self._safe_cfg_parallel_world_size() > 1:
            raise NotImplementedError("Wan2.2 EasyCache does not yet support CFG parallel.")

        assert config.checkpoint_path is not None
        if getattr(self, "_easycache_loaded_signature", None) != config.signature:
            predictor, calibration_offsets, threshold, _ = load_horizon_predictor(
                config.checkpoint_path,
                self.device,
                config.threshold_override,
            )
            self._easycache_predictor = predictor
            self._easycache_calibration_offsets = calibration_offsets
            self._easycache_threshold = threshold
            self._easycache_loaded_signature = config.signature

        assert self._easycache_predictor is not None
        assert self._easycache_calibration_offsets is not None
        assert self._easycache_threshold is not None
        effective_warmup_steps = config.warmup_steps
        if is_pipefusion_initialized():
            effective_warmup_steps = max(effective_warmup_steps, get_pipefusion_runtime().warmup_steps)
        self._easycache_state = WanEasyCacheState(
            predictor=self._easycache_predictor,
            calibration_offsets=self._easycache_calibration_offsets,
            threshold=self._easycache_threshold,
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
            "predictions=%d, plans=%d, zero_prefix=%d, prefix_histogram=%s, "
            "calc_forwards=%d, skip_forwards=%d",
            stats.calc_pairs,
            stats.skip_pairs,
            100.0 * stats.skip_pairs / max(total_pairs, 1),
            stats.prediction_count,
            stats.plan_count,
            stats.zero_prefix_count,
            stats.prefix_histogram,
            stats.calc_forwards,
            stats.skip_forwards,
        )

    def _release_easycache_request_state(self) -> None:
        """Release per-request histories while retaining predictor weights."""
        self._easycache_state = None

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
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
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
