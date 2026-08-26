# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

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


def tensor_l1_mean(tensor: torch.Tensor, other: torch.Tensor | None = None) -> float:
    if other is not None:
        if tensor.shape != other.shape:
            raise RuntimeError(f"EasyCache tensor shape mismatch: {tuple(tensor.shape)} != {tuple(other.shape)}")
        tensor = tensor - other
    if tensor.numel() == 0:
        raise RuntimeError("cannot compute a mean over an empty tensor")
    return float(tensor.float().abs().mean().item())


def longest_safe_prefix(cumulative_risk: torch.Tensor, threshold: float) -> int:
    if cumulative_risk.ndim != 1:
        raise ValueError("cumulative_risk must be one-dimensional")
    prefix = 0
    for risk in cumulative_risk:
        if float(risk.item()) >= threshold:
            break
        prefix += 1
    return prefix


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
        timestep_value: float,
        step_idx: int,
        pair_key: EasyCachePairKey,
    ) -> torch.Tensor:
        cond = self._branch_state((pair_key[0], pair_key[1], "cond"))
        assert cond.previous_raw_input is not None
        assert cond.previous_raw_output is not None
        assert cond.cache is not None

        input_change_curr = tensor_l1_mean(raw_input, cond.previous_raw_input)
        input_norm_prev = tensor_l1_mean(cond.previous_raw_input)
        input_change_curr_rel = input_change_curr / (input_norm_prev + 1e-8)

        if cond.prev_prev_raw_input is None:
            input_change_prev_rel = 0.0
        else:
            input_change_prev = tensor_l1_mean(cond.previous_raw_input, cond.prev_prev_raw_input)
            input_norm_prev_prev = tensor_l1_mean(cond.prev_prev_raw_input)
            input_change_prev_rel = input_change_prev / (input_norm_prev_prev + 1e-8)

        input_mean_curr = tensor_l1_mean(raw_input)
        input_mean_prev = input_norm_prev

        if cond.prev_prev_raw_output is None:
            output_change_prev_rel = 0.0
        else:
            output_change_prev = tensor_l1_mean(cond.previous_raw_output, cond.prev_prev_raw_output)
            output_norm_prev_prev = tensor_l1_mean(cond.prev_prev_raw_output)
            output_change_prev_rel = output_change_prev / (output_norm_prev_prev + 1e-8)

        cache_norm = tensor_l1_mean(cond.cache)
        residual_norm = cache_norm / (input_mean_curr + 1e-8)

        return torch.tensor(
            [
                timestep_value / 1000.0,
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
        timestep_value: float,
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
        state.cache = (output - raw_input).detach().clone()
        self.stats.calc_forwards += 1
