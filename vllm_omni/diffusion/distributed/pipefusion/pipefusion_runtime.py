# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project and the xDiT authors.
#
# This module is adapted from xDiT (https://github.com/xdit-project/xdit)

"""
PipeFusion runtime state management.

Tracks patch metadata (sizes, token ranges, split dimension) and
provides the per-patch index counter used by the pipeline, scheduler,
and transformer mixins.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.diffusion_kv.config import PIPEFUSION_KV_MAX_TOKENS
from vllm_omni.diffusion.distributed.parallel_state import get_pipeline_parallel_world_size, get_pp_group

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
    from vllm_omni.diffusion.distributed.pipefusion.offload import PipeFusionKVOffloadManager
    from vllm_omni.diffusion.request import OmniDiffusionRequest

PipeFusionBranchKey = Literal["inputs", "inputs_uncond"]
PipeFusionCacheIdentity = tuple[str | None, int | None, PipeFusionBranchKey]

_PF_RUNTIME: PipeFusionRuntime | None = None


def get_pipefusion_token_capacity(
    latent_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> int:
    """Return the full post-patch token count for one dense KV row."""

    if len(latent_shape) != 3 or len(patch_size) != 3:
        raise ValueError("PipeFusion latent_shape and patch_size must contain (frames, height, width)")
    if any(type(value) is not int or value <= 0 for value in (*latent_shape, *patch_size)):
        raise ValueError("PipeFusion latent dimensions and patch dimensions must be positive integers")
    if any(size % patch != 0 for size, patch in zip(latent_shape, patch_size, strict=True)):
        raise ValueError(f"PipeFusion latent shape {latent_shape!r} must be divisible by patch size {patch_size!r}")
    return latent_shape[0] // patch_size[0] * (latent_shape[1] // patch_size[1]) * (latent_shape[2] // patch_size[2])


def build_pipefusion_kv_requests(
    request_id: str,
    *,
    token_capacity: int,
    sequence_count: int = 1,
    cfg_branches: tuple[PipeFusionBranchKey, ...] = ("inputs",),
    estimated_bytes_per_row: int = 0,
) -> tuple[DiffusionKVRequest, ...]:
    """Build contiguous diffusion_kv rows for PipeFusion sequences/branches."""

    from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest

    if not request_id:
        raise ValueError("PipeFusion request_id must be non-empty")
    if token_capacity <= 0:
        raise ValueError("PipeFusion token_capacity must be positive")
    if sequence_count <= 0:
        raise ValueError("PipeFusion sequence_count must be positive")
    if estimated_bytes_per_row < 0:
        raise ValueError("PipeFusion estimated_bytes_per_row must be non-negative")
    if not cfg_branches or len(cfg_branches) != len(set(cfg_branches)):
        raise ValueError("PipeFusion cfg_branches must be non-empty and unique")

    requests: list[DiffusionKVRequest] = []
    row_id = 0
    for logical_sequence_id in range(sequence_count):
        for branch in cfg_branches:
            if branch not in ("inputs", "inputs_uncond"):
                raise ValueError(f"Invalid PipeFusion cache branch: {branch!r}")
            requests.append(
                DiffusionKVRequest(
                    f"{request_id}:pf:{logical_sequence_id}:{branch}",
                    sequence_id=row_id,
                    logical_sequence_id=logical_sequence_id,
                    cache_branch=branch,
                    prefix_len=0,
                    target_len=token_capacity,
                    seq_len=token_capacity,
                    estimated_bytes=estimated_bytes_per_row,
                )
            )
            row_id += 1
    return tuple(requests)


def attach_pipefusion_kv_requests(
    request: OmniDiffusionRequest,
    od_config: OmniDiffusionConfig,
) -> None:
    """Attach Scheduler-only dense rows using the request's final Wan shape."""

    parallel_config = od_config.parallel_config
    if not getattr(parallel_config, "enable_pipefusion", False):
        return
    sampling = request.sampling_params
    height = sampling.height or 480
    width = sampling.width or 832
    num_frames = sampling.num_frames or 81
    transformer_config = od_config.tf_model_config
    patch_size = tuple(transformer_config.get("patch_size", (1, 2, 2)))
    if len(patch_size) != 3:
        raise ValueError(f"PipeFusion transformer patch_size must have three dimensions, got {patch_size!r}")

    # PipeFusion is currently implemented by the Wan family. Keep these
    # defaults aligned with Wan's VAE, while allowing model_config overrides.
    vae_temporal = int(od_config.model_config.get("vae_scale_factor_temporal", 4))
    vae_spatial = int(od_config.model_config.get("vae_scale_factor_spatial", 8))
    height = height // (vae_spatial * patch_size[1]) * (vae_spatial * patch_size[1])
    width = width // (vae_spatial * patch_size[2]) * (vae_spatial * patch_size[2])
    num_frames = num_frames // vae_temporal * vae_temporal + 1 if num_frames % vae_temporal != 1 else num_frames
    latent_shape = (
        (num_frames - 1) // vae_temporal + 1,
        height // vae_spatial,
        width // vae_spatial,
    )
    token_capacity = get_pipefusion_token_capacity(latent_shape, patch_size)
    if token_capacity > PIPEFUSION_KV_MAX_TOKENS:
        raise ValueError(
            f"PipeFusion request {request.request_id!r} requires {token_capacity} KV tokens per row, "
            f"exceeding the built-in limit of {PIPEFUSION_KV_MAX_TOKENS}"
        )

    # Wan materializes an unconditional embedding for its default guidance
    # even when the caller omits negative_prompt. Every Worker receives the
    # same metadata, including CFG-parallel ranks, so install both identities.
    cfg_branches: tuple[PipeFusionBranchKey, ...] = ("inputs", "inputs_uncond")
    num_layers = int(transformer_config.get("num_layers", 40))
    num_heads = int(transformer_config.get("num_attention_heads", 40))
    head_dim = int(transformer_config.get("attention_head_dim", 128))
    pp_size = max(1, int(getattr(parallel_config, "pipeline_parallel_size", 1)))
    tp_size = max(1, int(getattr(parallel_config, "tensor_parallel_size", 1)))
    local_layers = (num_layers + pp_size - 1) // pp_size
    local_heads = (num_heads + tp_size - 1) // tp_size
    estimated_bytes_per_row = (
        token_capacity
        * 2
        * local_layers
        * local_heads
        * head_dim
        * torch.empty((), dtype=od_config.dtype).element_size()
    )
    # Multiple outputs are carried in the dense tensor's batch dimension; they
    # share one logical sequence row and remain isolated within that tensor.
    required_rows = len(cfg_branches)
    max_rows = getattr(od_config, "diffusion_kv_max_rows_per_request", None) or 2
    if required_rows > max_rows:
        raise ValueError(
            f"PipeFusion request {request.request_id!r} requires {required_rows} managed KV rows, "
            f"exceeding diffusion_kv_max_rows_per_request={max_rows}"
        )
    request.diffusion_kv_requests = build_pipefusion_kv_requests(
        request.request_id,
        token_capacity=token_capacity,
        sequence_count=1,
        cfg_branches=cfg_branches,
        estimated_bytes_per_row=estimated_bytes_per_row,
    )


class PipeFusionRuntime:
    patch_mode: bool
    pipeline_patch_idx: int
    pp_patches_height: list[int] | None
    pp_patches_start_end_idx: list[tuple[int, int]] | None
    pp_patches_token_num: list[int] | None
    pp_patches_token_start_end_idx: list[tuple[int, int]] | None

    def __init__(self):
        self.patch_size: tuple[int, int, int] = (1, 2, 2)
        self.patch_mode = False
        self.pipeline_patch_idx = 0
        self.warmup_steps = 1
        self.split_dim: Literal["height", "temporal"] = "height"
        self.request_id: str | None = None
        self.sequence_id: int | None = 0
        self.cache_key: PipeFusionBranchKey = "inputs"
        self.patch_idx_tensor = torch.tensor(0, dtype=torch.int32)
        self.warmup_cache_timestep: torch.Tensor | None = None
        self.update_warmup_cache = True
        self._kv_offload_manager: PipeFusionKVOffloadManager | None = None
        self._kv_layer_order: list[str] = []
        self._kv_row_binding_resolver: Callable[[str, int, str], Any] | None = None

    @property
    def cache_identity(self) -> PipeFusionCacheIdentity:
        return (self.request_id, self.sequence_id, self.cache_key)

    def set_input_parameters(self, latents: torch.Tensor, dtype):
        self._calc_patches_metadata(latents)
        self._reset_recv_buffer(dtype)

    def set_request_context(self, request_id: str | None, sequence_id: int | None = 0) -> None:
        if request_id is not None and (not isinstance(request_id, str) or not request_id):
            raise ValueError("PipeFusion request_id must be a non-empty string or None.")
        if sequence_id is not None and (not isinstance(sequence_id, int) or sequence_id < 0):
            raise ValueError("PipeFusion sequence_id must be a non-negative integer or None.")
        self.request_id = request_id
        self.sequence_id = sequence_id

    def clear_request_context(self) -> None:
        self.request_id = None
        self.sequence_id = 0

    def set_cache_key(self, key: PipeFusionBranchKey) -> None:
        if key not in ("inputs", "inputs_uncond"):
            raise ValueError(f"Invalid PipeFusion cache key: {key!r}. Must be 'inputs' or 'inputs_uncond'.")
        self.cache_key = key

    def get_kv_offload_manager(self) -> PipeFusionKVOffloadManager:
        if self._kv_offload_manager is None:
            from vllm_omni.diffusion.distributed.pipefusion.offload import PipeFusionKVOffloadManager

            self._kv_offload_manager = PipeFusionKVOffloadManager.from_env()
        return self._kv_offload_manager

    def register_kv_layer(self, layer_id: str) -> None:
        if layer_id not in self._kv_layer_order:
            self._kv_layer_order.append(layer_id)

    def set_kv_row_binding_resolver(
        self,
        resolver: Callable[[str, int, str], Any] | None,
    ) -> None:
        self._kv_row_binding_resolver = resolver

    def validate_kv_row_binding(self, observed_seq_len: int | None = None) -> None:
        """Validate the installed logical row against the runtime token shape."""

        if self._kv_row_binding_resolver is None:
            return
        if self.request_id is None or self.sequence_id is None:
            raise RuntimeError("Managed PipeFusion KV requires an active request/sequence identity")
        binding = self._kv_row_binding_resolver(
            self.request_id,
            self.sequence_id,
            self.cache_key,
        )
        token_capacity = self.ppf * self.pph * self.ppw
        required_capacity = max(token_capacity, observed_seq_len or 0)
        if binding.max_seq_len < required_capacity:
            raise RuntimeError(
                f"Installed PipeFusion KV row {binding.row_index} has capacity {binding.max_seq_len}, "
                f"but runtime requires {required_capacity} tokens"
            )

    def prefetch_following_layers(self, identity: PipeFusionCacheIdentity, layer_id: str) -> None:
        manager = self.get_kv_offload_manager()
        if not manager.enabled or manager.prefetch_layers == 0 or layer_id not in self._kv_layer_order:
            return
        layer_index = self._kv_layer_order.index(layer_id)
        for next_layer in self._kv_layer_order[layer_index + 1 : layer_index + 1 + manager.prefetch_layers]:
            cache_identity = (*identity, next_layer)
            if manager.contains(cache_identity):
                manager.prefetch(cache_identity)

    def set_run_config(self, warmup_steps: int | None = None, split_dim: str | None = None) -> None:
        if warmup_steps is not None:
            if not isinstance(warmup_steps, int) or warmup_steps < 1:
                raise ValueError(f"PipeFusion warmup_steps must be a positive integer, got {warmup_steps!r}.")
            self.warmup_steps = warmup_steps
        if split_dim is not None:
            if split_dim not in ("height", "temporal"):
                raise ValueError(f"Invalid PipeFusion split_dim: {split_dim!r}. Must be 'height' or 'temporal'.")
            self.split_dim = cast(Literal["height", "temporal"], split_dim)

    def set_patched_mode(self, patch_mode: bool):
        self.patch_mode = patch_mode
        self.pipeline_patch_idx = 0
        if hasattr(self, "patch_idx_tensor"):
            self.patch_idx_tensor.fill_(0)

    def next_patch(self):
        if self.patch_mode:
            self.pipeline_patch_idx += 1
            if self.pipeline_patch_idx == self.num_pipeline_patch:
                self.pipeline_patch_idx = 0
        else:
            self.pipeline_patch_idx = 0
        if hasattr(self, "patch_idx_tensor"):
            self.patch_idx_tensor.fill_(self.pipeline_patch_idx)

    def _calc_patch_metadata(self, seq_length):
        lengths = [seq_length // self.num_pipeline_patch] * (self.num_pipeline_patch - 1)
        # Give more tokens to the last patch, if it's the case.
        lengths.append(seq_length // self.num_pipeline_patch + seq_length % self.num_pipeline_patch)
        start = 0
        start_end_idx = []
        for num in lengths:
            start_end_idx.append((start, start + num))
            start += num
        return lengths, start_end_idx

    def _calc_patches_metadata(self, latents):
        self.num_pipeline_patch = get_pipeline_parallel_world_size()

        p_t, p_h, p_w = self.patch_size
        ppf = latents.size(-3) // p_t  # post-patch frames
        pph = latents.size(-2) // p_h  # post-patch height (full)
        ppw = latents.size(-1) // p_w  # post-patch width

        # Store post-patch spatial dims for KV cache reshape in attention
        self.ppf = ppf
        self.pph = pph
        self.ppw = ppw

        # Create 1-element tensors for dynamic shape tracking
        self.ppf_tensor = torch.tensor(ppf, dtype=torch.int64, device=latents.device)
        self.pph_tensor = torch.tensor(pph, dtype=torch.int64, device=latents.device)
        self.ppw_tensor = torch.tensor(ppw, dtype=torch.int64, device=latents.device)

        if self.split_dim == "height":
            # Split along spatial height
            self.latent_split_dim = -2  # dim in 5D [B,C,T,H,W]

            # Post-patch heights split among patches
            self.pp_patches_post_height, self.pp_patches_post_start_end_idx = self._calc_patch_metadata(pph)
            self.pp_patches_post_frames = None  # not split

            # Latent-space heights (multiply by p_h to ensure divisibility)
            self.pp_patches_height = [h * p_h for h in self.pp_patches_post_height]
            start = 0
            self.pp_patches_start_end_idx = []
            for h in self.pp_patches_height:
                self.pp_patches_start_end_idx.append((start, start + h))
                start += h

            # Token count: each patch covers all frames and widths but partial height
            self.pp_patches_token_num = [h * ppw * ppf for h in self.pp_patches_post_height]

        elif self.split_dim == "temporal":
            # Split along temporal (frames) dimension
            self.latent_split_dim = -3  # dim in 5D [B,C,T,H,W]

            # Post-patch frames split among patches
            self.pp_patches_post_frames, self.pp_patches_post_start_end_idx = self._calc_patch_metadata(ppf)
            self.pp_patches_post_height = None  # not split

            # Latent-space frames (multiply by p_t to ensure divisibility)
            self.pp_patches_height = [f * p_t for f in self.pp_patches_post_frames]
            start = 0
            self.pp_patches_start_end_idx = []
            for f in self.pp_patches_height:
                self.pp_patches_start_end_idx.append((start, start + f))
                start += f

            # Token count: each patch covers all heights and widths but partial frames
            self.pp_patches_token_num = [f * pph * ppw for f in self.pp_patches_post_frames]

        else:
            raise ValueError(f"Unknown split_dim: {self.split_dim}. Use 'height' or 'temporal'.")

        # Calculate start/end indices for each patch's tokens
        start = 0
        self.pp_patches_token_start_end_idx = []
        for num in self.pp_patches_token_num:
            self.pp_patches_token_start_end_idx.append((start, start + num))
            start += num

    def _reset_recv_buffer(self, dtype):
        get_pp_group().reset_buffer()
        get_pp_group().set_config(dtype)
        self.patch_idx_tensor = torch.tensor(self.pipeline_patch_idx, dtype=torch.int32, device=get_pp_group().device)


def initialize_pupefusion_runtime():
    global _PF_RUNTIME
    if _PF_RUNTIME is not None:
        RuntimeError("PipeFusion runtime is already initialized...")
    _PF_RUNTIME = PipeFusionRuntime()


def is_pipefusion_initialized() -> bool:
    return _PF_RUNTIME is not None


def get_pipefusion_runtime():
    if _PF_RUNTIME is None:
        raise RuntimeError("PipeFusion runtime has not been initialized.")
    return _PF_RUNTIME


def set_pipefusion_cache_key_if_initialized(key: PipeFusionBranchKey) -> None:
    if _PF_RUNTIME is not None:
        _PF_RUNTIME.set_cache_key(key)
