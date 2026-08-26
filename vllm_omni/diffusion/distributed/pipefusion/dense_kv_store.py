# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_omni.diffusion.distributed.pipefusion.offload import PipeFusionKVOffloadManager
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_runtime import PipeFusionCacheIdentity

PipeFusionLayerCacheIdentity = tuple[str | None, int | None, str, str]


@dataclass(frozen=True)
class PipeFusionPatchLayout:
    split_dim: str
    ppf: int
    pph: int
    ppw: int
    num_pipeline_patch: int


class PipeFusionDenseKVStore:
    """Dense, layer-aware K/V storage independent of the attention kernel."""

    def __init__(self, offload_manager: PipeFusionKVOffloadManager | None = None) -> None:
        self.offload_manager = offload_manager or PipeFusionKVOffloadManager.from_env()

    @staticmethod
    def layer_identity(
        identity: PipeFusionCacheIdentity,
        layer_id: str,
    ) -> PipeFusionLayerCacheIdentity:
        return identity[0], identity[1], identity[2], layer_id

    def put_full(
        self,
        identity: PipeFusionCacheIdentity,
        layer_id: str,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.offload_manager.put(self.layer_identity(identity, layer_id), key, value)

    def get_full(
        self,
        identity: PipeFusionCacheIdentity,
        layer_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_identity = self.layer_identity(identity, layer_id)
        try:
            return self.offload_manager.get(layer_identity)
        except KeyError as exc:
            raise RuntimeError(
                f"PipeFusion KV cache for {identity!r} at layer {layer_id!r} is missing. "
                "Run at least one warmup step before patched execution."
            ) from exc

    def update_patch(
        self,
        identity: PipeFusionCacheIdentity,
        layer_id: str,
        patch_id: int | torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        layout: PipeFusionPatchLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_key, full_value = self.get_full(identity, layer_id)
        patch_index = int(patch_id.item()) if isinstance(patch_id, torch.Tensor) else patch_id
        if not 0 <= patch_index < layout.num_pipeline_patch:
            raise ValueError(f"Invalid PipeFusion patch id {patch_index}")

        batch, _, heads, dim = key.shape
        split_size = layout.ppf if layout.split_dim == "temporal" else layout.pph
        base_patch_size = split_size // layout.num_pipeline_patch
        patch_start = patch_index * base_patch_size
        patch_end = patch_start + base_patch_size
        if patch_index == layout.num_pipeline_patch - 1:
            patch_end = split_size

        if layout.split_dim == "temporal":
            token_start = patch_start * layout.pph * layout.ppw
            token_len = (patch_end - patch_start) * layout.pph * layout.ppw
            full_key.narrow(1, token_start, token_len).copy_(key)
            full_value.narrow(1, token_start, token_len).copy_(value)
        elif layout.split_dim == "height":
            patch_height = patch_end - patch_start
            key_5d = key.reshape(batch, layout.ppf, patch_height, layout.ppw, heads, dim)
            value_5d = value.reshape(batch, layout.ppf, patch_height, layout.ppw, heads, dim)
            full_key.view(batch, layout.ppf, layout.pph, layout.ppw, heads, dim).narrow(
                2, patch_start, patch_height
            ).copy_(key_5d)
            full_value.view(batch, layout.ppf, layout.pph, layout.ppw, heads, dim).narrow(
                2, patch_start, patch_height
            ).copy_(value_5d)
        else:
            raise ValueError(f"Unsupported PipeFusion split dimension: {layout.split_dim!r}")
        return full_key, full_value

    def offload(self, identity: PipeFusionCacheIdentity, layer_id: str) -> bool:
        layer_identity = self.layer_identity(identity, layer_id)
        if not self.offload_manager.contains(layer_identity):
            return False
        return self.offload_manager.offload(layer_identity)

    def prefetch(self, identity: PipeFusionCacheIdentity, layer_id: str) -> bool:
        layer_identity = self.layer_identity(identity, layer_id)
        if not self.offload_manager.contains(layer_identity):
            return False
        return self.offload_manager.prefetch(layer_identity)

    def reset_request(self, request_id: str, sequence_id: int | None = None) -> int:
        return self.offload_manager.remove_request(request_id, sequence_id)

    def clear(self) -> None:
        self.offload_manager.clear()

    def cache_view(self, layer_id: str) -> dict[PipeFusionCacheIdentity, tuple[torch.Tensor, torch.Tensor]]:
        view: dict[PipeFusionCacheIdentity, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_identity, entry in self.offload_manager.items():
            if not isinstance(layer_identity, tuple) or len(layer_identity) != 4 or layer_identity[3] != layer_id:
                continue
            if entry.key is not None and entry.value is not None:
                view[layer_identity[:3]] = (entry.key, entry.value)
        return view
