# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from vllm_omni.diffusion.diffusion_kv.config import PIPEFUSION_KV_MAX_CPU_BYTES
from vllm_omni.platforms import current_omni_platform


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class PipeFusionKVResidency(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    PREFETCHING = "prefetching"


@dataclass
class PipeFusionOffloadEntry:
    key: torch.Tensor | None
    value: torch.Tensor | None
    cpu_key: torch.Tensor | None = None
    cpu_value: torch.Tensor | None = None
    residency: PipeFusionKVResidency = PipeFusionKVResidency.GPU
    device: torch.device | None = None

    @property
    def num_bytes(self) -> int:
        tensors = (
            (self.key, self.value)
            if self.residency is PipeFusionKVResidency.GPU
            else (
                self.cpu_key,
                self.cpu_value,
            )
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not None)


class PipeFusionKVOffloadManager:
    """Pinned-CPU tier for dense PipeFusion K/V tensors.

    The first implementation synchronizes before releasing source tensors. This
    is conservative around attention and pipeline sends; dedicated streams are
    still used so overlap can be introduced without changing the state model.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        pin_memory: bool = True,
        prefetch_layers: int = 1,
    ) -> None:
        if prefetch_layers < 0:
            raise ValueError("prefetch_layers must be non-negative")
        self.enabled = enabled
        self.pin_memory = pin_memory
        self.prefetch_layers = prefetch_layers
        self._entries: dict[object, PipeFusionOffloadEntry] = {}
        self._offload_stream: Any | None = None
        self._prefetch_stream: Any | None = None

    @classmethod
    def from_env(cls) -> PipeFusionKVOffloadManager:
        raw_window = os.environ.get("VLLM_OMNI_PIPEFUSION_KV_CACHE_PREFETCH_LAYERS", "1")
        try:
            prefetch_layers = int(raw_window)
        except ValueError as exc:
            raise ValueError("VLLM_OMNI_PIPEFUSION_KV_CACHE_PREFETCH_LAYERS must be an integer") from exc
        return cls(
            enabled=_env_flag("VLLM_OMNI_PIPEFUSION_KV_CACHE_OFFLOAD", False),
            pin_memory=_env_flag("VLLM_OMNI_PIPEFUSION_KV_CACHE_PIN_MEMORY", True),
            prefetch_layers=prefetch_layers,
        )

    def put(self, identity: object, key: torch.Tensor, value: torch.Tensor) -> None:
        self._entries[identity] = PipeFusionOffloadEntry(
            key=key,
            value=value,
            device=key.device,
        )

    def get(self, identity: object) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            entry = self._entries[identity]
        except KeyError as exc:
            raise KeyError(identity) from exc
        if entry.residency is not PipeFusionKVResidency.GPU:
            self.prefetch(identity)
        assert entry.key is not None and entry.value is not None
        return entry.key, entry.value

    def offload(self, identity: object) -> bool:
        if not self.enabled:
            return False
        entry = self._entries[identity]
        if entry.residency is not PipeFusionKVResidency.GPU:
            return False
        assert entry.key is not None and entry.value is not None
        requested_bytes = entry.num_bytes
        if self.cpu_bytes + requested_bytes > PIPEFUSION_KV_MAX_CPU_BYTES:
            raise RuntimeError(
                f"PipeFusion KV CPU offload budget exceeded: requested={requested_bytes}, "
                f"resident={self.cpu_bytes}, limit={PIPEFUSION_KV_MAX_CPU_BYTES}"
            )
        if entry.key.device.type == "cuda":
            current_omni_platform.current_stream().synchronize()
            if self._offload_stream is None:
                self._offload_stream = current_omni_platform.Stream()
            with current_omni_platform.stream(self._offload_stream):
                entry.cpu_key = self._to_cpu(entry.key)
                entry.cpu_value = self._to_cpu(entry.value)
            self._offload_stream.synchronize()
        else:
            entry.cpu_key = entry.key.clone()
            entry.cpu_value = entry.value.clone()
        entry.key = None
        entry.value = None
        entry.residency = PipeFusionKVResidency.CPU
        return True

    def prefetch(self, identity: object) -> bool:
        entry = self._entries[identity]
        if entry.residency is PipeFusionKVResidency.GPU:
            return False
        assert entry.cpu_key is not None and entry.cpu_value is not None
        entry.residency = PipeFusionKVResidency.PREFETCHING
        target = entry.device or torch.device("cpu")
        if target.type == "cuda":
            if self._prefetch_stream is None:
                self._prefetch_stream = current_omni_platform.Stream()
            with current_omni_platform.stream(self._prefetch_stream):
                entry.key = entry.cpu_key.to(target, non_blocking=self.pin_memory)
                entry.value = entry.cpu_value.to(target, non_blocking=self.pin_memory)
            self._prefetch_stream.synchronize()
        else:
            entry.key = entry.cpu_key.clone()
            entry.value = entry.cpu_value.clone()
        entry.cpu_key = None
        entry.cpu_value = None
        entry.residency = PipeFusionKVResidency.GPU
        return True

    def remove(self, identity: object) -> None:
        self._entries.pop(identity, None)

    def remove_request(self, request_id: str, sequence_id: int | None = None) -> int:
        identities = [
            identity
            for identity in self._entries
            if isinstance(identity, tuple)
            and len(identity) >= 2
            and identity[0] == request_id
            and (sequence_id is None or identity[1] == sequence_id)
        ]
        for identity in identities:
            del self._entries[identity]
        return len(identities)

    def clear(self) -> None:
        self._entries.clear()

    def residency(self, identity: object) -> PipeFusionKVResidency:
        return self._entries[identity].residency

    def contains(self, identity: object) -> bool:
        return identity in self._entries

    def items(self):
        return self._entries.items()

    @property
    def gpu_bytes(self) -> int:
        return sum(entry.num_bytes for entry in self._entries.values() if entry.residency is PipeFusionKVResidency.GPU)

    @property
    def cpu_bytes(self) -> int:
        return sum(
            entry.num_bytes for entry in self._entries.values() if entry.residency is not PipeFusionKVResidency.GPU
        )

    def _to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        cpu_tensor = torch.empty_like(
            tensor,
            device="cpu",
            pin_memory=self.pin_memory and current_omni_platform.is_available(),
        )
        cpu_tensor.copy_(tensor, non_blocking=self.pin_memory)
        return cpu_tensor
