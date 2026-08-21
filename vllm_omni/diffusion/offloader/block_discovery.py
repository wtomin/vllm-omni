# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block discovery for layerwise offload.

Shared between LayerWiseOffloadBackend and DistributedLayerwiseOffloadBackend.
"""

from __future__ import annotations

from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import PPMissingLayer

logger = init_logger(__name__)


def get_blocks_attr_names(model: nn.Module) -> list[str]:
    """Get block attribute names from model class."""
    attrs: list[str] = getattr(model.__class__, "_layerwise_offload_blocks_attrs", [])

    if not attrs:
        old_attr = getattr(model.__class__, "_layerwise_offload_blocks_attr", None)
        if old_attr is not None:
            logger.warning(
                "'_layerwise_offload_blocks_attr' is deprecated, "
                "please use '_layerwise_offload_blocks_attrs' instead. "
                "Example: _layerwise_offload_blocks_attrs = ['blocks']"
            )
            attrs = [old_attr] if isinstance(old_attr, str) else list(old_attr)

    return attrs


def set_blocks_attr_names(model: nn.Module, names: list[str]) -> None:
    if not hasattr(model.__class__, "_layerwise_offload_blocks_attrs"):
        setattr(model.__class__, "_layerwise_offload_blocks_attrs", names)


def _iter_local_blocks(model: nn.Module, attr_name: str, attr: object) -> list[nn.Module]:
    try:
        blocks = list(iter(attr))  # type: ignore[arg-type]
    except TypeError:
        if isinstance(attr, nn.Module):
            logger.warning(
                "Attribute '%s' on %s is not iterable; treating it as one block.",
                attr_name,
                model.__class__.__name__,
            )
            return [attr]

        logger.warning(
            "Attribute '%s' on %s is not iterable (got %s); skipping it.",
            attr_name,
            model.__class__.__name__,
            type(attr).__name__,
        )
        return []

    start_layer = getattr(model, "start_layer", None)
    end_layer = getattr(model, "end_layer", None)
    if (
        attr_name in {"blocks", "layers"}
        and isinstance(start_layer, int)
        and isinstance(end_layer, int)
        and 0 <= start_layer <= end_layer <= len(blocks)
    ):
        blocks = blocks[start_layer:end_layer]

    return [block for block in blocks if isinstance(block, nn.Module) and not isinstance(block, PPMissingLayer)]


def get_blocks_from_dit(model: nn.Module) -> tuple[list[str], list[nn.Module]]:
    """Retrieve blocks and attribute names from provided DiT model."""
    blocks_attr_names = get_blocks_attr_names(model)
    if not blocks_attr_names:
        logger.warning(
            f"No _layerwise_offload_blocks_attrs defined for {model.__class__.__name__}, "
            "skipping distributed layerwise offloading"
        )
        return [], []

    blocks: list[nn.Module] = []
    for name in blocks_attr_names:
        attr = getattr(model, name, None)
        if attr is None:
            raise AttributeError(
                f"Attribute '{name}' declared in _layerwise_offload_blocks_attrs "
                f"does not exist on model {model.__class__.__name__}"
            )
        blocks.extend(_iter_local_blocks(model, name, attr))

    if not blocks:
        logger.warning(
            "No blocks found in %s for %s, skipping distributed layerwise offloading",
            blocks_attr_names,
            model.__class__.__name__,
        )
        return [], []

    return blocks_attr_names, blocks
