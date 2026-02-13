# CPU Offload Design Document

This document describes the design and implementation of CPU offloading for diffusion models in vLLM-Omni.

For user guidance documentation, see:
- [CPU Offload User Guide](../../user_guide/diffusion/cpu_offload_diffusion.md)

## Table of Contents

- [Overview](#overview)
- [Architecture Overview](#architecture-overview)
- [Hook System](#hook-system)
- [Backend Architectures](#backend-architectures)
- [Module Discovery](#module-discovery)
- [Backend Selection](#backend-selection)
- [Configuration](#configuration)
- [Model Support](#model-support)
- [Implementation Files](#implementation-files)

## Overview

CPU offload reduces GPU memory usage by transferring model components between CPU and GPU memory during inference. vLLM-Omni provides two complementary offloading strategies:

| Strategy | Description | Best For |
| :--- | :--- | :--- |
| Model-level Offloading | Swaps DiT transformer and encoders between GPU/CPU | Models where DiT + encoders don't fit together |
| Layerwise Offloading | Keeps only one transformer block on GPU at a time | Large video models with high compute-per-block ratio |

Both strategies use pinned memory for faster CPU-GPU transfers and are mutually exclusive.


## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Pipeline Model                      │
│  (e.g., Wan22Pipeline, QwenImagePipeline)           │
└──────────────────┬──────────────────────────────────┘
                   │
                   │ enable()
                   ▼
┌─────────────────────────────────────────────────────┐
│              OffloadBackend                          │
│  (Base class for offloading strategies)             │
├─────────────────────────────────────────────────────┤
│  • discover_modules()  - Find DiT, encoders, VAE    │
│  • register_hooks()    - Install forward hooks      │
│  • enable() / disable()                             │
└──────────────┬──────────────────────────────────────┘
               │
               ├── ModelLevelOffloadBackend
               │   └── SequentialOffloadHook
               │
               └── LayerWiseOffloadBackend
                   └── LayerwiseOffloadHook
```

### Key Components

1. **OffloadBackend**: Abstract base class for offloading strategies
2. **ModelHook**: Base class for forward hooks with pre/post callbacks
3. **HookRegistry**: Manages hook registration and removal on modules
4. **ModuleDiscovery**: Discovers pipeline components (DiT, encoders, VAE)

## Hook System

### Design Principles

The hook system is built on PyTorch's forward hooks but with a more structured approach:

- **Pre-forward hooks**: Execute before module.forward(), can modify inputs
- **Post-forward hooks**: Execute after module.forward(), can modify outputs
- **Hook registry**: Centralized management of all hooks
- **No model modification**: Hooks are external, don't change model code

### Hook Base Class

```python
class ModelHook(ABC):
    """Base class for model hooks with pre/post forward callbacks."""

    _HOOK_NAME: str = "base_hook"

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple, dict]:
        """Called before module.forward(). Can modify inputs."""
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Called after module.forward(). Can modify output."""
        return output

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        """Called once when hook is registered. Can modify module."""
        return module
```

### HookRegistry

Manages hook installation and removal:

```python
class HookRegistry:
    """Centralized registry for all model hooks."""

    def register_hook(self, module: nn.Module, hook: ModelHook) -> None:
        """Register hook on module's forward pass."""

    def remove_hooks(self, module: nn.Module) -> None:
        """Remove all hooks from module."""

    def get_hooks(self, module: nn.Module) -> list[ModelHook]:
        """Get all hooks registered on module."""
```

## Backend Architectures

### Model-Level (Sequential) Offloading

**Strategy**: Mutual exclusion between DiT transformer and encoders.

#### Architecture

```
┌──────────────────────────────────────────────────┐
│  ModelLevelOffloadBackend                        │
├──────────────────────────────────────────────────┤
│  discovers:                                       │
│    • DiT modules (transformer, dit)              │
│    • Encoders (text_encoder*, image_encoder)     │
│    • VAE (stays on GPU)                          │
│                                                   │
│  registers:                                       │
│    • SequentialOffloadHook on DiT modules        │
│    • SequentialOffloadHook on encoder modules    │
└──────────────────────────────────────────────────┘
```

#### SequentialOffloadHook

```python
class SequentialOffloadHook(ModelHook):
    """Hook for sequential offloading with mutual exclusion."""

    def __init__(self,
                 offload_targets: list[nn.Module],  # Modules to offload
                 device: torch.device,               # GPU device
                 pin_memory: bool = True):
        self.offload_targets = offload_targets
        self.device = device
        self.pin_memory = pin_memory

    def pre_forward(self, module, *args, **kwargs):
        # 1. Offload target modules to CPU
        for target in self.offload_targets:
            self._to_cpu(target)

        # 2. Load current module to GPU
        self._to_gpu(module)

        # 3. Synchronize
        torch.cuda.synchronize()

        return args, kwargs
```

**Execution Flow**:

```
Encoder phase:
  pre_forward(text_encoder):
    - DiT → CPU
    - text_encoder → GPU
  forward(text_encoder)

Denoising phase:
  pre_forward(transformer):
    - text_encoder* → CPU
    - transformer → GPU
  forward(transformer) [multiple steps]

VAE decode:
  forward(vae)  # VAE always on GPU
```


### Layerwise (Blockwise) Offloading

**Strategy**: Keep only one transformer block on GPU at a time with async prefetching.

#### Architecture

```
┌─────────────────────────────────────────────────┐
│  LayerWiseOffloadBackend                        │
├─────────────────────────────────────────────────┤
│  discovers:                                      │
│    • Transformer blocks (via _layerwise_        │
│      offload_blocks_attr)                       │
│                                                  │
│  registers:                                      │
│    • LayerwiseOffloadHook on each block         │
│                                                  │
│  keeps on GPU:                                   │
│    • Encoders, VAE, non-block DiT modules       │
└─────────────────────────────────────────────────┘
```

#### LayerwiseOffloadHook

```python
class LayerwiseOffloadHook(ModelHook):
    """Hook for layerwise CPU offloading with async prefetching."""

    def __init__(self,
                 next_block: nn.Module,           # Next block to prefetch
                 device: torch.device,
                 stream: torch.cuda.Stream,       # Async copy stream
                 pin_memory: bool = True):
        self.next_block = next_block
        self.device = device
        self.copy_stream = stream
        self.pin_memory = pin_memory

        # Pre-materialized CPU tensors (flattened, pinned)
        self.dtype_cpu_flattened_weights = {}
        self.dtype_metadata = {}

    def pre_forward(self, module, *args, **kwargs):
        # 1. Wait for previous prefetch to complete
        if self._prefetch_done:
            self._prefetch_done.wait()

        # 2. Prefetch next block (async on copy_stream)
        with torch.cuda.stream(self.copy_stream):
            self._load_to_device(self.next_block)
            self._prefetch_done = self.copy_stream.record_event()

        return args, kwargs

    def post_forward(self, module, output):
        # 3. Free current block memory
        self._offload_to_cpu(module)
        return output
```

**Execution Flow**:

```
Block 0:
  pre_forward:  Prefetch block-1 (async) ┐
  forward:      Compute block-0           ├─ Overlap
  post_forward: Free block-0              ┘

Block 1:
  pre_forward:  Wait for block-1, Prefetch block-2 (async)
  forward:      Compute block-1
  post_forward: Free block-1

...

Block N-1:
  pre_forward:  Wait for block-(N-1), Prefetch block-0 (async)
  forward:      Compute block-(N-1)
  post_forward: Free block-(N-1)
```


#### Weight Consolidation

To maximize H2D bandwidth, parameters are flattened and grouped by dtype:

```python
def _to_cpu(params, buffers, device, pin_memory):
    """Flatten and pin parameters grouped by dtype."""

    dtype_tensors = defaultdict(list)
    dtype_metadata = defaultdict(list)

    # Group by dtype
    for name, param in chain(params.items(), buffers.items()):
        dtype = param.dtype
        dtype_tensors[dtype].append(param.data.flatten())
        dtype_metadata[dtype].append({
            'name': name,
            'shape': param.shape,
            'numel': param.numel(),
        })

    # Concatenate and pin
    dtype_cpu_flattened = {}
    for dtype, tensors in dtype_tensors.items():
        cpu_flat = torch.cat(tensors, dim=0).cpu()
        if pin_memory:
            cpu_flat = cpu_flat.pin_memory()
        dtype_cpu_flattened[dtype] = cpu_flat

    return dtype_cpu_flattened, dtype_metadata
```

## Module Discovery

The offloader automatically discovers pipeline components using naming conventions:

```python
class ModuleDiscovery:
    """Discover diffusion pipeline components."""

    # Component naming patterns
    DIT_NAMES = ["transformer", "transformer_2", "dit"]
    ENCODER_NAMES = ["text_encoder", "text_encoder_2", "text_encoder_3",
                     "image_encoder"]
    VAE_NAMES = ["vae"]

    @staticmethod
    def discover_components(pipeline: nn.Module) -> tuple[list, list, list]:
        """Find DiT, encoders, and VAE in pipeline.

        Returns:
            (dit_modules, encoder_modules, vae_modules)
        """
        dit_modules = []
        encoder_modules = []
        vae_modules = []

        for name, module in pipeline.named_children():
            if name in ModuleDiscovery.DIT_NAMES:
                dit_modules.append(module)
            elif name in ModuleDiscovery.ENCODER_NAMES:
                encoder_modules.append(module)
            elif name in ModuleDiscovery.VAE_NAMES:
                vae_modules.append(module)

        return dit_modules, encoder_modules, vae_modules
```

**Extension**: To support new models, ensure components follow naming conventions or add to discovery patterns.

## Backend Selection

Factory function selects appropriate backend:

```python
def get_offload_backend(config: OffloadConfig,
                        device: torch.device) -> OffloadBackend | None:
    """Get offload backend based on config."""

    if config.strategy == OffloadStrategy.NONE:
        return None
    elif config.strategy == OffloadStrategy.MODEL_LEVEL:
        from .sequential_backend import ModelLevelOffloadBackend
        return ModelLevelOffloadBackend(config, device)
    elif config.strategy == OffloadStrategy.LAYER_WISE:
        from .layerwise_backend import LayerWiseOffloadBackend
        return LayerWiseOffloadBackend(config, device)
    else:
        raise ValueError(f"Unknown offload strategy: {config.strategy}")
```

## Configuration

Offload configuration is extracted from `OmniDiffusionConfig`:

```python
@dataclass
class OffloadConfig:
    strategy: OffloadStrategy
    pin_cpu_memory: bool = True

    @classmethod
    def from_od_config(cls, od_config: OmniDiffusionConfig):
        enable_cpu_offload = od_config.enable_cpu_offload
        enable_layerwise_offload = od_config.enable_layerwise_offload

        # Mutual exclusion: layerwise takes priority
        if enable_layerwise_offload:
            strategy = OffloadStrategy.LAYER_WISE
            if enable_cpu_offload:
                logger.info("Layer-wise takes priority over model-level")
        elif enable_cpu_offload:
            strategy = OffloadStrategy.MODEL_LEVEL
        else:
            strategy = OffloadStrategy.NONE

        return cls(strategy=strategy,
                   pin_cpu_memory=od_config.pin_cpu_memory)
```

## Model Support

### Model-Level Offload

**Requirements**:
- Model must have **both** DiT/transformer module(s) AND encoder module(s)
- Modules discovered via attribute names:
  - DiT: `transformer`, `transformer_2`, `dit`, `language_model`, `transformer_blocks`
  - Encoders: `text_encoder`, `text_encoder_2`, `text_encoder_3`, `image_encoder`
  - VAE: `vae` (optional, stays on GPU if present)

**Validation Logic** (from `sequential_backend.py`):
```python
if not modules.dits:
    logger.warning("No DiT/transformer modules found, skipping model-level offloading")
    return
if not modules.encoders:
    logger.warning("No encoder modules found, skipping model-level offloading")
    return
```

**Supported**:
- ✅ Conditional generation models (text-to-image/video with text encoders)
- ✅ Wan22Pipeline, QwenImagePipeline, FluxPipeline, SD3Pipeline
- ❌ Unconditional models (no text encoder requirement)
- ❌ Models with custom attribute names not in discovery list

**Graceful Degradation**: If requirements not met, logs warning and skips offloading (model still runs)

### Layerwise Offload

**Requirements**: DiT class must define `_layerwise_offload_blocks_attr`:

```python
class WanTransformer3DModel(nn.Module):
    _layerwise_offload_blocks_attr = "blocks"  # Attribute name for transformer blocks

    def __init__(self):
        self.blocks = nn.ModuleList([
            TransformerBlock(...) for _ in range(num_layers)
        ])
```

**Currently Supported**:
- `WanTransformer3DModel` (`blocks`)
- `QwenImageTransformer2DModel` (`transformer_blocks`)

**To Add Support**: Define `_layerwise_offload_blocks_attr` pointing to `nn.ModuleList` of transformer blocks.


## Implementation Files

```
vllm_omni/diffusion/offloader/
├── __init__.py              # Public API (get_offload_backend)
├── base.py                  # OffloadBackend, OffloadConfig, OffloadStrategy
├── sequential_backend.py    # ModelLevelOffloadBackend, SequentialOffloadHook
├── layerwise_backend.py     # LayerWiseOffloadBackend, LayerwiseOffloadHook
└── module_collector.py      # ModuleDiscovery

vllm_omni/diffusion/hooks.py  # HookRegistry, ModelHook
```
