# VAE Patch Parallelism Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

VAE Patch Parallelism distributes the VAE (Variational AutoEncoder) decode/encode computation across multiple GPUs by splitting the latent space into spatial tiles or patches. Each GPU processes a subset of tiles in parallel, significantly reducing peak memory consumption during the VAE decode stage while maintaining output quality.

This is particularly useful for:
- **High-resolution image generation** where VAE decode can become a memory bottleneck
- **Memory-constrained environments** where the VAE decode activation peak exceeds available VRAM
- **Multi-GPU setups** where you want to leverage distributed resources for the VAE stage

See supported models list in [Supported Models](../diffusion_features.md#supported-models).


VAE Patch Parallelism uses two strategies based on image size:

| Strategy | Use Case | How It Works | Overlap Handling | Output Quality |
|----------|----------|--------------|------------------|----------------|
| **Tiled Decode** | Large images (triggers VAE tiling) | Distributes existing VAE tiling computation across ranks. Each rank decodes a subset of overlapping tiles. | Uses VAE's native `blend_v` and `blend_h` functions to seamlessly merge overlapping regions | Bit-identical (same logic as single-GPU tiling) |
| **Patch Decode** | Small images (no VAE tiling) | Splits latent into spatial patches with halos. Each rank decodes one patch with boundary context. | Halo regions provide edge context; core regions are directly stitched without blending | Near-identical (diff < 0.5%, visually imperceptible) |


VAE Patch Parallelism **reuses the DiT process group** (`dit_group`) and does not initialize a separate ProcessGroup. This means:

- **Shared ranks**: VAE patch parallelism uses the same GPU ranks as DiT parallelism (Tensor Parallel, Sequence Parallel, etc.)
- **Combined usage**: VAE patch parallelism is typically used together with other parallelism methods
- **Configuration alignment**: The `vae_patch_parallel_size` should be no greater than the size of your DiT process group

---

## Quick Start

### Basic Usage

Simplest working example:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

# TP=2 for DiT, VAE patch parallel also uses these 2 GPUs
omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    parallel_config=DiffusionParallelConfig(
        tensor_parallel_size=2,          # Enable tensor parallelism for DiT
        vae_patch_parallel_size=2,       # Enable VAE patch parallelism
    ),
    vae_use_tiling=True,  # Required for VAE patch parallelism
)

outputs = omni.generate(
    "a futuristic city at sunset, high resolution, 8k",
    OmniDiffusionSamplingParams(
        num_inference_steps=9,
        height=1152,  # High resolution benefits from VAE patch parallel
        width=1152,
    ),
)
```

---

## Example Script

### Offline Inference

Use Python script under `examples/offline_inference/text_to_image/`:

```bash
# Text-to-Image with Z-Image
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Tongyi-MAI/Z-Image-Turbo \
    --prompt "a futuristic city at sunset" \
    --height 1152 \
    --width 1152 \
    --tensor-parallel-size 2 \
    --vae-patch-parallel-size 2 \
    --vae-use-tiling
```

### Online Serving

Online serving with VAE patch parallelism is not supported yet.

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vae_patch_parallel_size` | int | 1 | Number of GPUs for VAE patch/tile parallelism. Set to 2 or higher to enable. Should typically match `tensor_parallel_size` as they share the same process group. |

Additional requirements:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vae_use_tiling` | bool | False | Must be set to `True` when using VAE patch parallelism. Automatically enabled if not set (for allowlisted models). |

!!! note "Automatic VAE Tiling"
    When `vae_patch_parallel_size > 1` and the model is in the allowlist, the system automatically sets `vae_use_tiling=True` if not already enabled.

---

## Best Practices

### When to Use

**Good for:**
- High-resolution image generation (≥1024x1024, especially ≥1152x1152)
- Memory-constrained setups where VAE decode causes OOM
- Multi-GPU environments with tensor parallelism enabled
- Z-Image model with TP=2 (validated configuration)
- Reducing VAE decode peak memory usage without sacrificing quality

**Not for:**
- Low-resolution images (<1024x1024) where VAE decode is not a bottleneck
- Single GPU setups (adds unnecessary communication overhead)
- Models not in the allowlist (currently only Z-Image is validated)
- Scenarios where VAE decode memory is not a concern


### Expected Performance

| Configuration | Memory Reduction | Quality Impact | Use Case |
|--------------|------------------|----------------|----------|
| No VAE PP (single GPU) | Baseline | Perfect | VAE decode fits in memory |
| VAE PP=2 | ~50% VAE peak | Negligible* | High-res generation, memory-constrained |
| VAE PP=4 | ~75% VAE peak | Negligible* | Ultra-high-res, extreme memory constraints |

---

## Troubleshooting

### Common Issue 1: Model Not in Allowlist

**Symptoms**:
```
WARNING: vae_patch_parallel_size=2 is set but VAE patch parallelism is only enabled for xxxPipeline; ignoring.
```

**Root Cause**: VAE Patch Parallelism uses a **registry allowlist** mechanism to ensure only validated models can use this feature. The allowlist is defined in `vllm_omni/diffusion/registry.py`:

```python
_VAE_PATCH_PARALLEL_ALLOWLIST = {
    "ZImagePipeline",  # Tongyi-MAI/Z-Image-Turbo
}
```


**Solutions**:

1. **Use a supported model** (recommended):

2. Waiting for updates (contributings are welcomed).


### Common Issue 2: `vae_patch_parallel_size` Exceeds DiT Process Group Size

**Symptoms**: Shows warning message, and vae patch parallel size is resized to DiT process group size

**Root Cause**: VAE Patch Parallelism reuses the DiT process group.

**Recommendation**: Always set `vae_patch_parallel_size` to be no greater than your DiT process group size.

Note that the size of DiT process group size equals to:
```text
dit_parallel_size = data_parallel_size
                  × cfg_parallel_size
                  × sequence_parallel_size
                  × pipeline_parallel_size
                  × tensor_parallel_size

```
_sequence_parallel_size = ulysses_degree × ring_degree_

---

## Summary

1. ✅ **Enable VAE Patch Parallelism** - Set `vae_patch_parallel_size`， `vae_use_tiling=True` in `DiffusionParallelConfig` to reduce VAE decode peak memory
2. ✅ **Use High Sequence** - VAE PP benefits are most apparent at ≥1152x1152
4. ✅ **Combine with TP** - Use together with `tensor_parallel_size=2` for maximum memory savings
5. ⚠️ **Check Model Support** - Currently validated for Z-Image only; verify in [supported models](../diffusion_features.md#supported-models)
