# CPU Offload Guide


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

CPU Offload reduces GPU memory usage for diffusion models by transferring model components between CPU and GPU memory during inference. vLLM-Omni provides two complementary offloading strategies:

1. **Model-level Offloading**: Swaps DiT transformer and encoders between GPU/CPU - only one is on GPU at a time
2. **Layerwise Offloading**: Keeps only one transformer block on GPU at a time with compute-memory overlap

Both strategies use pinned memory for faster CPU-GPU transfers. The strategies are **mutually exclusive** - if both are enabled, layerwise takes priority.

**Key Benefits**:
- Enables running large models on limited VRAM (e.g., consumer GPUs)
- Reduces peak memory usage by ~50-70% depending on strategy
- Particularly effective for video generation models with high compute-per-block ratio

See supported models list in [Supported Models](../diffusion_features.md#supported-models).

!!! note "Two Offloading Strategies"
    **Model-level (Sequential) Offloading**:
    - Mutual exclusion between DiT and encoders
    - VAE stays on GPU
    - Good for: Models where encoder+DiT don't fit together

    **Layerwise (Blockwise) Offloading**:
    - Only one transformer block on GPU at a time
    - Overlaps weight transfer with computation
    - Good for: Large video generation models with high compute cost per block

---

## Quick Start

### Basic Usage

**Model-level Offload**

Simplest working example for model-level offloading:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    enable_cpu_offload=True,  # Enable model-level offload
)

outputs = omni.generate(
    "a cat playing with a ball",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
        height=480,
        width=720,
    ),
)
```


**Layerwise Offload**

For large video models with better compute-memory overlap:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    enable_layerwise_offload=True,  # Enable layerwise offload
)

outputs = omni.generate(
    "a beautiful sunset over mountains",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

---

## Example Script

### Offline Inference


**Model-level Offload:**
```bash
# Text-to-Video with model-level offload
python examples/offline_inference/text_to_video/text_to_video.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "a cat playing with a ball" \
    --enable-cpu-offload

# Image generation with model-level offload
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a futuristic city" \
    --enable-cpu-offload
```

**Layerwise Offload:**
```bash
# Text-to-Video with layerwise offload (better for large video models)
python examples/offline_inference/text_to_video/text_to_video.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "a cat playing with a ball" \
    --enable-layerwise-offload

# Image-to-Video
python examples/offline_inference/image_to_video/image_to_video.py \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --image input.png \
    --prompt "make it move" \
    --enable-layerwise-offload
```

### Online Serving

Enable CPU offload in online serving:

```bash
# Model-level offload
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --enable-cpu-offload

# Layerwise offload
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --enable-layerwise-offload
```

---

## Configuration Parameters

### Model-level Offload

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_cpu_offload` | bool | False | Enable model-level (sequential) offloading between DiT and encoders |

### Layerwise Offload

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_layerwise_offload` | bool | False | Enable layerwise (blockwise) offloading for transformer blocks |

!!! warning "Mutual Exclusivity"
    If both `enable_cpu_offload` and `enable_layerwise_offload` are set to `True`, **layerwise offload takes priority** and model-level offload is ignored.

---

## Best Practices

### When to Use Model-level Offload

**Good for:**

- Models where DiT + encoders exceed available VRAM
- Relatively small models where transfer overhead is acceptable
- Single GPU setups with limited memory (e.g., RTX 3060 12GB)

**Not for:**

- Models that already fit in VRAM (adds unnecessary overhead)
- Real-time or latency-sensitive applications
- Scenarios where encoder and DiT can fit together on GPU

### When to Use Layerwise Offload

**Good for:**

- Models with high compute cost per transformer block
- Scenarios where compute-memory overlap is effective
- Maximum memory savings needed (only 1 block on GPU at a time)

**Not for:**

- Small models with fast block execution (overhead dominates)
- Low H2D bandwidth (e.g., PCIe 3.0 or older)
- Models without proper block attribute definition


### Memory Optimization Tips

1. **Combine with other methods**:
```python
omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    enable_layerwise_offload=True,
    gpu_memory_utilization=0.9,  # Use more GPU memory
)
```

2. **Profile your model**: Use [profiler](../../contributing/profiling.md) to check if compute overlaps with transfers
3. **Use pinned memory**: Automatically enabled for faster CPU-GPU transfers

---

## Troubleshooting

### Common Issue 1: Slower than Expected

**Symptoms**: Inference is much slower with offloading enabled

**Possible Causes & Solutions**:

1. **Model already fits in VRAM**:
   - **Cause**: Offloading adds overhead when not needed
   - **Solution**: Disable offloading if model fits in memory

2. **Transfer overhead dominates**:
   - **Cause**: Small blocks, fast computation, slow PCIe
   - **Solution**: Use model-level offload instead of layerwise


### Common Issue 2: Out of Memory (OOM)

**Symptoms**: Still getting CUDA OOM errors even with offloading enabled

**Possible Causes & Solutions**:

1. **VAE decode still too large**:
   - **Cause**: VAE stays on GPU for both offload strategies
   - **Solution**: Reduce resolution, or use VAE tiling, or use vae patch parallel

```python
omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    enable_layerwise_offload=True,
    vae_use_tiling=True,  # Enable VAE tiling
)
```

2. **Peak memory during transfers**:
   - **Cause**: Brief spike when both modules are on GPU
   - **Solution**: Reduce `gpu_memory_utilization`

```python
omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    enable_layerwise_offload=True,
    gpu_memory_utilization=0.7,  # Leave more headroom
)
```


### Common Issue 3: Model Not Supported

**For Model-Level Offload (`enable_cpu_offload=True`):**

**Symptoms**: Warning "No DiT/transformer modules found" or "No encoder modules found, skipping model-level offloading"

**Root Cause**: Model-level offload requires **BOTH** DiT and encoder modules with specific attribute names:

- **DiT modules** (≥1 required): `transformer`, `transformer_2`, `dit`, `language_model`, `transformer_blocks`
- **Encoder modules** (≥1 required): `text_encoder`, `text_encoder_2`, `text_encoder_3`, `image_encoder`

**Unsupported**: Unconditional models (no text encoder) or models with custom attribute names

**Solutions**:
1. Use a supported model - check [Supported Models](../diffusion_features.md#supported-models)
2. For custom models, ensure attributes match the discovery patterns above
3. If model can't be adapted, offloading won't work

---

**For Layerwise Offload (`enable_layerwise_offload=True`):**

**Symptoms**: Error about missing `_layerwise_offload_blocks_attr`

**Cause**: Layerwise offload requires model to define transformer blocks attribute

**Solutions**:

1. **Use model-level offload instead**:
```python
omni = Omni(
    model="YourModel",
    enable_cpu_offload=True,  # Falls back to model-level
)
```

2. **Add support to model class**:
```python
class YourTransformer(nn.Module):
    _layerwise_offload_blocks_attr = "blocks"  # Point to transformer blocks

    def __init__(self):
        self.blocks = nn.ModuleList([...])  # Your transformer blocks
```

3. **Check supported models**: Refer to [Supported Models](../diffusion_features.md#supported-models)

---

## Summary

1. ✅ **Choose the right strategy** - Model-level for most cases, layerwise for large video models
2. ✅ **Use when needed** - Only enable offloading if model doesn't fit in VRAM
3. ✅ **Profile performance** - Measure latency impact vs memory savings
4. ✅ **Combine techniques** - Use with VAE tiling for maximum memory savings
5. ⚠️ **Check model support** - Ensure your model is supported, especially for layerwise offload
