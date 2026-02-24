# Diffusion Advanced Features

## Table of Contents

- [Overview](#overview)
- [Supported Features](#supported-features)
- [Supported Models](#supported-models)
- [Feature Compatibility](#feature-compatibility)
- [Learn More](#learn-more)

## Overview

vLLM-Omni supports various advanced features for diffusion models:
- Acceleration: **cache methods**, **parallelism methods**
- Memory optimization: **cpu offloading**, **quantization**
- Extensions: **LoRA inference**

## Supported Features

### Acceleration

#### Lossy Acceleration

Cache methods trade minimal quality for significant speedup. Quality loss is typically imperceptible with proper tuning.

| Method | Description | Best For |
|--------|-------------|----------|
| **[TeaCache](diffusion/teacache.md)** | Adaptive caching using modulated inputs | Quick setup, balanced quality/speed on single GPU |
| **[Cache-DiT](diffusion/cache_dit.md)** | Multiple caching techniques: DBCache, TaylorSeer, SCM | Fine-grained control, tunable quality-speed tradeoff |


#### Lossless Acceleration

Parallelism methods distribute computation across GPUs without quality loss (mathematically equivalent to single-GPU).

| Method | Description | Best For |
|--------|-------------|----------|
| **[Ulysses-SP](diffusion/sequence_parallel.md)** | Sequence parallelism via all-to-all communication | High-resolution images (>1536px) or long videos with 2-8 GPUs |
| **[Ring-Attention](diffusion/sequence_parallel.md)** | Sequence parallelism via ring-based communication | Videos, very long sequences, memory-constrained, with 2-8 GPUs |
| **[CFG-Parallel](diffusion/cfg_parallel.md)** | Splits CFG positive/negative branches across devices | Image editing with CFG guidance (true_cfg_scale > 1) on 2 GPUs |
| **[Tensor Parallelism](diffusion/tensor_parallel.md)** | Shards model weights across devices | Large models that don't fit in single GPU, with 2+ GPUs |

**Note:** Some acceleration methods can be combined together for optimized performance. See [Feature Compatibility](diffusion/feature_compatibility.md) for detailed configuration examples.

### Memory Optimization

Memory optimization methods help reduce GPU memory usage, enabling inference on resource-constrained hardware or larger models.

| Method | Description | Best For |
|--------|-------------|----------|
| **[CPU Offload](diffusion/cpu_offload_diffusion.md)** | Offloads model components to CPU memory | Limited VRAM, large models on consumer GPUs |
| **[FP8 Quantization](diffusion/quantization/overview.md)** | Reduces DiT linear layers from BF16 to FP8 | Limited VRAM, accuracy preserved    |
| **[VAE Patch Parallelism](diffusion/vae_patch_parallel.md)** | Distributes VAE decode tiling across GPUs | High-resolution generation with reduced VAE memory peak |

### Extensions

Extension methods add specialized capabilities to diffusion models beyond standard inference.

| Method | Description | Best For |
|--------|-------------|----------|
| **[LoRA Inference](diffusion/lora_inference.md)** | Enables inference with Low-Rank Adaptation (LoRA) adapters weights | Reinforcement learning extensions |


### Quantization Methods

| Method | Configuration | Description | Best For |
|--------|--------------|-------------|----------|
| **FP8** | `quantization="fp8"` | FP8 W8A8 on Ada/Hopper, weight-only on older GPUs | Memory reduction, inference speedup |

## Supported Models

The following tables show which models support each acceleration method:
- **Sequence Parallel**: Includes both Ulysses-SP and Ring-Attention methods
- ✅ = Fully supported
- ❌ = Not supported

> Note:
  CPU Offload has two methods: Model-level (default for models with DiT + text encoder) and Layerwise. The tables below show Layerwise support only.

### ImageGen

| Model | Model Identifier | TeaCache | Cache-DiT | Sequence Parallel | CFG-Parallel | Tensor Parallel | CPU Offload (Layerwise) | LoRA Inference | VAE-Patch-Parallel | FP8-Quantization |
|-------|------------------|:----------:|:-----------:|:-----------------:|:------------:|:---------------:|:-----------------------:|:-----------:|:------------------:|:----------------:|
| **Bagel** | `ByteDance-Seed/BAGEL-7B-MoT` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **FLUX.1-dev** | `black-forest-labs/FLUX.1-dev` | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **FLUX.2-klein** | `black-forest-labs/FLUX.2-klein-4B` | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **GLM-Image** | `THUDM/glm-4-vision` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **LongCat-Image** | `meituan-longcat/LongCat-Image` | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **LongCat-Image-Edit** | `meituan-longcat/LongCat-Image-Edit` | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Ovis-Image** | `OvisAI/Ovis-Image` | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Qwen-Image** | `Qwen/Qwen-Image` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Qwen-Image-2512** | `Qwen/Qwen-Image-2512` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Qwen-Image-Edit** | `Qwen/Qwen-Image-Edit` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Qwen-Image-Edit-2509** | `Qwen/Qwen-Image-Edit-2509` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Qwen-Image-Layered** | `Qwen/Qwen-Image-Layered` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Stable-Diffusion3.5** | `stabilityai/stable-diffusion-3.5` | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Z-Image** | `Tongyi-MAI/Z-Image-Turbo` | ✅ | ✅ | ✅ | ❌ | ✅ (TP=2 only) | ❌ | ✅ | ✅ | ✅ |

### VideoGen

| Model | Model Identifier | TeaCache | Cache-DiT | Sequence Parallel | CFG-Parallel | Tensor Parallel | CPU Offload (Layerwise) | LoRA Inference | VAE-Patch-Parallel |
|-------|------------------|:--------:|:---------:|:-----------------:|:------------:|:---------------:|:-----------------------:|:-----------:|:------------------:|
| **Wan2.2-T2V** | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Wan2.2-I2V** | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Wan2.2-TI2V** | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

### AudioGen

| Model | Model Identifier | TeaCache | Cache-DiT | Sequence Parallel | CFG-Parallel | Tensor Parallel | CPU Offload (Layerwise) | LoRA Inference | VAE-Patch-Parallel |
|-------|------------------|:--------:|:---------:|:-----------------:|:------------:|:---------------:|:-----------------------:|:-----------:|:------------------:|
| **Stable-Audio-Open** | `stabilityai/stable-audio-open-1.0` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |


## Feature Compatibility

**Legend:**
- ✅: Fully supported and tested
- ❌: No support plan
- ❓: Not verified yet, help wanted!

|                            | TeaCache | Cache-DiT | Seq Parallel | Tensor Parallel | CFG-Parallel | LoRA Inference | CPU Offloading | FP8 Quantization | VAE Patch Parallel |
|----------------------------|:--------:|:---------:|:------------:|:---------------:|:------------:|:--------------:|:--------------:|:---------:|:------------------:|
| **TeaCache**               |          |           |              |                 |              |                |                |           |                    |
| **Cache-DiT**              |    ❌    |           |              |                 |              |                |                |           |                    |
| **Seq Parallel**           |    ❓    |    ❓     |              |                 |              |                |                |           |                    |
| **Tensor Parallel**        |    ❓    |    ❓     |      ❓      |                 |              |                |                |           |                    |
| **CFG-Parallel**           |    ❓    |    ❓     |      ❓      |       ❓        |              |                |                |           |                    |
| **LoRA Inference**         |    ❓    |    ❓     |      ❓      |       ❓        |      ❓      |                |                |           |                    |
| **CPU Offloading**         |    ❓    |    ❓     |      ❓      |       ❓        |      ❓      |       ❓       |                |           |                    |
| **FP8 Quantization**              |    ❓    |    ❓     |      ❓      |       ❓        |      ❓      |       ❓       |       ❓       |           |                    |
| **VAE Patch Parallel**     |    ❓    |    ❓     |      ❓      |       ❓        |      ❓      |       ❓       |       ❓       |     ❓    |                    |

**Note:** The table shows cross-compatibility between features. If you'd like to help verify untested combinations, please contribute test results to the project!


## Learn More

**Cache Acceleration:**
- **[TeaCache Configuration Guide](diffusion/teacache.md)** - Parameter tuning, performance tips, troubleshooting
- **[Cache-DiT Advanced Guide](diffusion/cache_dit.md)** - DBCache, TaylorSeer, SCM techniques and optimization

**Parallelism Methods:**
- **[Tensor Parallelism Guide](diffusion/tensor_parallel.md)** - Shard DiT weights across GPUs to reduce per-GPU memory
- **[Sequence Parallelism Guide](diffusion/sequence_parallel.md)** - Ulysses-SP and Ring-Attention for long sequence handling
- **[CFG Parallelism Guide](diffusion/cfg_parallel.md)** - Distribute CFG positive/negative branches across GPUs

**Memory Optimization:**
- **[CPU Offload Guide](diffusion/cpu_offload_diffusion.md)** - Offload model components to CPU, reduce GPU memory usage
- **[VAE Patch Parallelism Guide](diffusion/vae_patch_parallel.md)** - Distribute VAE decode tiling across GPUs for high-resolution images
- **[Quantization Overview](diffusion/quantization/overview.md)** - Overview of quantization methods for diffusion models

**Extensions:**
- **[LoRA Inference Guide](diffusion/lora_inference.md)** - Low-Rank Adaptation for style customization and fine-tuning

**Advanced Topics:**
- **[Feature Compatibility](diffusion/feature_compatibility.md)** - How to combine cache + parallelism methods for maximum performance
