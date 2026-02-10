# Diffusion Advanced Features

## Table of Contents

- [Overview](#overview)
- [Supported Features](#supported-features)
- [Supported Models](#supported-models)
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
| **[Ulysses-SP](diffusion/parallelism_acceleration.md#ulysses-sp)** | Sequence parallelism via all-to-all communication | High-resolution images (>1536px) or long videos with 2-8 GPUs |
| **[Ring-Attention](diffusion/parallelism_acceleration.md#ring-attention)** | Sequence parallelism via ring-based communication | Videos, very long sequences, memory-constrained, with 2-8 GPUs |
| **[CFG-Parallel](diffusion/parallelism_acceleration.md#cfg-parallel)** | Splits CFG positive/negative branches across devices | Image editing with CFG guidance (true_cfg_scale > 1) on 2 GPUs |
| **[Tensor Parallelism](diffusion/parallelism_acceleration.md#tensor-parallel)** | Shards model weights across devices | Large models that don't fit in single GPU, with 2+ GPUs |

**Note:** Some acceleration methods can be combined together for optimized performance. See [Combining Acceleration Methods](diffusion/combining_methods.md) for detailed configuration examples.

### Memory Optimization

Memory optimization methods help reduce GPU memory usage, enabling inference on resource-constrained hardware or larger models.

| Method | Description | Best For |
|--------|-------------|----------|
| **[CPU Offload](diffusion/cpu_offload_diffusion.md)** | Offloads model components to CPU memory | Limited VRAM, large models on consumer GPUs |
| **[FP8 Quantization](diffusion/quantization/overview.md)** | Reduces DiT linear layers from BF16 to FP8 | Limited VRAM, accuracy perserved    |
| **[VAE Patch Parallelism](diffusion/parallelism_acceleration.md#vae-patch-parallelism)** | run vae decode tiling across ranks | Reduced memory |

### Extensions

Extension methods add specialized capabilities to diffusion models beyond standard inference.

| Method | Description | Best For |
|--------|-------------|----------|
| **[LoRA Inference](examples/offline_inference/lora_inference.md)** | Enables inference with Low-Rank Adaptation (LoRA) adapters weights | Reinforcement learning extensions |


### Quantization Methods

| Method | Configuration | Description | Best For |
|--------|--------------|-------------|----------|
| **FP8** | `quantization="fp8"` | FP8 W8A8 on Ada/Hopper, weight-only on older GPUs | Memory reduction, inference speedup |

## Supported Models

The following tables show which models support each acceleration method:
- **Sequence Parallel**: Includes both Ulysses-SP and Ring-Attention methods
- ✅ = Fully supported
- ❌ = Not supported

### ImageGen

| Model | Model Identifier | TeaCache | Cache-DiT | Sequence Parallel | CFG-Parallel | Tensor Parallel | CPU Offload | LoRA Inference | VAE-Patch-Parallel | FP8-Quantization |
|-------|------------------|:----------:|:-----------:|:-----------------:|:------------:|:---------------:|:-----------:|:-----------:|:------------------:|:----------------:|
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

| Model | Model Identifier | TeaCache | Cache-DiT | Sequence Parallel | CFG-Parallel | Tensor Parallel | CPU Offload | LoRA Inference | VAE-Patch-Parallel |
|-------|------------------|:--------:|:---------:|:-----------------:|:------------:|:---------------:|:-----------:|:-----------:|:------------------:|
| **Wan2.2-T2V** | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Wan2.2-I2V** | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Wan2.2-TI2V** | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

### AudioGen

| Model | Model Identifier | TeaCache | Cache-DiT | Sequence Parallel | CFG-Parallel | Tensor Parallel | CPU Offload | LoRA Inference | VAE-Patch-Parallel |
|-------|------------------|:--------:|:---------:|:-----------------:|:------------:|:---------------:|:-----------:|:-----------:|:------------------:|
| **Stable-Audio-Open** | `stabilityai/stable-audio-open-1.0` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

### Quantization

| Model | Model Identifier | FP8 |
|-------|------------------|:---:|
| **Qwen-Image** | `Qwen/Qwen-Image` | ✅ |
| **Qwen-Image-2512** | `Qwen/Qwen-Image-2512` | ✅ |
| **Z-Image** | `Tongyi-MAI/Z-Image-Turbo` | ✅ |


## Learn More

**Cache Acceleration:**
- **[TeaCache Configuration Guide](diffusion/teacache.md)** - Parameter tuning, performance tips, troubleshooting
- **[Cache-DiT Advanced Guide](diffusion/cache_dit.md)** - DBCache, TaylorSeer, SCM techniques and optimization

**Parallelism Methods:**
- **[Parallelism Acceleration Guide](diffusion/parallelism_acceleration.md)** - Ulysses-SP, Ring-Attention, CFG-Parallel, Tensor Parallelism details

**Memory Optimization:**
- **[CPU Offload Guide](diffusion/cpu_offload_diffusion.md)** - Offload model components to CPU, reduce GPU memory usage

**Extensions:**
- **[LoRA Inference Guide](examples/offline_inference/lora_inference.md)** - Low-Rank Adaptation for style customization and fine-tuning

**Advanced Topics:**
- **[Combining Acceleration Methods](diffusion/combining_methods.md)** - How to use cache + parallelism together for maximum performance
- **[Optimization & Troubleshooting Guide](diffusion/optimization_guide.md)** - Best practices, performance tuning, common issues and solutions
