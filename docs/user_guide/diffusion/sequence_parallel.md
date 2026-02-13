# Sequence Parallelism Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

Sequence parallelism splits the input along the sequence dimension across multiple GPUs, allowing each device to process only a portion of the sequence. vLLM-Omni provides 1.5x-3.6x speedup for large images (2048x2048+) and videos using DeepSpeed Ulysses, Ring-Attention, or hybrid approaches. Use sequence parallelism when generating high-resolution images/videos that don't fit on a single GPU or require faster inference.

See supported models list in [Diffusion Features - Supported Models](../diffusion_features.md#supported-models).

**Supported Methods:**

- **DeepSpeed Ulysses Sequence Parallel (Ulysses-SP)** ([paper](https://arxiv.org/pdf/2309.14509)): Uses all-to-all communication for subset of attention heads per device
- **Ring-Attention** ([paper](https://arxiv.org/abs/2310.01889)): Uses ring-based P2P communication with sharded sequence dimension throughout
- **Hybrid Ulysses + Ring**: Combines both for larger scale parallelism (`ulysses_degree × ring_degree`)

---

## Quick Start

### Basic Usage - Ulysses-SP

Simplest working example with Ulysses Sequence Parallel:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2)  # Enable Ulysses-SP
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50, width=2048, height=2048),
)
```

### Alternative Methods

**Ring-Attention** (better for very long sequences):

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ring_degree=2)  # Enable Ring-Attention
)
```

**Hybrid Ulysses + Ring** (for larger scale):

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2, ring_degree=2)  # 4 GPUs total
)
```

---

## Example Script

### Offline Inference

Use Python script under `examples/offline_inference/text_to_image/text_to_image.py`:

**Ulysses-SP:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "A cat sitting on a windowsill" \
    --ulysses-degree 2 \
    --width 2048 --height 2048
```

**Ring-Attention:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "A cat sitting on a windowsill" \
    --ring-degree 2 \
    --width 2048 --height 2048
```

**Hybrid Ulysses + Ring:**

```bash
# Hybrid: 2 Ulysses × 2 Ring = 4 GPUs total
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "A cat sitting on a windowsill" \
    --ulysses-degree 2 --ring-degree 2 \
    --width 2048 --height 2048
```

### Online Serving

**Ulysses-SP:**

```bash
# Text-to-image (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2
```

**Ring-Attention:**

```bash
# Text-to-image (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --ring 2
```

**Hybrid Ulysses + Ring:**

```bash
# Text-to-image (requires >= 4 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2 --ring 2
```

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ulysses_degree` | int | 1 | Number of GPUs for Ulysses-SP. Uses all-to-all communication. Best for moderate sequences (4K-32K tokens). |
| `ring_degree` | int | 1 | Number of GPUs for Ring-Attention. Uses P2P ring communication. Better for very long sequences (>32K tokens). |

**Notes:**
- Total GPUs used = `ulysses_degree × ring_degree`
- Degrees must evenly divide the sequence length for optimal performance

---

## Benchmarks

!!! note "Benchmark Disclaimer"
    These benchmarks are provided for **general reference only**. The configurations shown use default or common parameter settings and have not been exhaustively optimized for maximum performance. Actual performance may vary based on specific model, hardware, parameter tuning, and inference settings.

**Ulysses-SP** - Qwen/Qwen-Image, 2048x2048, 50 steps, NVIDIA H800, `sdpa` backend:

| Configuration | Ulysses degree |Generation Time | Speedup |
|---------------|----------------|----------------|---------|
| **Baseline (diffusers)** | - | 112.5s | 1.0x |
| Ulysses-SP  |  2  |  65.2s | 1.73x |
| Ulysses-SP  |  4  | 39.6s | 2.84x |
| Ulysses-SP  |  8  | 30.8s | 3.65x |

**Ring-Attention** - Qwen/Qwen-Image, 1024x1024, 50 steps, NVIDIA A100, `flash_attn` backend:

| Configuration | Ring degree |Generation Time | Speedup |
|---------------|-------------|----------------|---------|
| **Baseline (diffusers)** | - | 45.2s | 1.0x |
| Ring-Attention  |  2  |  29.9s | 1.51x |
| Ring-Attention  |  4  | 23.3s | 1.94x |

**Hybrid Ulysses + Ring** - Qwen/Qwen-Image, 1024x1024, 50 steps, NVIDIA A100, `flash_attn` backend:

| Configuration | Ulysses | Ring | Generation Time | Speedup |
|---------------|---------|------|-----------------|---------|
| **Baseline (diffusers)** | - | - | 45.2s | 1.0x |
| Hybrid  |  2  |  2  |  24.3s | 1.87x |

---

## Best Practices

### When to Use

**Good for Ulysses-SP:**
- Large images (2048x2048 or higher) with moderate sequences (4K-32K tokens)
- Fast inter-GPU communication (NVLink/InfiniBand)
- Maximum speed is priority
- 2048x2048: degree=2, 4096x4096: degree=4-8

**Good for Ring-Attention:**
- Very long sequences (>32K tokens) or limited GPU memory
- Slower inter-GPU communication (PCIe)
- Video generation
- 2048x2048: degree=2-4, Videos: degree=4-8+

**Not for:**
- Small images (<1024px) - overhead exceeds benefit, use single GPU with cache instead
- Very long sequences with Ulysses-SP - Ring-Attention is better
- Small to moderate sequences with Ring - Ulysses-SP is faster


### Expected Performance

| Configuration | GPUs | Expected Speedup | Best Resolution |
|--------------|------|------------------|----------------|
| Ulysses-SP=2 | 2 | 1.5x-1.8x | 2048x2048+ |
| Ulysses-SP=4 | 4 | 2.5x-3.0x | 2048x2048+ |
| Ulysses-SP=8 | 8 | 3.5x-4.0x | 4096x4096+ |
| Ring=2 | 2 | 1.4x-1.6x | 2048x2048+ |
| Ring=4 | 4 | 2.2x-2.5x | 2048x2048+ |
| Hybrid (2×2) | 4 | 2.0x-2.5x | 4096x4096+ |

**Notes:**
- Speedups assume fast inter-GPU communication (NVLink)
- Check GPU topology: `nvidia-smi topo -m` - Look for NV links
- Start with degree=2, scale to 4 or 8 based on testing
- Degrees should divide sequence length: `seq_len = (height × width) / (patch_size²)`

---

## Troubleshooting

### Common Issue 1: Performance Not Scaling

**Symptoms**: Adding GPUs doesn't improve speed proportionally, or higher parallelism degree is slower

**Diagnosis:**
```bash
# Check GPU topology
nvidia-smi topo -m

# Monitor GPU utilization during inference
nvidia-smi dmon -s u -d 1
```

**Solutions:**

1. Check inter-GPU communication - ensure NVLink (NV1, NV2) is available, not PHB/SYS (slow PCIe)
2. Reduce parallelism degree if over-parallelized:
```python
# If 8 GPUs is slower than 4
parallel_config=DiffusionParallelConfig(ulysses_degree=4)
```
3. Use Ring-Attention instead of Ulysses-SP for better communication efficiency:
```python
parallel_config=DiffusionParallelConfig(ring_degree=4)
```
4. Don't over-parallelize - For 1024×1024: degree=2 max, 2048×2048: degree=4 max

### Common Issue 2: Out of Memory (OOM)

**Symptoms**: CUDA OOM errors or process crashes with memory errors

**Solutions:**

1. Use Ring-Attention (more memory-efficient):
```python
parallel_config=DiffusionParallelConfig(ring_degree=2)
```
2. Increase parallelism degree to split sequence more:
```python
parallel_config=DiffusionParallelConfig(ulysses_degree=4)  # From 2
```



## Summary

1. ✅ **Enable Sequence Parallelism** - Set `ulysses_degree` or `ring_degree` for long sequence generation
2. ✅ **Troubleshooting** - Check GPU topology with `nvidia-smi topo -m`, use Ring for memory issues, reduce degree if performance doesn't scale
