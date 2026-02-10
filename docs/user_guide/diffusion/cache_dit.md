# Cache-DiT Acceleration Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)
- [Additional Resources](#additional-resources)

---

## Overview

Cache-DiT accelerates diffusion transformer models through intelligent caching mechanisms (DBCache, TaylorSeer, SCM), providing significant speedup with minimal quality loss. It's ideal for production deployments where inference speed matters and can be combined with other acceleration techniques for optimal performance.

---

## Quick Start

### Basic Usage

Simplest working example - enable cache-dit acceleration by setting `cache_backend="cache_dit"`:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",  # Enable Cache-DiT with defaults
)

outputs = omni.generate(
    "a beautiful landscape",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

**Note**: When `cache_config` is not provided, Cache-DiT uses optimized default values. See the [Configuration Parameters](#configuration-parameters) section for details.

### Custom Configuration

To customize cache-dit settings, provide a `cache_config` dictionary, for example:

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.12,
    },
)
```

---

## Example Script

### Offline Inference

Use the example script under `examples/offline_inference/text_to_image`:

```bash
cd examples/offline_inference/text_to_image
python text_to_image.py \
    --model Qwen/Qwen-Image \
    --prompt "a cup of coffee on the table" \
    --cache_backend cache_dit \
    --num_inference_steps 50
```

See the [text_to_image.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/text_to_image/text_to_image.py) for detailed configuration options.

For image-to-image tasks, use the example script under `examples/offline_inference/image_to_image`:

```bash
cd examples/offline_inference/image_to_image
python image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --prompt "make the sky more colorful" \
    --image-path path/to/input/image.jpg \
    --cache_backend cache_dit \
    --num-inference-steps 50 \
    --cache_dit_max_continuous_cached_steps 3 \
    --cache_dit_residual_diff_threshold 0.24 \
    --cache_dit_enable_taylorseer
```

See the [image_edit.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py) for detailed configuration options.


### Online Serving

```bash
# Default configuration (recommended)
vllm serve Qwen/Qwen-Image --omni --port 8091 --cache-backend cache_dit

# Custom configuration
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks": 1, "residual_diff_threshold": 0.12'
```

---

## Configuration Parameters

In `cache_config` dictionary (passed to `Omni` constructor)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Fn_compute_blocks` | int | 1 | First n blocks for difference computation (optimized for single-transformer models) |
| `Bn_compute_blocks` | int | 0 | Last n blocks for fusion |
| `max_warmup_steps` | int | 4 | Steps before caching starts (optimized for few-step distilled models) |
| `max_cached_steps` | int | -1 | Max cached steps (-1 = unlimited) |
| `max_continuous_cached_steps` | int | 3 | Max consecutive cached steps (prevents precision degradation) |
| `residual_diff_threshold` | float | 0.24 | Residual difference threshold (higher for more aggressive caching) |
| `num_inference_steps` | int \| None | None | Initial inference steps for SCM mask generation (optional, auto-refreshed during inference) |
| `enable_taylorseer` | bool | False | Enable TaylorSeer acceleration (not suitable for few-step distilled models) |
| `taylorseer_order` | int | 1 | Taylor expansion order |
| `scm_steps_mask_policy` | str \| None | None | SCM mask policy (None, "slow", "medium", "fast", "ultra") |
| `scm_steps_policy` | str | "dynamic" | SCM computation policy ("dynamic" or "static") |

---

## Best Practices

### When to Use

**Good for:**
- Production deployments requiring fast inference
- Diffusion transformer models (DiT architecture)
- Scenarios where 1.5x-3x speedup is valuable
- Batch inference workloads
- Models with 28+ inference steps

**Not for:**
- Non-DiT architectures (use model-specific acceleration instead)
- Scenarios requiring exact pixel-perfect reproduction
- Models already using few-step distillation (< 10 steps) - TaylorSeer may not help


### Expected Performance

| Configuration | Speedup | Quality | Use Case |
|--------------|---------|---------|----------|
| Default (DBCache only) | 1.5x-2.0x | Excellent | General use, production |
| DBCache + SCM Medium | 2.0x-2.5x | Very Good | Balanced speed/quality |
| Hybrid (All methods) | 2.5x-3.0x | Good | Speed-critical applications |

---

## Troubleshooting

### Common Issue 1: Quality Degradation

**Symptoms**: Generated images have visible artifacts or lower quality

**Solution**:
```python
# Reduce aggressiveness - use more conservative settings
cache_config={
    "residual_diff_threshold": 0.20,  # Lower threshold (closer to default 0.24)
    "Fn_compute_blocks": 8,            # Use more blocks for better decisions
    "max_warmup_steps": 6,             # Longer warmup
    "scm_steps_mask_policy": "slow",   # More compute steps
}
```


## Summary

Using Cache-DiT acceleration:

1. ✅ **Enable Cache-DiT** - Set `cache_backend="cache_dit"` to get 1.5x-3x speedup with optimized defaults
2. ✅ **(Optional) Customize** - Adjust `cache_config` parameters for specific speed/quality trade-offs

---

## Additional Resources

- [Cache-DiT User Guide](https://cache-dit.readthedocs.io/en/latest/user_guide/OVERVIEWS/)
- [Cache-DiT Benchmark](https://cache-dit.readthedocs.io/en/latest/benchmark/HYBRID_CACHE/)
- [DBCache Technical Details](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/)
