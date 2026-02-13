# FP8 Quantization Guide


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

FP8 quantization reduces memory usage and improves inference speed by converting BF16/FP16 weights to FP8 format at model load time. No calibration dataset or pre-quantized checkpoint is required, making it easy to deploy.

Depending on the model architecture, either all layers can be quantized, or some sensitive layers should remain in BF16 for optimal quality. See the [Supported Models](#supported-models) table for model-specific recommendations.

**Understanding Sensitive Layers:**

Common sensitive layers in DiT-based diffusion models include **image-stream MLPs** (`img_mlp`). These layers are particularly vulnerable to FP8 precision loss because they process denoising latents whose dynamic range shifts significantly across timesteps. Unlike attention projections (which benefit from QK-Norm stabilization), MLPs have no built-in normalization to absorb quantization error. In deep architectures (e.g., 60+ residual blocks), small per-layer errors compound and degrade output quality.

Other layers such as **attention projections** (`to_qkv`, `to_out`) and **text-stream MLPs** (`txt_mlp`) are generally more robust due to normalization or more stable input statistics.

See supported models list in [Supported Models](#supported-models).

---

## Quick Start

### Basic Usage

Simplest working example - quantize all layers to FP8:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    quantization="fp8",
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

### Custom Configuration

For models with sensitive layers, skip specific layers to maintain quality:

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    quantization_config={
        "method": "fp8",
        "ignored_layers": ["img_mlp"],  # Keep img_mlp in BF16
    },
)
```

---

## Example Script

### Offline Inference

Use Python script under `examples/offline_inference/text_to_image/`:

**Quantize all layers:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "A cat sitting on a windowsill" \
  --quantization fp8 \
  --num-inference-steps 50
```

**Skip sensitive layers:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "A beautiful landscape" \
  --quantization fp8 \
  --ignored-layers '["img_mlp"]' \
  --num-inference-steps 50
```

See the [text_to_image.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/text_to_image/text_to_image.py) for detailed configuration options.

### Online Serving

```bash
# Quantize all layers
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091 --quantization fp8

```

---

## Configuration Parameters

In `quantization_config` passed to `Omni` constructor:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | Quantization method (`"fp8"`) |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16 |
| `activation_scheme` | str | `"dynamic"` | `"dynamic"` (no calibration) or `"static"` |
| `weight_block_size` | list[int] \| None | `None` | Block size for block-wise weight quantization |

**Notes:**
- The available `ignored_layers` names depend on the model architecture (e.g., `to_qkv`, `to_out`, `img_mlp`, `txt_mlp`)
- Consult the transformer source code for your target model to identify layer names

---

## Supported Models

| Model | HF Models | Recommendation | `ignored_layers` |
|-------|-----------|---------------|------------------|
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | All layers | None |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Skip sensitive layers | `img_mlp` |

---

## Best Practices

### When to Use

**Good for:**

- Production deployments where memory is limited (reduces memory footprint by ~50%)
- Scenarios requiring faster inference speed
- Models with robust architectures that tolerate quantization well (e.g., Z-Image)

**Not for:**

- Maximum quality requirements where no degradation is acceptable
- Extremely small models where memory is not a concern

### Model-Specific Recommendations

1. **Z-Image Models:**
   - Quantize all layers - the architecture is robust to FP8 quantization
   - Expected: minimal quality loss with significant speedup

2. **Qwen-Image Models:**
   - Always skip `img_mlp` layers to maintain quality
   - Image-stream MLPs are particularly sensitive to quantization errors
   - Expected: good quality with moderate speedup

### Combining with Other Features

FP8 quantization can be combined with cache acceleration for maximum performance:

```bash
# FP8 + TeaCache
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "A beautiful sunset" \
  --quantization fp8 \
  --cache-backend tea_cache

# FP8 + Cache-DiT (skip sensitive layers)
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "A mountain landscape" \
  --quantization fp8 \
  --cache-backend cache_dit
```

---

## Troubleshooting

### Common Issue 1: Quality Degradation

**Symptoms**: Generated images show artifacts, color shifts, or loss of detail compared to BF16

**Solution**:

1. Identify and skip sensitive layers:
   ```bash
   python examples/offline_inference/text_to_image/text_to_image.py \
     --model Qwen/Qwen-Image \
     --quantization fp8 \
     --ignored-layers "img_mlp"
   ```

2. If quality issues persist, try skipping additional layers:
   ```python
   quantization_config={
       "method": "fp8",
       "ignored_layers": ["img_mlp", "txt_mlp"],
   }
   ```

### Common Issue 2: Out of Memory Despite Quantization

**Symptoms**: CUDA OOM errors even with FP8 quantization enabled

**Solutions**:

1. Reduce batch size or resolution
2. Combine with other memory optimization techniques:
   ```bash
   python examples/offline_inference/text_to_image/text_to_image.py \
     --model Qwen/Qwen-Image \
     --quantization fp8 \
     --tensor-parallel-size 2  # Use tensor parallelism
   ```

---

## Summary

1. ✅ **Enable FP8** - Set `quantization="fp8"` to reduce memory usage by ~50% with minimal quality loss
2. ✅ **Skip Sensitive Layers** - Use `ignored_layers` for models like Qwen-Image to maintain quality
3. ✅ **Combine with Other Features** - Stack FP8 with cache acceleration, or tensor parallel
