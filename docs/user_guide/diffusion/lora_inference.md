# LoRA Inference Guide


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

LoRA (Low-Rank Adaptation) enables efficient fine-tuning and inference with diffusion models by using lightweight adapter weights. vLLM-Omni provides flexible LoRA support with minimal memory overhead and dynamic loading capabilities. It's ideal for serving multiple specialized models from a single base model, enabling personalized image generation without maintaining separate model copies.

Similar to vLLM, vLLM-omni uses a unified LoRA handling mechanism:

- **Pre-loaded LoRA**: Loaded at initialization via `--lora-path` (pre-loaded into cache)
- **Per-request LoRA**: Loaded on-demand. In the example, the LoRA is loaded via `--lora-request-path` in each request

Both approaches use the same underlying mechanism - all LoRA adapters are handled uniformly through `set_active_adapter()`. If no LoRA request is provided in a request, all adapters are deactivated.

See supported models list in [Supported Models](../diffusion_features.md#supported-models)

---

## Quick Start


### Basic Usage

Simplest working example with a LoRA adapter:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Initialize with base model
omni = Omni(model="stabilityai/stable-diffusion-3.5-medium")

# Generate with LoRA adapter
outputs = omni.generate(
    "A piece of cheesecake",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
        height=1024,
        width=1024,
        lora_request={"path": "/path/to/lora/adapter", "scale": 1.0}
    ),
)
```


## Example Script

### Offline Inference

Use python script under `examples/offline_inference/lora_inference`:

**Pre-loaded LoRA:**
```bash
python -m examples.offline_inference.lora_inference.lora_inference \
    --prompt "A piece of cheesecake" \
    --lora-path /path/to/lora/ \
    --lora-scale 1.0 \
    --num-inference-steps 50 \
    --height 1024 \
    --width 1024 \
    --output output_preloaded.png
```

**Note**: When using `--lora-path`, the adapter is loaded at init time with a stable ID derived from the adapter path. This example activates it automatically for the request.


**Per-request LoRA:**

Load a LoRA adapter on-demand for each request:

```bash
python -m examples.offline_inference.lora_inference.lora_inference \
    --prompt "A piece of cheesecake" \
    --lora-request-path /path/to/lora/ \
    --lora-scale 1.0 \
    --num-inference-steps 50 \
    --height 1024 \
    --width 1024 \
    --output output_per_request.png
```


### Online Serving


**Call API with curl:**
```bash
export LORA_PATH=/path/to/lora_adapter
export SERVER=http://localhost:8091
export PROMPT="A piece of cheesecake"
export LORA_NAME=my_lora
export LORA_SCALE=1.0

curl -X POST "${SERVER}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"stabilityai/stable-diffusion-3.5-medium\",
    \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
    \"lora_request\": {
      \"path\": \"${LORA_PATH}\",
      \"name\": \"${LORA_NAME}\",
      \"scale\": ${LORA_SCALE}
    },
    \"sampling_params\": {
      \"height\": 1024,
      \"width\": 1024,
      \"num_inference_steps\": 50
    }
  }"
```

**Call API with Python:**
```bash
python openai_chat_client.py \
  --prompt "A piece of cheesecake" \
  --lora-path /path/to/lora_adapter \
  --lora-name my_lora \
  --lora-scale 1.0 \
  --output output.png
```

---

## Configuration Parameters

### LoRA Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_path` | str | None | Path to LoRA adapter folder to pre-load at initialization. Loaded into cache with a stable ID derived from the path. |
| `lora_request_path` | str | Required | Path to LoRA adapter folder for the current request. Must be readable on the server machine. |
| `lora_request_id` | int | Auto-generated | Integer ID for the LoRA adapter. If not provided, a stable ID is derived from the adapter path. |
| `lora_request_scale` | float | 1.0 | Scale factor for LoRA weights. Higher values (e.g., 1.5) increase adapter influence; lower values (e.g., 0.5) reduce it. |

### LoRA Adapter Format

LoRA adapters must be in PEFT (Parameter-Efficient Fine-Tuning) format:

```
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

---


## Best Practices

### When to Use

**Good for:**

- Personalized image/video generation (styles, characters, objects)
- Reinforcement Learning integration

---

## Troubleshooting

### Common Issue 1: LoRA Adapter Not Found

**Symptoms**: Error message like `FileNotFoundError: [Errno 2] No such file or directory: '/path/to/lora/'`

**Solutions**:
1. Verify the adapter path is correct and accessible from the server machine
2. For online serving, ensure the path is on the **server machine**, not the client
3. Check that the adapter directory contains `adapter_config.json` and `adapter_model.safetensors`

```python
# Correct path format
lora_request={"path": "/absolute/path/to/lora_adapter"}

# Check adapter structure
# lora_adapter/
# ├── adapter_config.json
# └── adapter_model.safetensors
```

### Common Issue 2: Unexpected Visual Results

**Symptoms**: Generated images don't match expected style or quality

**Solutions**:
1. Adjust the `lora_scale` parameter:
   - Too high (>1.5): Over-applied style, artifacts
   - Too low (<0.3): Minimal effect
2. Verify adapter compatibility with the base model
3. Ensure adapter was trained on compatible model architecture

---


## Summary

1. ✅ **Enable LoRA** - Use `lora_request={"path": "/path/to/lora/", "scale": 1.0}` for per-request adapters, or `lora_path="/path/to/lora/"` for pre-loaded adapters
2. ✅ **Adjust Scale** - Set `scale` between 0.5-1.5 for optimal quality/style balance (default: 1.0)
3. ✅ **Choose Loading Strategy** - Pre-load frequent adapters for speed, use per-request loading for flexibility
