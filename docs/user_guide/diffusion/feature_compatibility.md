# Feature Compatibility

This guide explains the compatibility matrix of different diffusion features in vLLM-Omni. You can use cache methods together with parallelism methods and other features to achieve optimal speed and efficiency.

## Overview

vLLM-Omni supports combining:
- **Cache methods** (TeaCache, Cache-DiT) with **Parallelism methods** (Ulysses-SP, Ring-Attention, CFG-Parallel, Tensor Parallelism)
- **Multiple parallelism methods** together (e.g., Ulysses-SP + Ring-Attention, CFG-Parallel + Sequence Parallelism)
- **LoRA adapters** with most acceleration features
- **CPU offloading** with other memory optimization features

See the feature compatibility matrix in [Table](../diffusion_features.md#feature-compatibility)

## Common Combinations

### 1. Cache + Sequence Parallelism (Recommended)

Best for: **Large images (>1536px) or videos**

Combines cache acceleration with sequence parallelism for maximum speedup on single-device-challenging workloads.

**Using TeaCache + Ulysses-SP:**

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
    parallel_config=DiffusionParallelConfig(ulysses_degree=2)
)

outputs = omni.generate(
    "A beautiful mountain landscape",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
        width=2048,
        height=2048
    ),
)
```

**Using Cache-DiT + Ring-Attention:**

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 8,
        "residual_diff_threshold": 0.12,
    },
    parallel_config=DiffusionParallelConfig(ring_degree=4)
)

outputs = omni.generate(
    "A futuristic city",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
        width=2048,
        height=2048
    ),
)
```

### 2. Cache + CFG-Parallel

Best for: **Image editing with Classifier-Free Guidance**

Accelerates both the diffusion process and CFG computation.

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig
from PIL import Image

omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    cache_backend="cache_dit",
    parallel_config=DiffusionParallelConfig(cfg_parallel_size=2),
)

input_image = Image.open("input.png").convert("RGB")

outputs = omni.generate(
    {
        "prompt": "make it sunset",
        "negative_prompt": "low quality, blurry",
        "multi_modal_data": {"image": input_image}
    },
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
        true_cfg_scale=4.0,  # CFG-Parallel requires cfg_scale > 1
    ),
)
```

### 3. CFG-Parallel + Sequence Parallelism

Best for: **Large resolution image editing with CFG**

Combines both CFG branch splitting and sequence parallelism for maximum GPU utilization.

**CFG-Parallel + Ulysses-SP:**

```python
omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    cache_backend="cache_dit",
    parallel_config=DiffusionParallelConfig(
        cfg_parallel_size=2,
        ulysses_degree=2  # Total 4 GPUs: 2 for CFG × 2 for sequence
    ),
)

outputs = omni.generate(
    {
        "prompt": "transform into autumn scene",
        "negative_prompt": "low quality",
        "multi_modal_data": {"image": input_image}
    },
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
        true_cfg_scale=4.0,
        width=2048,
        height=2048
    ),
)
```

### 4. Hybrid Ulysses + Ring

Best for: **Very large images or videos on 8+ GPUs**

Combines Ulysses-SP (all-to-all) with Ring-Attention (ring P2P) for scalable parallelism.

```python
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    parallel_config=DiffusionParallelConfig(
        ulysses_degree=2,
        ring_degree=2  # Total 4 GPUs: 2 × 2
    )
)

outputs = omni.generate(
    "Epic fantasy landscape",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
        width=4096,
        height=4096
    ),
)
```

### 5. Cache + Tensor Parallelism

Best for: **Large models that don't fit in single GPU memory**

Reduces per-GPU memory usage while maintaining cache acceleration.

```python
omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
    parallel_config=DiffusionParallelConfig(tensor_parallel_size=2)
)

outputs = omni.generate(
    "A cat reading a book",
    OmniDiffusionSamplingParams(
        num_inference_steps=9,
        width=512,
        height=512
    ),
)
```

## Online Serving

### Cache + Sequence Parallelism

```bash
# TeaCache + Ulysses-SP
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh": 0.2}' \
  --usp 2

# Cache-DiT + Ring-Attention
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks": 1, "max_warmup_steps": 8}' \
  --ring 4
```

### Cache + CFG-Parallel

```bash
vllm serve Qwen/Qwen-Image-Edit --omni --port 8091 \
  --cache-backend cache_dit \
  --cfg-parallel-size 2
```

### Multiple Parallelism Methods

```bash
# CFG-Parallel + Ulysses-SP (4 GPUs total)
vllm serve Qwen/Qwen-Image-Edit --omni --port 8091 \
  --cache-backend cache_dit \
  --cfg-parallel-size 2 \
  --usp 2

# Hybrid Ulysses + Ring (4 GPUs total)
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend cache_dit \
  --usp 2 \
  --ring 2
```

## Performance Expectations

### Expected Speedups (Approximate)

These are rough estimates. Actual performance varies by model, hardware, and configuration.

| Configuration | GPUs | Expected Speedup vs Baseline |
|--------------|------|------------------------------|
| Cache-DiT alone | 1 | 1.5x-2.4x |
| Cache-DiT + Ulysses-SP (2) | 2 | 2.5x-4.0x |
| Cache-DiT + Ulysses-SP (4) | 4 | 3.5x-7.0x |
| Cache-DiT + CFG-Parallel | 2 | 2.0x-3.0x |
| Cache-DiT + Ulysses-SP + CFG-Parallel | 4 | 3.0x-6.0x |
| Cache-DiT + Hybrid Ulysses+Ring (4) | 4 | 3.5x-6.5x |

**Notes:**
- Speedups are cumulative but not perfectly multiplicative
- Communication overhead increases with more GPUs
- Cache methods provide consistent speedup across all GPU counts
- Sequence parallelism benefits increase with resolution

## Best Practices

### When to Combine Methods

1. **Single GPU (<1536px images):**
   - Use cache methods only (TeaCache or Cache-DiT)
   - No need for parallelism methods

2. **Multi-GPU (>1536px images):**
   - Add Ulysses-SP or Ring-Attention
   - Keep cache methods enabled

3. **Image Editing with CFG:**
   - Always enable CFG-Parallel when `true_cfg_scale > 1`
   - Combine with cache methods

4. **Very Large Models:**
   - Enable Tensor Parallelism to fit in memory
   - Add cache methods for speed

### Configuration Tips

1. **Start Simple:**
   - Enable cache first, validate quality
   - Add parallelism methods incrementally
   - Test each change

2. **GPU Count Planning:**
   - Total GPUs = `ulysses_degree × ring_degree × cfg_parallel_size × tensor_parallel_size`
   - Example: `ulysses_degree=2, cfg_parallel_size=2` requires 4 GPUs

3. **Memory vs Speed:**
   - Ring-Attention: More memory-efficient, slightly slower than Ulysses-SP
   - Ulysses-SP: Faster but higher memory usage
   - Use hybrid for best balance

4. **Cache Configuration:**
   - Cache settings independent of parallelism
   - Use same cache config as single-GPU

## Limitations

### Known Constraints

1. **Model Support:**
   - Not all models support all parallelism methods
   - Check [Supported Models](../diffusion_features.md#supported-models) table

2. **Memory Overhead:**
   - Text encoder not sharded in Tensor Parallelism (see [Issue #771](https://github.com/vllm-project/vllm-omni/issues/771))
   - Each method adds some memory overhead

3. **Communication Requirements:**
   - Parallelism methods require fast inter-GPU communication
   - Best with NVLink or InfiniBand
   - Performance degrades with slow interconnects

4. **Specific Model Limitations:**
   - Z-Image: Only supports `tensor_parallel_size` of 1 or 2
   - Some models don't support certain parallelism methods

## Troubleshooting

### Performance Not Scaling

**Symptoms:** Adding more GPUs doesn't improve speed proportionally

**Solutions:**
1. Check GPU communication bandwidth (use `nvidia-smi topo -m`)
2. Reduce parallelism degree if communication overhead is high
3. For very long sequences, prefer Ring-Attention over Ulysses-SP
4. Ensure batch size is large enough to saturate GPUs

### Out of Memory with Parallelism

**Symptoms:** OOM errors when combining methods

**Solutions:**
1. Enable Tensor Parallelism to shard weights
2. Reduce resolution or batch size
3. Use Ring-Attention instead of Ulysses-SP (more memory-efficient)
4. Disable some parallelism methods

### Configuration Errors

**Symptoms:** Errors about invalid parallel configuration

**Solutions:**
1. Verify total GPU count matches: `ulysses × ring × cfg × tensor`
2. Check model supports all enabled methods
3. Ensure divisibility constraints (e.g., Z-Image TP=1 or 2 only)

## Examples

### Complete Example: Maximum Acceleration

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig
from PIL import Image

# Setup: 4 GPUs, large resolution, CFG-enabled editing
omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 8,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.12,
        "enable_taylorseer": True,
        "taylorseer_order": 1,
    },
    parallel_config=DiffusionParallelConfig(
        cfg_parallel_size=2,  # 2 GPUs for CFG
        ulysses_degree=2,     # 2 GPUs for sequence
        # Total: 2 × 2 = 4 GPUs
    )
)

input_image = Image.open("input.png").convert("RGB")

outputs = omni.generate(
    {
        "prompt": "transform into a magical forest",
        "negative_prompt": "low quality, blurry, distorted",
        "multi_modal_data": {"image": input_image}
    },
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
        true_cfg_scale=4.0,
        width=2048,
        height=2048
    ),
)
```

## See Also

- [Diffusion Acceleration Overview](../diffusion_features.md) - Main acceleration guide
- [TeaCache Guide](teacache.md) - TeaCache configuration
- [Cache-DiT Guide](cache_dit.md) - Cache-DiT configuration
