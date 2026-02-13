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

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "A beautiful mountain landscape" \
  --cache-backend tea_cache \
  --ulysses-degree 2
```

**Using Cache-DiT + Ring-Attention:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "A futuristic city" \
  --cache-backend cache_dit \
  --ring-degree 2
```

### 2. Cache + CFG-Parallel

Best for: **Image editing with Classifier-Free Guidance**

Accelerates both the diffusion process and CFG computation.

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-Edit \
  --prompt "make it sunset" \
  --negative-prompt "low quality, blurry" \
  --image input.png \
  --cache-backend cache_dit \
  --cfg-parallel-size 2 \
  --true-cfg-scale 4.0
```

### 3. CFG-Parallel + Sequence Parallelism

Best for: **Large resolution image editing with CFG**

Combines both CFG branch splitting and sequence parallelism for maximum GPU utilization.

**CFG-Parallel + Ulysses-SP:**

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-Edit \
  --prompt "transform into autumn scene" \
  --negative-prompt "low quality" \
  --image input.png \
  --cache-backend cache_dit \
  --cfg-parallel-size 2 \
  --ulysses-degree 2 \
  --true-cfg-scale 4.0
```

### 4. Hybrid Ulysses + Ring + Vae tiling

Best for: **Very large images or videos on 4 GPUs**

Combines Ulysses-SP (all-to-all) with Ring-Attention (ring P2P) for scalable parallelism.

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "Epic fantasy landscape" \
  --cache-backend cache_dit \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --num-inference-steps 50 \
  --width 2048 \
  --height 2048 \
  --vae-use-tiling
```

### 5. Cache + Tensor Parallelism

Best for: **Large models that don't fit in single GPU memory**

Reduces per-GPU memory usage while maintaining cache acceleration.

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "A cat reading a book" \
  --cache-backend tea_cache \
  --tensor-parallel-size 2 \
  --num-inference-steps 9 \
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


### Configuration Tips

## Limitations


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

## See Also

- [Diffusion Acceleration Overview](../diffusion_features.md) - Main acceleration guide
