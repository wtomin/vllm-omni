# How to add cache-dit support for a new model

This section describes how to add cache-dit acceleration to a new diffusion **pipeline**. We use the Qwen-Image pipeline and LongCat-Image pipeline as reference implementations.

Cache-dit is a powerful library that accelerates diffusion transformers through intelligent caching mechanisms (DBCache, TaylorSeer, SCM). It typically provides around **2x speedup** with minimal quality loss.

## Overview

vLLM-omni integrates [`cache-dit`](https://github.com/vipshop/cache-dit) through the `CacheDiTBackend` class, which provides a unified interface for managing cache-dit acceleration on diffusion models. The backend automatically handles:

- DBCache configuration and management
- TaylorSeer calibration
- SCM (Step Computation Masking)
- Dynamic refresh when `num_inference_steps` changes

---

## Standard Models: Using Default Enabler


Most DiT models follow this pattern:
- Single transformer with one `ModuleList` of blocks
- Standard forward signature
- Compatible with cache-dit's automatic detection

**Examples:** Qwen-Image, Z-Image, StableDiffusion3

For standard single-transformer models, **no code changes are needed**. The `CacheDiTBackend` automatically uses `enable_cache_for_dit()`:

```python
from vllm_omni import Omni

# Works automatically for standard models
omni = Omni(
    model="Qwen/Qwen-Image",  # Standard single-transformer model
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
    }
)
```

**What happens automatically:**

```python
def enable_cache_for_dit(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Default enabler for standard single-transformer DiT models."""

    # Build cache configuration
    db_cache_config = DBCacheConfig(
        num_inference_steps=None,  # Will be set during first inference
        Fn_compute_blocks=cache_config.Fn_compute_blocks,
        Bn_compute_blocks=cache_config.Bn_compute_blocks,
        max_warmup_steps=cache_config.max_warmup_steps,
        max_cached_steps=cache_config.max_cached_steps,
        max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
        residual_diff_threshold=cache_config.residual_diff_threshold,
    )

    # Enable cache-dit on transformer
    cache_dit.enable_cache(
        pipeline.transformer,
        cache_config=db_cache_config,
    )

    # Return refresh function for dynamic num_inference_steps updates
    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True):
        cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose)

    return refresh_cache_context
```

---

## Custom Architectures: Writing a Custom Enabler

Some models require custom handling:
- **Dual-transformer:** Models with separate high-noise and low-noise transformers (e.g., Wan2.2)
- **Multi-block-list:** Models with multiple block lists in one transformer (e.g., LongCatImage with `transformer_blocks` + `single_transformer_blocks`)
- **Special forward patterns:** Models with non-standard block execution patterns

For complex architectures, you need to write a custom enabler function. This section shows how.

### Example 1: Dual-Transformer Model (Wan2.2)

Wan2.2 uses two transformers: one for high-noise steps and one for low-noise steps.

**Key challenges:**
- Need to apply cache-dit to both transformers
- Need to split `num_inference_steps` based on `boundary_ratio`
- Each transformer may have different cache configurations

**Solution:**

```python
def enable_cache_for_wan22(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for Wan2.2 dual-transformer architecture."""

    # Enable cache on both transformers using BlockAdapter
    cache_dit.enable_cache(
        BlockAdapter(
            transformer=[
                pipeline.transformer,      # High-noise transformer
                pipeline.transformer_2,    # Low-noise transformer
            ],
            blocks=[
                pipeline.transformer.blocks,
                pipeline.transformer_2.blocks,
            ],
            forward_pattern=[
                ForwardPattern.Pattern_2,  # Both use Pattern_2
                ForwardPattern.Pattern_2,
            ],
            params_modifiers=[
                # High-noise transformer: more warmup steps
                ParamsModifier(
                    cache_config=DBCacheConfig().reset(
                        max_warmup_steps=cache_config.max_warmup_steps,
                        max_cached_steps=cache_config.max_cached_steps,
                    ),
                ),
                # Low-noise transformer: fewer warmup steps (only ~30% of total steps)
                ParamsModifier(
                    cache_config=DBCacheConfig().reset(
                        max_warmup_steps=2,
                        max_cached_steps=20,
                    ),
                ),
            ],
            has_separate_cfg=True,  # Each transformer handles CFG separately
        ),
        cache_config=DBCacheConfig(
            Fn_compute_blocks=cache_config.Fn_compute_blocks,
            Bn_compute_blocks=cache_config.Bn_compute_blocks,
            max_warmup_steps=cache_config.max_warmup_steps,
            max_cached_steps=cache_config.max_cached_steps,
            max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
            residual_diff_threshold=cache_config.residual_diff_threshold,
            num_inference_steps=None,
        ),
    )

    # Helper to split inference steps between transformers
    def _split_inference_steps(num_inference_steps: int) -> tuple[int, int]:
        """Split steps into high-noise and low-noise based on boundary_ratio."""
        if pipeline.boundary_ratio is not None:
            boundary_timestep = pipeline.boundary_ratio * pipeline.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        device = next(pipeline.transformer.parameters()).device
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = pipeline.scheduler.timesteps
        num_high_noise_steps = sum(1 for t in timesteps if boundary_timestep is None or t >= boundary_timestep)
        num_low_noise_steps = num_inference_steps - num_high_noise_steps

        return num_high_noise_steps, num_low_noise_steps

    # Return refresh function that updates both transformers
    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True):
        num_high_noise_steps, num_low_noise_steps = _split_inference_steps(num_inference_steps)

        # Refresh high-noise transformer
        cache_dit.refresh_context(
            pipeline.transformer,
            num_inference_steps=num_high_noise_steps,
            verbose=verbose,
        )

        # Refresh low-noise transformer
        cache_dit.refresh_context(
            pipeline.transformer_2,
            num_inference_steps=num_low_noise_steps,
            verbose=verbose,
        )

    return refresh_cache_context
```

### Example 2: Multi-Block-List Model (LongCatImage)

LongCatImage has a single transformer with two block lists: `transformer_blocks` and `single_transformer_blocks`.

**Key challenges:**
- Multiple block lists in one transformer
- Need to specify different forward patterns for different block lists

**Solution:**

```python
def enable_cache_for_longcat_image(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for LongCatImage pipeline."""

    # Build cache configuration
    db_cache_config = DBCacheConfig(
        num_inference_steps=None,
        Fn_compute_blocks=cache_config.Fn_compute_blocks,
        Bn_compute_blocks=cache_config.Bn_compute_blocks,
        max_warmup_steps=cache_config.max_warmup_steps,
        max_cached_steps=cache_config.max_cached_steps,
        max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
        residual_diff_threshold=cache_config.residual_diff_threshold,
    )

    # Build modifier with TaylorSeer support
    calibrator = None
    if cache_config.enable_taylorseer:
        calibrator = TaylorSeerCalibratorConfig(taylorseer_order=cache_config.taylorseer_order)

    modifier = ParamsModifier(
        cache_config=db_cache_config,
        calibrator_config=calibrator,
    )

    # Enable cache using BlockAdapter with multiple block lists
    cache_dit.enable_cache(
        BlockAdapter(
            transformer=pipeline.transformer,
            blocks=[
                pipeline.transformer.transformer_blocks,        # Joint blocks
                pipeline.transformer.single_transformer_blocks, # Single blocks
            ],
            forward_pattern=[
                ForwardPattern.Pattern_1,  # Both use Pattern_1
                ForwardPattern.Pattern_1,
            ],
            params_modifiers=[modifier],  # Apply same modifier to both block lists
        ),
        cache_config=db_cache_config,
    )

    # Return refresh function
    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True):
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose)
        else:
            # With SCM support
            cache_dit.refresh_context(
                pipeline.transformer,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_inference_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy,
                        total_steps=num_inference_steps,
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

    return refresh_cache_context
```

## Registering Custom Enablers

After writing your custom enabler, register it in `CUSTOM_DIT_ENABLERS` in `vllm_omni/diffusion/cache/cache_dit_backend.py`:

```python
CUSTOM_DIT_ENABLERS = {
    "Wan22Pipeline": enable_cache_for_wan22,
    "Wan22I2VPipeline": enable_cache_for_wan22,
    "FluxPipeline": enable_cache_for_flux,
    "LongCatImagePipeline": enable_cache_for_longcat_image,
    "LongCatImageEditPipeline": enable_cache_for_longcat_image,
    "StableDiffusion3Pipeline": enable_cache_for_sd3,
    "BagelPipeline": enable_cache_for_bagel,
    "YourCustomPipeline": enable_cache_for_your_model,  # Add here
}
```

---

## Understanding Cache-DiT API

This section lists important API names from Cache-DiT, which are relevant to vLLM-Omni use cases. A detailed API document is in [cache_api_doc](https://github.com/vipshop/cache-dit/blob/main/docs/user_guide/CACHE_API.md).

### Key APIs

**`BlockAdapter`** - Core abstraction for applying cache-dit to transformers
- Specifies transformer module(s), block list(s), and forward signature pattern(s)
- Configures per-transformer cache settings via `params_modifiers`
- Handles single/dual transformer architectures

**`ForwardPattern`** - Defines block forward signature patterns
- `Pattern_0`: `(hidden_states, encoder_hidden_states)` - Language model blocks
- `Pattern_1`: `(hidden_states, temb, **kwargs)` - Single-stream transformers
- `Pattern_2`: `(hidden_states, encoder_hidden_states, temb, **kwargs)` - Dual-stream transformers

**`ParamsModifier`** - Per-transformer or per-block-list cache configuration customization
- Overrides `cache_config` (warmup steps, cached steps, etc.)
- Overrides `calibrator_config` (calibration strategy, order, etc.)

**`refresh_context()`** - Updates cache context when `num_inference_steps` changes
- Required function returned by every enabler
- Supports SCM (Selective Computation Masking) for dynamic step skipping

---

## Testing

After adding support, test with:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Test your custom model
omni = Omni(
    model="your-model-name",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,Cache-dit enabled successfully on
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.24,
    }
)

images = omni.generate(
    "a beautiful landscape",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

**Verify:**
1. Cache is applied (check logs for "Cache-dit enabled successfully on xxx")
2. Performance improvement (should be around 2x faster)
3. Image quality (compare with `cache_backend=None`)

---


## Troubleshooting


### Cache not applied

**Cause:** Enabler not registered or pipeline name mismatch.

**Solution:**
1. Verify `pipeline.__class__.__name__` matches registry key
2. Check enabler is in `CUSTOM_DIT_ENABLERS`

### Quality degradation

**Cause:** Cache parameters too aggressive.

**Solution:**
1. Lower `residual_diff_threshold` (try 0.12-0.18)
2. Increase `max_warmup_steps` (try 6-8)
3. Reduce `max_continuous_cached_steps` (try 2)

Check the [user guide for cache_dit](../../user_guide/diffusion/cache_dit_acceleration.md) for more adjustable parameters.


---

## Reference Implementations

See these files for complete examples:

- **Single transformer (default):** `cache_dit_backend.py::enable_cache_for_dit`
- **Dual transformer (Wan2.2):** `cache_dit_backend.py::enable_cache_for_wan22`
- **Multi-block-list (LongCat):** `cache_dit_backend.py::enable_cache_for_longcat_image`
- **Omni model (BAGEL):** `cache_dit_backend.py::enable_cache_for_bagel`

---

## Summary

Adding cache-dit support:

1. ✅ **Standard models:** No code needed - works automatically
2. ✅ **Custom architectures:** Write enabler function
3. ✅ Register in `CUSTOM_DIT_ENABLERS`
4. ✅ Return `refresh_cache_context` function
5. ✅ Test with `cache_backend="cache_dit"`

For most models, the default enabler is sufficient. Only write custom enablers for complex architectures!
