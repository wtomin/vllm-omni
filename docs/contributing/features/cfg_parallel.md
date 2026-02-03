# How to add CFG-Parallel support for a new pipeline

This section describes how to add CFG-Parallel (Classifier-Free Guidance Parallel) to a diffusion **pipeline**. We use the Qwen-Image pipeline (`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`) as the reference implementation.

CFG-Parallel accelerates diffusion inference by distributing the conditional (positive) and unconditional (negative) forward passes to different GPU ranks, enabling **2x speedup** for CFG-enabled generation compared to sequential execution.

## Overview

### What is CFG-Parallel?

In standard Classifier-Free Guidance, each diffusion step requires two forward passes through the transformer:

1. **Positive/Conditional**: Guided by the text prompt
2. **Negative/Unconditional**: Typically using empty or negative prompt

These predictions are combined with:
```
noise_pred = negative_pred + cfg_scale * (positive_pred - negative_pred)
```

**Problem:** Sequential execution wastes compute resources - the GPU must run both passes one after another.

**Solution:** CFG-Parallel distributes these two passes across different GPU ranks:
- **Rank 0**: Computes positive prediction
- **Rank 1**: Computes negative prediction
- Results are gathered and combined on Rank 0

---

## Architecture

vLLM-omni provides `CFGParallelMixin` that encapsulates all CFG parallel logic. Pipelines inherit from this mixin and implement a `diffuse()` method that orchestrates the denoising loop.

### CFGParallelMixin Key Methods

| Method | Purpose | Automatic Behavior |
|--------|---------|-------------------|
| `predict_noise_maybe_with_cfg()` | Predict noise with CFG | Detects parallel mode, distributes computation, gathers results |
| `scheduler_step_maybe_with_cfg()` | Step scheduler with sync | Rank 0 steps, broadcasts latents to all ranks |
| `combine_cfg_noise()` | Combine positive/negative | Applies CFG formula with optional normalization |
| `predict_noise()` | Forward pass wrapper | Override for custom transformer calls |
| `cfg_normalize_function()` | Normalize CFG output | Override for custom normalization |

---

## Step-by-Step Implementation

### Step 1: Create Pipeline-Specific Mixin

Create a mixin class that inherits from `CFGParallelMixin` and implements the `diffuse()` method for your specific model.

**Example (Qwen-Image):**

```python
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
import torch

class QwenImageCFGParallelMixin(CFGParallelMixin):
    """
    CFG-Parallel mixin for Qwen-Image pipelines.
    Shared by QwenImagePipeline, QwenImageEditPipeline, etc.
    """

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        latents: torch.Tensor,
        img_shapes: torch.Tensor,
        txt_seq_lens: torch.Tensor,
        negative_txt_seq_lens: torch.Tensor,
        timesteps: torch.Tensor,
        do_true_cfg: bool,
        guidance: torch.Tensor,
        true_cfg_scale: float,
        cfg_normalize: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Prepare timestep tensor
            timestep = t.expand(latents.shape[0]).to(
                device=latents.device,
                dtype=latents.dtype
            )

            # Prepare kwargs for positive (conditional) prediction
            positive_kwargs = {
                "hidden_states": latents,
                "timestep": timestep / 1000,
                "guidance": guidance,
                "encoder_hidden_states": prompt_embeds,
                "encoder_hidden_states_mask": prompt_embeds_mask,
                "img_shapes": img_shapes,
                "txt_seq_lens": txt_seq_lens,
            }

            # Prepare kwargs for negative (unconditional) prediction
            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latents,
                    "timestep": timestep / 1000,
                    "guidance": guidance,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                    "img_shapes": img_shapes,
                    "txt_seq_lens": negative_txt_seq_lens,
                }
            else:
                negative_kwargs = None

            # Predict noise with automatic CFG parallel handling
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=cfg_normalize,
            )

            # Step scheduler with automatic CFG synchronization
            latents = self.scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg
            )

        return latents
```

**Key Points:**
- Separate `positive_kwargs` and `negative_kwargs` for the two forward passes
- Pass both to `predict_noise_maybe_with_cfg()` which handles distribution
- Use `scheduler_step_maybe_with_cfg()` for synchronized latent updates

### Step 2: Inherit Mixin in Pipeline Class

Make your pipeline class inherit from the CFG mixin alongside the base pipeline class.

```python
from diffusers import DiffusionPipeline

class QwenImagePipeline(QwenImageCFGParallelMixin, DiffusionPipeline):
    """Qwen-Image pipeline with CFG-Parallel support."""

    def __call__(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        **kwargs,
    ):
        # Encode prompts
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt)

        if negative_prompt or guidance_scale > 1.0:
            negative_embeds, negative_mask = self.encode_prompt(
                negative_prompt or ""
            )
            do_true_cfg = True
        else:
            negative_embeds = None
            negative_mask = None
            do_true_cfg = False

        # Initialize latents
        latents = self.prepare_latents(...)

        # Get timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Run diffusion loop (calls the mixin's diffuse method)
        latents = self.diffuse(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_embeds,
            negative_prompt_embeds_mask=negative_mask,
            latents=latents,
            timesteps=timesteps,
            do_true_cfg=do_true_cfg,
            true_cfg_scale=guidance_scale,
            ...
        )

        # Decode latents
        images = self.vae.decode(latents)
        return images
```

### Step 3: Handle Image Editing Pipelines (Optional)

For image editing pipelines that concatenate condition latents with noise latents, use the `output_slice` parameter.

```python
class QwenImageEditPipeline(QwenImageCFGParallelMixin, DiffusionPipeline):
    def diffuse(
        self,
        latents: torch.Tensor,
        image_latents: torch.Tensor,  # Condition image latents
        **kwargs,
    ) -> torch.Tensor:
        for i, t in enumerate(timesteps):
            # Concatenate noise latents with condition latents
            latent_model_input = torch.cat([latents, image_latents], dim=1)

            positive_kwargs = {
                "hidden_states": latent_model_input,  # Concatenated input
                ...
            }
            negative_kwargs = {
                "hidden_states": latent_model_input,  # Same for negative
                ...
            }

            # Specify output_slice to remove condition latents from output
            noise_pred = self.predict_noise_maybe_with_cfg(
                ...,
                output_slice=latents.size(1),  # Only keep noise prediction
            )

            # Step only the noise latents (not condition latents)
            latents = self.scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg
            )

        return latents
```

---

## How It Works

### Automatic Mode Detection

`predict_noise_maybe_with_cfg()` automatically detects the execution mode:

```python
# Check if CFG parallel is enabled
cfg_parallel_ready = (
    do_true_cfg and
    get_classifier_free_guidance_world_size() > 1
)

if cfg_parallel_ready:
    # CFG-Parallel mode: distribute computation
    cfg_rank = get_classifier_free_guidance_rank()

    if cfg_rank == 0:
        local_pred = self.predict_noise(**positive_kwargs)  # Rank 0: positive
    else:
        local_pred = self.predict_noise(**negative_kwargs)  # Rank 1: negative

    # Gather predictions from both ranks
    gathered = cfg_group.all_gather(local_pred, separate_tensors=True)

    # Combine on rank 0
    if cfg_rank == 0:
        noise_pred = self.combine_cfg_noise(
            gathered[0],  # positive
            gathered[1],  # negative
            true_cfg_scale,
            cfg_normalize
        )
        return noise_pred
    else:
        return None  # Rank 1 doesn't need the result
else:
    # Sequential mode: compute both on same rank
    positive_pred = self.predict_noise(**positive_kwargs)
    negative_pred = self.predict_noise(**negative_kwargs)
    return self.combine_cfg_noise(positive_pred, negative_pred, ...)
```

### Scheduler Synchronization

`scheduler_step_maybe_with_cfg()` ensures all ranks have consistent latents:

```python
cfg_parallel_ready = (
    do_true_cfg and
    get_classifier_free_guidance_world_size() > 1
)

if cfg_parallel_ready:
    cfg_rank = get_classifier_free_guidance_rank()

    # Only rank 0 computes scheduler step
    if cfg_rank == 0:
        latents = self.scheduler_step(noise_pred, t, latents)

    # Broadcast updated latents to all ranks
    latents = latents.contiguous()
    cfg_group.broadcast(latents, src=0)
else:
    # Sequential mode: directly step
    latents = self.scheduler_step(noise_pred, t, latents)

return latents
```

---

## Customization

### Override `predict_noise()` for Custom Transformer Calls

If your transformer requires custom preprocessing or postprocessing:

```python
class MyPipelineCFGMixin(CFGParallelMixin):
    def predict_noise(self, hidden_states, **kwargs):
        """Custom transformer forward pass."""
        # Preprocess inputs
        hidden_states = self.preprocess(hidden_states)

        # Call transformer
        output = self.transformer(hidden_states, **kwargs)

        # Extract noise prediction (format varies by model)
        if isinstance(output, tuple):
            noise_pred = output[0]
        else:
            noise_pred = output.sample

        # Postprocess outputs
        noise_pred = self.postprocess(noise_pred)

        return noise_pred
```

### Override `cfg_normalize_function()` for Custom Normalization

Some models benefit from different normalization strategies:

```python
class MyPipelineCFGMixin(CFGParallelMixin):
    def cfg_normalize_function(self, noise_pred, comb_pred):
        """
        Custom CFG normalization.

        Default implementation rescales combined prediction to match
        the norm of the positive prediction, which stabilizes generation.
        """
        # Option 1: Default norm-based rescaling
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        return comb_pred * (cond_norm / noise_norm)

        # Option 2: No normalization (return as-is)
        # return comb_pred

        # Option 3: Custom rescaling
        # return comb_pred * self.cfg_rescale_factor
```

### Validate CFG Configuration

Add validation to ensure CFG parallel is correctly configured:

```python
class MyPipelineCFGMixin(CFGParallelMixin):
    def check_cfg_parallel_validity(
        self,
        true_cfg_scale: float,
        has_neg_prompt: bool
    ) -> bool:
        """
        Validate CFG parallel configuration.

        Args:
            true_cfg_scale: CFG scale (must be > 1 for CFG to work)
            has_neg_prompt: Whether negative prompt is provided

        Returns:
            True if configuration is valid, False otherwise
        """
        if get_classifier_free_guidance_world_size() == 1:
            return True  # Not using CFG parallel

        if true_cfg_scale <= 1:
            logger.warning(
                "CFG parallel requires true_cfg_scale > 1, "
                f"got {true_cfg_scale}"
            )
            return False

        if not has_neg_prompt:
            logger.warning(
                "CFG parallel requires negative prompt, but none provided"
            )
            return False

        return True
```

---

## Common Patterns

### Pattern 1: Dual-Stream Transformer (Text + Image)

**Models:** Qwen-Image, SD3, FLUX

```python
class DualStreamCFGMixin(CFGParallelMixin):
    def diffuse(self, latents, text_embeds, negative_text_embeds, ...):
        for t in timesteps:
            positive_kwargs = {
                "hidden_states": latents,           # Image stream
                "encoder_hidden_states": text_embeds,  # Text stream
                ...
            }
            negative_kwargs = {
                "hidden_states": latents,
                "encoder_hidden_states": negative_text_embeds,
                ...
            }

            noise_pred = self.predict_noise_maybe_with_cfg(...)
            latents = self.scheduler_step_maybe_with_cfg(...)

        return latents
```

### Pattern 2: Single-Stream Transformer (Unified Sequence)

**Models:** Z-Image (image + text concatenated)

```python
class SingleStreamCFGMixin(CFGParallelMixin):
    def diffuse(self, unified_latents, negative_unified_latents, ...):
        for t in timesteps:
            positive_kwargs = {
                "hidden_states": unified_latents,  # Image + text together
                ...
            }
            negative_kwargs = {
                "hidden_states": negative_unified_latents,
                ...
            }

            noise_pred = self.predict_noise_maybe_with_cfg(...)
            latents = self.scheduler_step_maybe_with_cfg(...)

        return latents
```

### Pattern 3: Image Editing Pipeline

**Models:** Qwen-Image-Edit, LongCat-Image-Edit

```python
class ImageEditCFGMixin(CFGParallelMixin):
    def diffuse(
        self,
        latents,
        image_latents,  # Condition image
        text_embeds,
        negative_text_embeds,
        ...
    ):
        for t in timesteps:
            # Concatenate noise latents with condition image latents
            latent_model_input = torch.cat([latents, image_latents], dim=1)

            positive_kwargs = {
                "hidden_states": latent_model_input,  # Concatenated
                "encoder_hidden_states": text_embeds,
                ...
            }
            negative_kwargs = {
                "hidden_states": latent_model_input,
                "encoder_hidden_states": negative_text_embeds,
                ...
            }

            # Slice output to remove condition latents
            noise_pred = self.predict_noise_maybe_with_cfg(
                ...,
                output_slice=latents.size(1),  # Only keep noise prediction
            )

            # Step only the noise latents
            latents = self.scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg
            )

        return latents
```

---

## Troubleshooting

### Issue: CFG parallel not activating

**Symptoms:** Generation still slow, logs don't show CFG parallel being used.

**Causes & Solutions:**

1. **CFG world size not set:**
   ```bash
   # Check if CFG parallel is enabled
   python -c "from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size; print(get_classifier_free_guidance_world_size())"

   # Should print 2 for CFG parallel, 1 for sequential
   ```

   **Solution:** Initialize parallel state with `cfg_parallel_size=2`:
   ```python
   from vllm_omni.diffusion.distributed import initialize_model_parallel
   initialize_model_parallel(cfg_parallel_size=2)
   ```

2. **`do_true_cfg` is False:**

   **Solution:** Ensure `guidance_scale > 1.0` and negative prompt is provided:
   ```python
   images = pipeline(
       prompt="a cat",
       negative_prompt="",  # Must provide (even if empty)
       guidance_scale=3.5,   # Must be > 1.0
   )
   ```

3. **Mixin methods not called:**

   **Solution:** Ensure pipeline calls `predict_noise_maybe_with_cfg()` instead of direct transformer calls:
   ```python
   # ❌ BAD: Direct transformer call
   noise_pred = self.transformer(latents, ...)

   # ✅ GOOD: Use mixin method
   noise_pred = self.predict_noise_maybe_with_cfg(...)
   ```

### Issue: Different results with/without CFG parallel

**Symptoms:** Images look different when CFG parallel is enabled vs disabled.

**Cause:** Incorrect rank assignment or gathering.

**Solution:** Verify rank 0 computes positive, rank 1 computes negative:
```python
# In predict_noise_maybe_with_cfg:
cfg_rank = get_classifier_free_guidance_rank()

if cfg_rank == 0:
    local_pred = self.predict_noise(**positive_kwargs)  # MUST be positive
else:
    local_pred = self.predict_noise(**negative_kwargs)  # MUST be negative
```

### Issue: Rank 1 crashes or hangs

**Symptoms:** Only rank 0 completes, rank 1 hangs or errors.

**Causes & Solutions:**

1. **Rank 1 trying to step scheduler:**

   **Solution:** Ensure only rank 0 steps:
   ```python
   if cfg_parallel_ready:
       if cfg_rank == 0:
           latents = self.scheduler_step(noise_pred, t, latents)
       # Don't step on rank 1!
       cfg_group.broadcast(latents, src=0)
   ```

2. **Missing broadcast:**

   **Solution:** Always broadcast latents after stepping:
   ```python
   latents = latents.contiguous()  # Important!
   cfg_group.broadcast(latents, src=0)
   ```

### Issue: Image editing produces artifacts

**Symptoms:** Edited images have visible artifacts or inconsistencies.

**Cause:** Condition latents included in noise prediction.

**Solution:** Use `output_slice` parameter:
```python
# Concatenate noise + condition latents
latent_model_input = torch.cat([latents, image_latents], dim=1)

# Predict noise and slice to remove condition latents
noise_pred = self.predict_noise_maybe_with_cfg(
    ...,
    output_slice=latents.size(1),  # Keep only first N channels
)
```

---

## Reference Implementations

Complete examples in the codebase:

| Pipeline | Path | Notes |
|----------|------|-------|
| **Qwen-Image** | `vllm_omni/diffusion/models/qwen_image/cfg_parallel.py` | Dual-stream transformer |
| **Qwen-Image-Edit** | `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit.py` | Image editing with `output_slice` |
| **LongCat-Image** | `vllm_omni/diffusion/models/longcat_image/pipeline_longcat_image.py` | FLUX-like dual-stream |
| **Wan2.2** | `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py` | Dual-transformer architecture |
| **CFGParallelMixin** | `vllm_omni/diffusion/distributed/cfg_parallel.py` | Base mixin implementation |

---

## Summary

Adding CFG-Parallel support to a pipeline:

1. ✅ Create pipeline-specific mixin inheriting from `CFGParallelMixin`
2. ✅ Implement `diffuse()` method with denoising loop
3. ✅ Prepare separate `positive_kwargs` and `negative_kwargs`
4. ✅ Call `predict_noise_maybe_with_cfg()` for noise prediction
5. ✅ Call `scheduler_step_maybe_with_cfg()` for synchronized latent updates
6. ✅ (Optional) Override `predict_noise()` or `cfg_normalize_function()` for custom behavior
7. ✅ (Optional) Use `output_slice` for image editing pipelines

The mixin handles all parallel logic automatically - just structure your code correctly and CFG-Parallel works out of the box!
