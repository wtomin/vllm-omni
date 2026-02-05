# How to add CFG-Parallel support for a new pipeline

This section describes how to add CFG-Parallel (Classifier-Free Guidance Parallel) to a diffusion **pipeline**. We use the Qwen-Image pipeline (`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`) as the reference implementation.

CFG-Parallel accelerates diffusion inference by distributing the conditional (positive) and unconditional (negative) forward passes to different GPU ranks.

## Overview

### What is CFG-Parallel?

In standard Classifier-Free Guidance, each diffusion step requires two forward passes through the transformer:

1. **Positive/Conditional**: Guided by the text prompt
2. **Negative/Unconditional**: Typically using empty or negative prompt

---

### Architecture

vLLM-omni provides `CFGParallelMixin` that encapsulates all CFG parallel logic. Pipelines inherit from this mixin and implement a `diffuse()` method that orchestrates the denoising loop.


| Method | Purpose | Automatic Behavior |
|--------|---------|-------------------|
| `predict_noise_maybe_with_cfg()` | Predict noise with CFG | Detects parallel mode, distributes computation, gathers results |
| `scheduler_step_maybe_with_cfg()` | Step scheduler with sync | Rank 0 steps, broadcasts latents to all ranks |
| `combine_cfg_noise()` | Combine positive/negative | Applies CFG formula with optional normalization |
| `predict_noise()` | Forward pass wrapper | Override for custom transformer calls |
| `cfg_normalize_function()` | Normalize CFG output | Override for custom normalization |

---

### How It Works

`predict_noise_maybe_with_cfg()` automatically detects and switches between two execution modes:

- **CFG-Parallel mode** (when `cfg_world_size > 1`):
  - Rank 0 computes positive prompt prediction
  - Rank 1 computes negative prompt prediction
  - Results are gathered via `all_gather()`
  - Combined on rank 0 using CFG formula

- **Sequential mode** (when `cfg_world_size == 1`):
  - Single rank computes both positive and negative predictions
  - Directly combines them with CFG formula


`scheduler_step_maybe_with_cfg()` ensures consistent latent states across all ranks:

- **CFG-Parallel mode**:
  - Only rank 0 performs the scheduler step (applies noise prediction to update latents)
  - Updated latents are broadcast to all other ranks via `broadcast()`
  - All ranks maintain synchronized latent states for the next iteration

- **Sequential mode**:
  - Single rank directly performs the scheduler step
  - No synchronization needed

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
- Pass both `positive_kwargs` and `negative_kwargs` to `predict_noise_maybe_with_cfg()`
- For image editing pipelines, `self.predict_noise_maybe_with_cfg(..., output_slice=image_seq_len)` is required. If `output_slice` is set, slice output to `[:, :output_slice]` to extract the generative image.
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
        # Encode prompts, Initialize latents, Get timesteps
        ...
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

```

## Customization

### Override `predict_noise()` for Custom Transformer Calls

If your transformer requires custom prediction function, you can rewrite `predict_noise` function. Taking wan2.2 as an example, which has two transformer models.

```python
class Wan22Pipeline(nn.Module, CFGParallelMixin):
    def predict_noise(self, current_model: nn.Module | None = None, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass through transformer to predict noise.

        Args:
            current_model: The transformer model to use (transformer or transformer_2)
            **kwargs: Arguments to pass to the transformer

        Returns:
            Predicted noise tensor
        """
        if current_model is None:
            current_model = self.transformer
        return current_model(**kwargs)[0]
```

### Override `cfg_normalize_function()` for Custom Normalization

Some models has its own normalization function. Taking LongCat Image model as an example

```python
class LongCatImagePipeline(nn.Module, CFGParallelMixin):
    def cfg_normalize_function(self, noise_pred, comb_pred, cfg_renorm_min=0.0):
        """
        Normalize the combined noise prediction.
        """
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        noise_pred = comb_pred * scale
        return noise_pred

        # The original cfg_normalize_function function in CFGParallelMixin
        # cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        # noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        # noise_pred = comb_pred * (cond_norm / noise_norm)
        # return noise_pred
```



## Testing


Taking text-to-image as an example:
```bash
cd examples/offline_inference/text_to_image
python text_to_image.py \
    --model Your-org/your-model \
    --prompt "a cup of coffee on the table" \
    --negative_prompt "ugly, unclear" \
    --cfg_scale 4.0 \
    --num_inference_steps 50 \
    --output "cfg_enabled.png" \
    --cfg_parallel_size 2
```
Please record the "e2e_time_ms" in the log and the generated result, and compare them with the results of CFG-Parallel not enabled. Please record the comparison results in your PR.

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

2. **`CFG is not enabled**

   **Solution:** Ensure `guidance_scale > 1.0` and negative prompt is provided:
   ```python
   images = pipeline(
       prompt="a cat",
       negative_prompt="",  # Must provide (even if empty)
       guidance_scale=3.5,   # Must be > 1.0
   )
   ```


## Reference Implementations

Complete examples in the codebase:

| Pipeline | Path | Notes |
|----------|------|-------|
| **Qwen-Image** | `vllm_omni/diffusion/models/qwen_image/cfg_parallel.py` | Dual-stream transformer |
| **Qwen-Image-Edit** | `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit.py` | Image editing with `output_slice` |
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
