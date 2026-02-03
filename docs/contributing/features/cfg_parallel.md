# How to parallelize a pipeline for CFG parallel

This section describes how to add CFG-Parallel to a diffusion **pipeline**. We use the Qwen-Image pipeline (`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`) as the reference implementation.

In `QwenImagePipeline`, each diffusion step runs two denoiser forward passes sequentially:

- positive (prompt-conditioned)
- negative (negative-prompt-conditioned)

CFG-Parallel assigns these two branches to different ranks in the **CFG group** and synchronizes the results.

vLLM-omni provides `CFGParallelMixin` base class that encapsulates the CFG parallel logic. By inheriting from this mixin and calling its methods, pipelines can easily implement CFG parallel without writing repetitive code.

**Key Methods in CFGParallelMixin:**
- `predict_noise_maybe_with_cfg()`: Automatically handles CFG parallel noise prediction
- `scheduler_step_maybe_with_cfg()`: Scheduler step with automatic CFG rank synchronization

**Example Implementation:**

```python
class QwenImageCFGParallelMixin(CFGParallelMixin):
    """
    Base Mixin class for Qwen Image pipelines providing shared CFG methods.
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
        image_latents: torch.Tensor | None = None,
        cfg_normalize: bool = True,
        additional_transformer_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        self.transformer.do_true_cfg = do_true_cfg

        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            # Prepare kwargs for positive (conditional) prediction
            positive_kwargs = {
                "hidden_states": latents,
                "timestep": timestep / 1000,
                "guidance": guidance,
                "encoder_hidden_states_mask": prompt_embeds_mask,
                "encoder_hidden_states": prompt_embeds,
                "img_shapes": img_shapes,
                "txt_seq_lens": txt_seq_lens,
            }

            # Prepare kwargs for negative (unconditional) prediction
            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latents,
                    "timestep": timestep / 1000,
                    "guidance": guidance,
                    "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "img_shapes": img_shapes,
                    "txt_seq_lens": negative_txt_seq_lens,
                }
            else:
                negative_kwargs = None

            # Predict noise with automatic CFG parallel handling
            # - In CFG parallel mode: rank0 computes positive, rank1 computes negative
            # - Automatically gathers results and combines them on rank0
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=cfg_normalize,
            )

            # Step scheduler with automatic CFG synchronization
            # - Only rank0 computes the scheduler step
            # - Automatically broadcasts updated latents to all ranks
            latents = self.scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg
            )

        return latents
```

**How it works:**
1. Prepare separate `positive_kwargs` and `negative_kwargs` for conditional and unconditional predictions
2. Call `predict_noise_maybe_with_cfg()` which:
   - Detects if CFG parallel is enabled (`get_classifier_free_guidance_world_size() > 1`)
   - Distributes computation: rank0 processes positive, rank1 processes negative
   - Gathers predictions and combines them using `combine_cfg_noise()` on rank0
   - Returns combined noise prediction (only valid on rank0)
3. Call `scheduler_step_maybe_with_cfg()` which:
   - Only rank0 computes the scheduler step
   - Broadcasts the updated latents to all ranks for synchronization

**How to customize**

Some pipelines may need to customize the following functions in `CFGParallelMixin`:
1. You may need to edit `predict_noise` function for custom behaviors.
```python
def predict_noise(self, *args, **kwargs):
    """
    Forward pass through transformer to predict noise.

    Subclasses should override this if they need custom behavior,
    but the default implementation calls self.transformer.
    """
    return self.transformer(*args, **kwargs)[0]

```
2. The default normalization function after combining the noise predictions from both branches is as follows. You may need to customize it.
```python
def cfg_normalize_function(self, noise_pred, comb_pred):
    """
    Normalize the combined noise prediction.

    Args:
        noise_pred: positive noise prediction
        comb_pred: combined noise prediction after CFG

    Returns:
        Normalized noise prediction tensor
    """
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    noise_pred = comb_pred * (cond_norm / noise_norm)
    return noise_pred
```
