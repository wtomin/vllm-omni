# PR #1924 Review: [New Model][Skill Eval] Add Stable Diffusion v1.5 as first UNet-based diffusion model

**URL:** https://github.com/vllm-project/vllm-omni/pull/1924  
**Author:** SamitHuang  
**Reviewed:** 2026-03-17  
**Decision:** REQUEST_CHANGES

---

## Gates

| Check | Status |
|-------|--------|
| DCO | ✅ SUCCESS |
| pre-commit | ✅ SUCCESS |
| Build (3.11 / 3.12) | ✅ SUCCESS |
| ReadTheDocs | ✅ SUCCESS |
| Mergeable | ✅ MERGEABLE |

All gates pass. Full review proceeds.

---

## What I Validated

- **PR body:** offline generation script ✅, sample image attached ✅, VRAM reported (2.0 GiB) ✅  
- **Code:** `CFGParallelMixin` used for CFG Parallel ✅, no diffusers pipeline mixins ✅  
- **Entrypoints:** `unet/config.json` fallback when `transformer/config.json` is absent ✅  
- **Docs:** parallelism / feature table row added ✅, offline usage example added to `text_to_image.md` ✅  
- **Memory optimizations in diff:** none found 🔴  
- **Test files added:** none found 🔴  

---

## Blocking Issues

### 🔴 1. No memory optimization feature

The diff has no CPU offload, quantization, or VAE tiling. The skill checklist requires at least one memory optimization for all new diffusion models.

> **Required action:** Implement at least one of: cpu-offload, quantization (int8/fp8), or VAE tiling.  
> VAE tiling is the most natural fit (splits the spatial VAE decode into tiles, reduces peak VRAM for larger resolutions). CPU offload is also straightforward.  
> Document the VRAM reduction in the PR body.

---

### 🔴 2. No e2e online serving test

No files under `tests/` were added. An e2e online serving test is required before merge.

> **Required action:** Add `tests/e2e/online_serving/test_sd15_expansion.py` (or equivalent) that:
> 1. Starts `vllm serve stable-diffusion-v1-5/stable-diffusion-v1-5 --omni`
> 2. Sends a generation request via the API endpoint
> 3. Asserts the response contains a valid image output

---

### 🔴 3. Online serving not demonstrated in PR body

The PR body only shows offline `Omni` usage. The required template asks for both offline and online evidence.

> **Required action:** Add a `vllm serve` command and a sample result showing the model works end-to-end through the API.

---

### 🔴 4. GPU model missing from performance measurements

The test result reports `26.87 it/s` and `2.0 GiB` but does not name the GPU model or count.

> **Required action:** Update the PR description with the full hardware spec (GPU model, GPU count, resolution, `num_inference_steps`). Example: `1× NVIDIA A100 40GB, 512×512, 20 steps`.

---

### 🔴 5. Step count discrepancy: script says 20, logs show 51

The generation script in the PR uses `num_inference_steps=20`, but the log shows `51/51 [00:01<00:00, 31.06it/s]`. PNDM with 20 steps produces 20 timesteps, not 51 (51 matches 50 inference steps + 1 PNDM init). The log was captured from a different run than the script shown in the PR.

> **Required action:** Re-run with the exact script shown in the PR body and update the log output and metrics to match.

---

## Code Bugs (inline comments)

### `vllm_omni/diffusion/models/sd1_5/pipeline_sd1_5.py` — line 78

```python
variant = "fp16"
```

`variant` is hardcoded regardless of `od_config.dtype`. If the user initializes with `dtype='bfloat16'` or `dtype='float32'`, this still requests fp16 variant weights, which may not exist or silently loads the wrong type.

> **Fix:**
> ```python
> variant = "fp16" if od_config.dtype == torch.float16 else None
> ```
> This was also raised by @gcanlin. The same fix should propagate to the `add-diffusion-model` skill template.

---

### `vllm_omni/diffusion/models/sd1_5/pipeline_sd1_5.py` — line 214

```python
latents = self.prepare_latents(
    batch_size=1,
    ...
)
```

`batch_size=1` is hardcoded. Requests with `OmniDiffusionSamplingParams(num_outputs_per_prompt=4)` silently generate only one image. The Codex bot also flagged this.

> **Fix:** Pass `sp.num_outputs_per_prompt` (defaulting to 1), and replicate `prompt_embeds` / `negative_prompt_embeds` accordingly:
> ```python
> num_images = getattr(sp, 'num_outputs_per_prompt', 1) or 1
> latents = self.prepare_latents(
>     batch_size=num_images,
>     ...
> )
> ```

---

## Recommended (not blocking)

### 💡 diffusers baseline comparison

Adding a side-by-side comparison would strengthen the PR significantly:

- diffusers generation script (same prompt, same resolution, same steps, same GPU)
- diffusers sample output vs vLLM-Omni output
- latency comparison: `X.X s (vLLM-Omni)` vs `X.X s (diffusers)`
- VRAM comparison

---

## Checklist Summary

| Dimension | Item | Status |
|-----------|------|--------|
| **Body** | Generation script (offline) | ✅ |
| **Body** | Sample output | ✅ |
| **Body** | e2e latency | ⚠️ GPU model missing |
| **Body** | Peak VRAM | ✅ |
| **Body** | Online serving demo | 🔴 Missing |
| **Body** | diffusers baseline | 💡 Recommended |
| **Code** | Offline inference (`Omni`) | ✅ |
| **Code** | Online serving (`AsyncOmni`) | ⚠️ Entrypoint fixed but not demonstrated |
| **Code** | No diffusers pipeline mixins | ✅ |
| **Code** | Acceleration feature (CFG Parallel) | ✅ |
| **Code** | Memory optimization | 🔴 Missing |
| **Docs** | Feature / parallelism table | ✅ |
| **Docs** | Usage example doc | ✅ (offline only) |
| **Docs** | Model support table | ⚠️ Not found in diff |
| **Tests** | e2e online serving test | 🔴 Missing |
| **Tests** | Offline inference test | 🔴 Missing |
