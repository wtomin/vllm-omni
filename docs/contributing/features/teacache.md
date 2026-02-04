# How to add TeaCache support for a new model

This section describes how to add TeaCache support to a new diffusion **transformer model**. We use the Qwen-Image transformer (`vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py`) as the reference implementation.

TeaCache (Timestep Embedding Aware Cache) speeds up diffusion inference by caching transformer block computations when consecutive timesteps are similar. It provides **1.5x-2.0x speedup** with minimal quality loss.

## Overview

vLLM-omni provides a **hook-based** TeaCache system that requires **zero changes to model code**. The hook completely intercepts the transformer's forward pass and implements adaptive caching transparently.

To add TeaCache support for a new model, you only need to:

1. Write an **extractor function** that returns a `CacheContext` object
2. Register the extractor in the `EXTRACTOR_REGISTRY`
3. Add model-specific polynomial coefficients to `TeaCacheConfig`

The TeaCache hook handles all caching logic automatically, including:
- CFG-aware state management (separate states for positive/negative branches)
- CFG-parallel compatibility
- L1 distance computation with polynomial rescaling
- Residual caching and reuse

---

## Understanding CacheContext

The `CacheContext` dataclass encapsulates all model-specific information needed for caching:

```python
@dataclass
class CacheContext:
    """
    Context object containing all model-specific information for caching.

    Attributes:
        modulated_input: Tensor used for cache decision (similarity comparison).
            Extracted from the first transformer block after normalization and modulation.

        hidden_states: Current hidden states (will be modified by caching).
            Main image/latent states after preprocessing but before transformer blocks.

        encoder_hidden_states: Optional encoder states (for dual-stream models).
            Set to None for single-stream models (e.g., Flux).
            For dual-stream models (e.g., Qwen), contains text encoder outputs.

        temb: Timestep embedding tensor.
            Contains the timestep conditioning.

        run_transformer_blocks: Callable that executes model-specific transformer blocks.
            Signature: () -> tuple[torch.Tensor, ...]
            Returns: (hidden_states, [encoder_hidden_states])

        postprocess: Callable that does model-specific output postprocessing.
            Signature: (torch.Tensor) -> Union[torch.Tensor, Transformer2DModelOutput, tuple]
            Applies final transformations (normalization, projection) to produce model output.

        extra_states: Optional dict for additional model-specific state.
    """
```

---

## Step-by-Step: Writing an Extractor

### Step 1: Model-Specific Preprocessing

Extract and process model inputs. This typically involves:
- Embedding image/latent inputs
- Processing text encoder outputs (if dual-stream)
- Creating timestep embeddings
- Computing positional embeddings

**Example (Qwen-Image):**

```python
def extract_qwen_context(
    module: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    timestep: torch.Tensor,
    img_shapes: torch.Tensor,
    txt_seq_lens: torch.Tensor,
    guidance: torch.Tensor | None = None,
    **kwargs: Any,
) -> CacheContext:
    # Validate model structure
    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks")

    # Preprocessing: embed inputs
    hidden_states = module.img_in(hidden_states)
    timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)
    encoder_hidden_states = module.txt_norm(encoder_hidden_states)
    encoder_hidden_states = module.txt_in(encoder_hidden_states)

    # Create timestep embedding
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    temb = (
        module.time_text_embed(timestep, hidden_states)
        if guidance is None
        else module.time_text_embed(timestep, guidance, hidden_states)
    )

    # Compute position embeddings
    image_rotary_emb = module.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
```

### Step 2: Extract Modulated Input

The modulated input is used for cache decisions. Extract it from the **first transformer block** after normalization and modulation.

**Example (Qwen-Image):**

```python
    # Extract modulated input from first transformer block
    block = module.transformer_blocks[0]
    img_mod_params = block.img_mod(temb)
    img_mod1, _ = img_mod_params.chunk(2, dim=-1)
    img_modulated, _ = block.img_norm1(hidden_states, img_mod1)
```

**Key Points:**
- Use the **first block** to extract modulated input early
- Apply the same normalization and modulation as the actual forward pass
- The tensor should represent the processed features that will change across timesteps

### Step 3: Define Transformer Execution

Create a callable that executes all transformer blocks. This encapsulates the main computation loop.

**Example (Qwen-Image dual-stream):**

```python
    def run_transformer_blocks():
        """Execute all Qwen transformer blocks."""
        h = hidden_states
        e = encoder_hidden_states

        for block in module.transformer_blocks:
            e, h = block(
                hidden_states=h,
                encoder_hidden_states=e,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        return (h, e)  # Return both image and text hidden states
```

**Example (Single-stream model like Flux):**

```python
    def run_transformer_blocks():
        """Execute all Flux transformer blocks."""
        h = hidden_states

        for block in module.transformer_blocks:
            h = block(h, temb=temb)
        return (h,)  # Return only image hidden states
```

**Key Points:**
- Return format: `(hidden_states, [encoder_hidden_states])`
- For single-stream models: return `(hidden_states,)`
- For dual-stream models: return `(hidden_states, encoder_hidden_states)`

### Step 4: Define Postprocessing

Create a callable that applies final transformations to produce the model output.

**Example (Qwen-Image):**

```python
    return_dict = kwargs.get("return_dict", True)

    def postprocess(h):
        """Apply Qwen-specific output postprocessing."""
        h = module.norm_out(h, temb)
        output = module.proj_out(h)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
```

**Key Points:**
- Apply final normalization
- Apply output projection
- Return in the format expected by the pipeline

### Step 5: Return CacheContext

Package all information into a `CacheContext` object.

```python
    return CacheContext(
        modulated_input=img_modulated,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,  # or None for single-stream
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )
```

---

## Step 6: Register the Extractor

Add your extractor to the `EXTRACTOR_REGISTRY` in `vllm_omni/diffusion/cache/teacache/extractors.py`:

```python
EXTRACTOR_REGISTRY: dict[str, Callable] = {
    "QwenImageTransformer2DModel": extract_qwen_context,
    "Bagel": extract_bagel_context,
    "ZImageTransformer2DModel": extract_zimage_context,
    "YourModelTransformer2DModel": extract_your_model_context,  # Add here
}
```

**Key:** Use the transformer class name (`module.__class__.__name__`)

---

## Step 7: Add Model Coefficients

Add polynomial rescaling coefficients to `vllm_omni/diffusion/cache/teacache/config.py`:

```python
_MODEL_COEFFICIENTS = {
    "FluxTransformer2DModel": [
        4.98651651e02,
        -2.83781631e02,
        5.58554382e01,
        -3.82021401e00,
        2.64230861e-01,
    ],
    "QwenImageTransformer2DModel": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    "YourModelTransformer2DModel": [  # Add your model's coefficients
        # 5 polynomial coefficients (can reuse similar model's coefficients initially)
    ],
}
```

**Initial approach:** Start with coefficients from a similar model architecture, then tune empirically if needed.

---



## Coefficient Estimation

While you can start with coefficients from a similar model architecture, estimating custom coefficients for your specific model typically improves TeaCache performance. This section shows how to estimate optimal polynomial coefficients.

### Why Estimate Coefficients?

The polynomial coefficients rescale L1 distances between consecutive modulated inputs to better predict when cached residuals can be reused. Model-specific coefficients account for:

- Architecture differences (layer count, hidden size, attention patterns)
- Training data characteristics
- Noise prediction behavior across timesteps

**Default vs. Custom Coefficients:**
- **Using defaults:** Works reasonably well (within 5-10% of optimal)
- **Estimating custom:** Provides best performance, especially for novel architectures

### Step 1: Implement Data Collection Adapter

Add an adapter in `vllm_omni/diffusion/cache/teacache/coefficient_estimator.py` to support your model:

```python
class YourModelAdapter:
    """Adapter for coefficient estimation on your model."""

    @staticmethod
    def load_pipeline(model_path: str, device: str, dtype: torch.dtype) -> Any:
        """
        Load your diffusion pipeline.

        Args:
            model_path: Path to model weights
            device: Device to load on ("cuda", "cpu")
            dtype: Model dtype (torch.float16, torch.bfloat16, torch.float32)

        Returns:
            Loaded pipeline instance
        """
        from your_model_package import YourModelPipeline

        pipeline = YourModelPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )
        pipeline = pipeline.to(device)
        return pipeline

    @staticmethod
    def get_transformer(pipeline: Any) -> tuple[Any, str]:
        """
        Extract transformer from pipeline.

        Args:
            pipeline: Pipeline instance

        Returns:
            (transformer_module, transformer_class_name)
        """
        return pipeline.transformer, "YourTransformer2DModel"

    @staticmethod
    def install_hook(transformer: Any, hook: DataCollectionHook) -> None:
        """
        Install data collection hook on transformer.

        Args:
            transformer: Transformer module
            hook: DataCollectionHook instance to install
        """
        from vllm_omni.diffusion.hooks import HookRegistry

        registry = HookRegistry.get_or_create(transformer)
        registry.register_hook(hook._HOOK_NAME, hook)


# Register your adapter
_MODEL_ADAPTERS["YourModel"] = YourModelAdapter
```

**Key Points:**
- `load_pipeline()`: Should return a working pipeline that can generate images
- `get_transformer()`: Must return the exact transformer used in TeaCache extractor
- `install_hook()`: Standard hook installation (same pattern for all models)

### Step 2: Collect Data from Diverse Prompts

Use diverse prompts to capture different generation scenarios:

```python
from vllm_omni.diffusion.cache.teacache.coefficient_estimator import (
    TeaCacheCoefficientEstimator,
)
from datasets import load_dataset
from tqdm import tqdm
import torch

# Initialize estimator
estimator = TeaCacheCoefficientEstimator(
    model_path="/path/to/your/model",
    model_type="YourModel",  # Must match key in _MODEL_ADAPTERS
    device="cuda",
    dtype=torch.float16,
)

# Load diverse prompts (paper recommends ~70 prompts)
# Using Parti prompts as an example
dataset = load_dataset("nateraw/parti-prompts", split="train")
prompts = dataset["Prompt"][:70]

# Collect modulated input differences across timesteps
print("Collecting data from prompts...")
for prompt in tqdm(prompts, desc="Processing prompts"):
    estimator.collect_from_prompt(
        prompt=prompt,
        num_inference_steps=50,  # Standard step count
        # Optional: guidance_scale=3.5, negative_prompt="", etc.
    )

print(f"Collected {len(estimator.data_pairs)} data points")
```

**Best Practices:**
- **Prompt diversity:** Use 70-100 prompts covering different subjects, styles, complexity
- **Inference steps:** Use your typical generation settings (20-50 steps)
- **Multiple datasets:** Combine Parti, COCO, custom prompts for better coverage

**Alternative prompt sources:**
```python
# Option 1: COCO captions
dataset = load_dataset("HuggingFaceM4/COCO", split="train")
prompts = [item["sentences"]["raw"][0] for item in dataset.select(range(100))]

# Option 2: Custom prompts
prompts = [
    "a cat sitting on a windowsill",
    "abstract geometric patterns in vibrant colors",
    "photorealistic portrait of a person",
    "minimalist landscape at sunset",
    # ... add 70+ diverse prompts
]
```

### Step 3: Estimate Polynomial Coefficients

Fit a 4th-order polynomial to the collected data:

```python
# Estimate coefficients (4th order polynomial)
coeffs = estimator.estimate(poly_order=4)

print(f"\nEstimated coefficients: {coeffs}")
print("\nCopy these coefficients to config.py:")
print(f"_MODEL_COEFFICIENTS['YourTransformer2DModel'] = {coeffs.tolist()}")
```

**Expected Output:**
```
Data statistics:
Count: 3450
Input Diffs (x): min=1.11e-02, max=5.26e-02, mean=2.84e-02, std=8.73e-03
Output Diffs (y): min=2.82e-02, max=2.98e-01, mean=7.03e-02, std=4.21e-02

Estimated coefficients: [1333131.29, -168644.23, 7950.51, -163.75, 1.26]

Copy these coefficients to config.py:
_MODEL_COEFFICIENTS['YourTransformer2DModel'] = [1333131.29, -168644.23, 7950.51, -163.75, 1.26]
```

### Step 4: Interpret and Validate Results

**Understanding the Statistics:**

| Metric | What to Check | Good Range | Warning Signs |
|--------|---------------|------------|---------------|
| **Count** | Number of timestep pairs | 2000-5000+ | < 1000: too few prompts |
| **Input Diffs (x)** | Modulated input changes | 0.01-0.10 | Very small (<0.001): model may not modulate properly |
| **Output Diffs (y)** | Residual changes | Should correlate with x | No correlation: check extractor implementation |
| **Coefficient magnitude** | Polynomial scale | -1e6 to 1e6 | > 1e8: numerical instability |

**Validation Checklist:**

1. **Sufficient data points:**
   ```python
   if len(estimator.data_pairs) < 2000:
       print("Warning: Few data points. Consider:")
       print("- Increase num_inference_steps (try 50-100)")
       print("- Add more prompts (100+)")
   ```

2. **Reasonable coefficient magnitude:**
   ```python
   import numpy as np
   if np.any(np.abs(coeffs) > 1e8):
       print("Warning: Extremely large coefficients.")
       print("This may cause numerical issues. Try:")
       print("- Collect more diverse data")
       print("- Check extractor implementation")
   ```

3. **Visual inspection (optional):**
   ```python
   import matplotlib.pyplot as plt

   # Plot data and fitted polynomial
   estimator.plot_fit()
   plt.savefig("coefficient_fit.png")
   ```

**Troubleshooting Common Issues:**

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Poor fit** | Scattered points, low R² | Add more diverse prompts (100+) |
| **Large coefficients** | Values > 1e8 | Collect more data, check for outliers |
| **Inconsistent results** | Coefficients vary significantly between runs | Use more prompts, ensure reproducibility |
| **No correlation** | Output diffs don't correlate with input diffs | Check extractor extracts correct modulated input |

### Step 5: Add Coefficients to Configuration

Add the estimated coefficients to `vllm_omni/diffusion/cache/teacache/config.py`:

```python
_MODEL_COEFFICIENTS = {
    "FluxTransformer2DModel": [
        4.98651651e02,
        -2.83781631e02,
        5.58554382e01,
        -3.82021401e00,
        2.64230861e-01,
    ],
    "QwenImageTransformer2DModel": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    "YourTransformer2DModel": [  # Add your estimated coefficients
        1.33313129e06,  # a₄ (x⁴ coefficient)
        -1.68644226e05, # a₃ (x³ coefficient)
        7.95050740e03,  # a₂ (x² coefficient)
        -1.63747873e02, # a₁ (x coefficient)
        1.26352397e00,  # a₀ (constant)
    ],
}
```

**Coefficient Interpretation:**

The polynomial rescales input differences to predict output differences:
```
rescaled_diff = a₄*x⁴ + a₃*x³ + a₂*x² + a₁*x + a₀
```

Where `x` is the relative L1 distance between consecutive modulated inputs.

**Sign patterns:**
- Positive high-order terms (a₄, a₃): Amplify large differences
- Negative mid-order terms: Create non-linear response
- Small constant (a₀): Base similarity threshold

### Step 6: Validate Performance

Test the estimated coefficients against the baseline:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
import time

# Test with estimated coefficients
omni_custom = Omni(
    model="your-model-name",
    cache_backend="tea_cache",
    cache_config={
        "rel_l1_thresh": 0.2,
        "coefficients": [1.33e6, -1.69e5, 7.95e3, -1.64e2, 1.26],  # Your coefficients
    }
)

# Baseline (no cache)
omni_baseline = Omni(model="your-model-name")

# Compare speed and quality
test_prompts = ["a cat", "a landscape", "abstract art"]

for prompt in test_prompts:
    # With TeaCache
    start = time.time()
    img_cached = omni_custom.generate(
        prompt, OmniDiffusionSamplingParams(num_inference_steps=50)
    )
    time_cached = time.time() - start

    # Without cache
    start = time.time()
    img_baseline = omni_baseline.generate(
        prompt, OmniDiffusionSamplingParams(num_inference_steps=50)
    )
    time_baseline = time.time() - start

    speedup = time_baseline / time_cached
    print(f"{prompt}: {speedup:.2f}x speedup")

    # Optionally: compute image similarity (LPIPS, SSIM, etc.)
```

**Expected Results:**
- **Speedup:** 1.5x-2.0x depending on `rel_l1_thresh`
- **Quality:** Visually identical (LPIPS < 0.01, SSIM > 0.99)
- **Consistency:** Similar results across different prompts

### Quick Start Script

Complete script for coefficient estimation:

```python
#!/usr/bin/env python3
"""
Estimate TeaCache coefficients for a new model.

Usage:
    python estimate_coefficients.py --model_path /path/to/model --model_type YourModel
"""
import argparse
from vllm_omni.diffusion.cache.teacache.coefficient_estimator import (
    TeaCacheCoefficientEstimator,
)
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--num_prompts", type=int, default=70)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Initialize estimator
    estimator = TeaCacheCoefficientEstimator(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
    )

    # Load prompts
    dataset = load_dataset("nateraw/parti-prompts", split="train")
    prompts = dataset["Prompt"][:args.num_prompts]

    # Collect data
    for prompt in tqdm(prompts, desc="Collecting data"):
        estimator.collect_from_prompt(
            prompt=prompt,
            num_inference_steps=args.num_steps,
        )

    # Estimate coefficients
    coeffs = estimator.estimate(poly_order=4)

    # Print results
    print(f"\n{'='*60}")
    print(f"Estimated Coefficients for {args.model_type}")
    print(f"{'='*60}")
    print(f"\nAdd to config.py:")
    print(f'_MODEL_COEFFICIENTS["{args.model_type}"] = [')
    for i, c in enumerate(coeffs):
        print(f"    {c:.8e},  # a_{4-i}")
    print("]")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
```

---


## Testing

After adding teacache support, test with:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="your-model-name",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2}
)

images = omni.generate(
    "a beautiful landscape",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

See more detailed examples in [user guide for teacache](../../user_guide/diffusion/teacache.md).

---

## Troubleshooting

### Error: "Unknown model type"

**Cause:** Extractor not registered or transformer class name mismatch.

**Solution:**
1. Check `pipeline.transformer.__class__.__name__` matches registry key
2. Verify extractor is in `EXTRACTOR_REGISTRY`

### Error: "Cannot find coefficients"

**Cause:** Missing model coefficients in `_MODEL_COEFFICIENTS`.

**Solution:**
1. Add coefficients to `config.py`
2. Or pass custom coefficients via `cache_config={"coefficients": [...]}`

### Quality Degradation

**Cause:** `rel_l1_thresh` too high or coefficients not tuned.

**Solution:**
1. Lower `rel_l1_thresh` (try 0.1-0.2)
2. Tune polynomial coefficients empirically for your model

---

## Reference Implementations

See these files for complete examples:

- **Dual-stream (Qwen-Image):** `vllm_omni/diffusion/cache/teacache/extractors.py::extract_qwen_context`
- **Omni model (Bagel):** `vllm_omni/diffusion/cache/teacache/extractors.py::extract_bagel_context`

---

## Summary

Adding TeaCache support requires:

1. ✅ Write extractor function returning `CacheContext`
2. ✅ Register in `EXTRACTOR_REGISTRY`
3. ✅ Add model coefficients to `_MODEL_COEFFICIENTS`
4. ✅ Test with `cache_backend="tea_cache"`
