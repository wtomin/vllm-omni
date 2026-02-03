# How to Add a Diffusion Model to vLLM-Omni

This comprehensive guide walks you through adding a new diffusion model to vLLM-Omni. We use Qwen-Image as the primary example, with references to other models (LongCat, Flux, Wan2.2) to illustrate different patterns.

vLLM-Omni provides a high-performance inference engine for diffusion models with support for batching, parallelism, and acceleration techniques. Adding a model involves adapting it from HuggingFace Diffusers to vLLM-Omni's optimized framework.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Directory Structure](#directory-structure)
4. [Basic Implementation](#basic-implementation)
   - [Step 1: Adapt Transformer Model](#step-1-adapt-transformer-model)
   - [Step 2: Adapt Pipeline](#step-2-adapt-pipeline)
   - [Step 3: Register Model](#step-3-register-model)
   - [Step 4: Add Example Script](#step-4-add-example-script)
   - [Step 5: Test Your Implementation](#step-5-test-your-implementation)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [Pull Request Checklist](#pull-request-checklist)
8. [Reference Implementations](#reference-implementations)

---

## Overview

### Why Adapt Models?

When adding a diffusion model from HuggingFace Diffusers to vLLM-Omni, adaptation is required for:

1. **Performance Optimization**
   - Replace standard attention with vLLM-Omni's optimized attention backends
   - Enable batched inference for higher throughput
   - Support various parallelism strategies (TP, SP, CFG-Parallel)

2. **Framework Integration**
   - Follow vLLM-Omni's parameter passing mechanisms
   - Use unified request/response interfaces
   - Support online serving and offline inference

3. **Advanced Features**
   - Cache acceleration (TeaCache, Cache-DiT)
   - Dynamic compilation with `torch.compile`
   - Quantization support

### Execution Flow

vLLM-Omni's diffusion inference follows this architecture:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/vllm-omni-diffusion-flow.png">
    <img alt="Diffusion Flow" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/vllm-omni-diffusion-flow.png" width=55%>
  </picture>
</p>

**Key Components:**
1. **Request Handling:** User prompts → `OmniDiffusionRequest`
2. **Pipeline Execution:** Request → Model forward pass → Output
3. **Post-processing:** Latents → Images (VAE decoding + image processing)

---

## Prerequisites

Before starting, ensure you have:

- ✅ **Working Diffusers implementation** - Model should work in standard Diffusers
- ✅ **Model weights** - Available on HuggingFace Hub or locally
- ✅ **Understanding of model architecture** - Know transformer structure, attention patterns
- ✅ **vLLM-Omni development environment** - Set up according to [installation guide](../../getting_started/installation.md)

**Recommended Knowledge:**
- Diffusion model basics (denoising, schedulers, CFG)
- PyTorch module structure
- HuggingFace Transformers/Diffusers APIs

---

## Directory Structure

Organize your model files following this structure:

```
vllm_omni/
├── diffusion/
│   ├── registry.py                          # ← Register your model here
│   ├── request.py                           # Request data structures
│   └── models/
│       └── your_model_name/                 # ← Create this directory
│           ├── __init__.py                  # Export pipeline and transformer
│           ├── pipeline_xxx.py              # Pipeline implementation
│           ├── xxx_transformer.py           # Transformer implementation
│           └── cfg_parallel.py              # (Optional) CFG-Parallel mixin
│
├── examples/
│   ├── offline_inference/
│   │   └── text_to_image/
│   │       └── your_model_example.py        # ← Add example script
│   └── online_serving/
│       └── serve_your_model.py              # (Optional) Serving example
│
└── tests/
    └── e2e/
        └── offline_inference/
            └── test_your_model.py           # ← Add tests
```

**Naming Conventions:**
- **Model directory:** `your_model_name` (lowercase, underscores)
  - Examples: `qwen_image`, `flux`, `longcat_image`, `wan2_2`
- **Pipeline file:** `pipeline_xxx.py` where `xxx` describes the task
  - Examples: `pipeline_qwen_image.py`, `pipeline_qwen_image_edit.py`
- **Transformer file:** `xxx_transformer.py` matching transformer class name
  - Examples: `qwen_image_transformer.py`, `flux_transformer.py`

---

## Basic Implementation

This section covers the minimal steps to get a model working in vLLM-Omni with basic features (online/offline serving, batch requests).

### Step 1: Adapt Transformer Model

The transformer is the core denoising network. Start by copying the transformer implementation from Diffusers and making these adaptations.

#### 1.1: Create Model Directory and Copy Files

```bash
# Create model directory
mkdir -p vllm_omni/diffusion/models/your_model_name

# Copy transformer from Diffusers (example for Qwen-Image)
cp path/to/diffusers/src/diffusers/models/transformers/transformer_qwen_image.py \
   vllm_omni/diffusion/models/your_model_name/your_model_transformer.py
```

#### 1.2: Remove Diffusers Mixins

Diffusers' `Mixin` classes are not needed in vLLM-Omni. Remove them:

```diff
# Before (Diffusers)
- from diffusers.models.modeling_utils import ModelMixin
- from diffusers.models.attention_processor import AttentionProcessor, AttentionModuleMixin

- class YourModelTransformer2DModel(ModelMixin, AttentionModuleMixin):
+ class YourModelTransformer2DModel(nn.Module):
    """Your transformer model."""
```

**Common mixins to remove:**
- `ModelMixin` - Weight loading utilities (vLLM-Omni has its own)
- `AttentionModuleMixin` - Attention processors (using vLLM-Omni's Attention layer instead)
- `ConfigMixin` - Config management (using `OmniDiffusionConfig` instead)

#### 1.3: Replace Attention Implementation

**The most important adaptation:** Replace Diffusers' attention with vLLM-Omni's optimized `Attention` layer.

**Before (Diffusers):**
```python
from diffusers.models.attention_processor import dispatch_attention_fn

class YourAttentionBlock(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states=None, ...):
        # Query, key, value projections
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states or hidden_states)
        value = self.to_v(encoder_hidden_states or hidden_states)

        # Attention computation
        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
```

**After (vLLM-Omni):**
```python
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

class YourAttentionBlock(nn.Module):
    def __init__(self, ...):
        super().__init__()

        # Initialize vLLM-Omni's Attention layer
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),  # Attention scaling
            causal=False,  # Diffusion models typically use bidirectional attention
            num_kv_heads=self.num_kv_heads,  # For GQA/MQA, set to num_heads for MHA
        )

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, ...):
        # Query, key, value projections
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states or hidden_states)
        value = self.to_v(encoder_hidden_states or hidden_states)

        # Reshape for multi-head attention: [B, seq, hidden] → [B, seq, num_heads, head_dim]
        batch_size, seq_len = query.shape[:2]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Create attention metadata
        attn_metadata = AttentionMetadata(attn_mask=attention_mask)

        # Attention computation - returns [B, seq, num_heads, head_dim]
        hidden_states = self.attn(query, key, value, attn_metadata=attn_metadata)

        # Reshape back: [B, seq, num_heads, head_dim] → [B, seq, hidden]
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1)

        return hidden_states
```

**Key Points:**
- **Attention layer initialization:** Done in `__init__`, not per-forward
- **Tensor reshaping:** vLLM-Omni expects `[B, seq, num_heads, head_dim]` format
- **AttentionMetadata:** Wraps attention mask and other metadata
- **No need for `dispatch_attention_fn`:** vLLM-Omni selects backend automatically

**Attention backends:** vLLM-Omni automatically selects the best backend (FlashAttention, xFormers, PyTorch SDPA) based on hardware and sequence length.

#### 1.4: Replace Imports and Utilities

**Logger:**
```diff
- from diffusers.utils import logging
- logger = logging.get_logger(__name__)

+ from vllm.logger import init_logger
+ logger = init_logger(__name__)
```

**Custom layers (if needed):**
```python
# vLLM's optimized implementations
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

# vLLM-Omni's diffusion-specific layers
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
```

**Use vLLM's layers when available for:**
- Better performance (fused kernels)
- Tensor parallelism support
- Quantization compatibility

#### 1.5: Remove Training-Only Code

Remove code that's only needed for training:

```diff
# Remove gradient checkpointing
- if torch.is_grad_enabled() and self.gradient_checkpointing:
-     hidden_states = torch.utils.checkpoint.checkpoint(
-         self._forward_block, hidden_states, ...
-     )
- else:
-     hidden_states = self._forward_block(hidden_states, ...)
+ hidden_states = self._forward_block(hidden_states, ...)

# Remove training-specific attributes
- self.gradient_checkpointing = False

# Remove dropout (set to 0 or remove)
- self.dropout = nn.Dropout(dropout_prob)
+ # Removed dropout for inference
```

#### 1.6: Add Configuration Support

Add support for vLLM-Omni's `OmniDiffusionConfig`:

```python
from vllm_omni.diffusion.data import OmniDiffusionConfig

class YourModelTransformer2DModel(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,  # vLLM-Omni config
        # ... other model-specific parameters
        num_layers: int = 28,
        hidden_size: int = 3072,
        num_heads: int = 24,
        **kwargs,
    ):
        super().__init__()

        # Store config
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config if od_config else None

        # Model architecture
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # ... initialize layers
```

#### 1.7: Complete Example (Qwen-Image Transformer)

Here's a simplified example showing the key adaptations:

```python
# vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

logger = init_logger(__name__)


class QwenImageTransformerBlock(nn.Module):
    """Single transformer block for Qwen-Image (dual-stream)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Image stream (self-attention)
        self.img_norm1 = AdaLayerNorm(hidden_size, ada_dim=hidden_size)
        self.img_attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
            num_kv_heads=num_heads,
        )

        # Text-to-image cross-attention
        self.txt_norm1 = RMSNorm(hidden_size)
        self.txt_attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
            num_kv_heads=num_heads,
        )

        # MLP
        self.img_norm2 = AdaLayerNorm(hidden_size, ada_dim=hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Image features
        encoder_hidden_states: torch.Tensor,  # Text features
        temb: torch.Tensor,  # Timestep embedding
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        # Image self-attention
        norm_hidden_states = self.img_norm1(hidden_states, temb)
        attn_output = self.img_attn(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            attn_metadata=AttentionMetadata(),
        )
        hidden_states = hidden_states + attn_output

        # Text-to-image cross-attention
        norm_encoder_states = self.txt_norm1(encoder_hidden_states)
        cross_attn_output = self.txt_attn(
            hidden_states,
            norm_encoder_states,
            norm_encoder_states,
            attn_metadata=AttentionMetadata(),
        )
        encoder_hidden_states = encoder_hidden_states + cross_attn_output

        # MLP
        norm_hidden_states = self.img_norm2(hidden_states, temb)
        mlp_output = self.mlp(norm_hidden_states)
        hidden_states = hidden_states + mlp_output

        return encoder_hidden_states, hidden_states


class QwenImageTransformer2DModel(nn.Module):
    """Qwen-Image transformer with dual-stream architecture."""

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,
        num_layers: int = 28,
        hidden_size: int = 3072,
        num_heads: int = 24,
        **kwargs,
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config if od_config else None

        # Input projections
        self.img_in = nn.Linear(16, hidden_size)  # VAE latent channels
        self.txt_in = nn.Linear(3584, hidden_size)  # Text encoder dim

        # Positional embeddings
        self.pos_embed = RotaryEmbedding(dim=self.head_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            QwenImageTransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        # Output projection
        self.norm_out = AdaLayerNorm(hidden_size, ada_dim=hidden_size)
        self.proj_out = nn.Linear(hidden_size, 16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ):
        # Input projection
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Create timestep embedding
        temb = self.timestep_embedding(timestep)

        # Process through blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
            )

        # Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return (output,)
```

---

### Step 2: Adapt Pipeline

The pipeline orchestrates the full generation process (text encoding, denoising loop, VAE decoding). Adapt it from Diffusers format to vLLM-Omni's interface.

#### 2.1: Copy and Remove Diffusers Inheritance

```bash
# Copy pipeline from Diffusers
cp path/to/diffusers/src/diffusers/pipelines/your_model/pipeline_your_model.py \
   vllm_omni/diffusion/models/your_model_name/pipeline_your_model.py
```

**Remove Diffusers base classes:**
```diff
- from diffusers import DiffusionPipeline
- from diffusers.loaders import LoraLoaderMixin

- class YourModelPipeline(DiffusionPipeline, LoraLoaderMixin):
+ class YourModelPipeline(nn.Module):
    """Your model pipeline for vLLM-Omni."""
```

#### 2.2: Adapt `__init__` Method

**Before (Diffusers):**
```python
class YourModelPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        transformer: YourTransformer,
        scheduler: FlowMatchScheduler,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
```

**After (vLLM-Omni):**
```python
import os
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.models.your_model_name.your_model_transformer import (
    YourModelTransformer2DModel,
)


class YourModelPipeline(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Load components from checkpoint
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            model,
            subfolder="text_encoder",
            local_files_only=local_files_only,
        ).to(self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )

        self.vae = AutoencoderKL.from_pretrained(
            model,
            subfolder="vae",
            local_files_only=local_files_only,
        ).to(self.device)

        # Initialize transformer with vLLM-Omni config
        transformer_kwargs = get_transformer_config_kwargs(
            od_config.tf_model_config,
            YourModelTransformer2DModel,
        )
        self.transformer = YourModelTransformer2DModel(
            od_config=od_config,
            **transformer_kwargs,
        )

        # Store VAE scale factor for latent space conversions
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = 128  # Default latent size
```

**Key Changes:**
1. **`od_config` parameter:** All configuration through `OmniDiffusionConfig`
2. **Manual component loading:** No `register_modules()`, load each component explicitly
3. **Local files support:** Check `os.path.exists(model)` for local checkpoints
4. **Transformer with config:** Pass `od_config` to transformer constructor

#### 2.3: Adapt `__call__` → `forward` Method

**Change signature:**
```diff
- @torch.no_grad()
- def __call__(
+ def forward(
    self,
+   req: OmniDiffusionRequest,  # ← Add request parameter
    prompt: str | list[str] = None,
    negative_prompt: str | list[str] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    **kwargs,
- ):
+ ) -> DiffusionOutput:  # ← Add return type
```

**Extract parameters from request:**
```python
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.data import DiffusionOutput

def forward(
    self,
    req: OmniDiffusionRequest,
    prompt: str | list[str] = None,
    negative_prompt: str | list[str] = None,
    **kwargs,
) -> DiffusionOutput:
    # Extract prompts from request
    if req.prompts is not None:
        prompt = [
            p if isinstance(p, str) else (p.get("prompt") or "")
            for p in req.prompts
        ]

    # Extract sampling parameters
    sampling_params = req.sampling_params
    num_inference_steps = sampling_params.num_inference_steps or 50
    guidance_scale = sampling_params.guidance_scale or 7.5
    height = sampling_params.height or (self.default_sample_size * self.vae_scale_factor)
    width = sampling_params.width or (self.default_sample_size * self.vae_scale_factor)

    # For image editing pipelines, extract images from multi_modal_data
    if hasattr(req, 'multi_modal_data') and req.multi_modal_data:
        input_images = req.multi_modal_data.get('images', [])

    # ... rest of generation logic
```

**Wrap output:**
```diff
    # Generate images
    images = self.vae.decode(latents)[0]

-   return {"images": images}
+   return DiffusionOutput(output=images)
```

#### 2.4: Extract Pre/Post-Processing Functions

vLLM-Omni separates image processing from the main pipeline for better modularity.

**Post-processing function (required):**
```python
# vllm_omni/diffusion/models/your_model_name/pipeline_your_model.py

def get_your_model_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    Create post-processing function for your model.

    Returns a function that converts latents to images.
    """
    from diffusers.image_processor import VaeImageProcessor
    import json

    # Load VAE config to get scale factor
    model_path = od_config.model
    if not os.path.exists(model_path):
        from vllm_omni.diffusion.model_loader.utils import download_weights_from_hf_specific
        model_path = download_weights_from_hf_specific(model_path, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1)

    # Create image processor
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def post_process_func(images: torch.Tensor):
        """
        Convert tensor images to PIL images.

        Args:
            images: Tensor of shape [B, C, H, W] in range [-1, 1]

        Returns:
            List of PIL images
        """
        return image_processor.postprocess(images, output_type="pil")

    return post_process_func
```

**Pre-processing function (for image editing pipelines):**
```python
def get_your_model_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    Create pre-processing function for image editing.

    Returns a function that prepares input images.
    """
    from PIL import Image
    from diffusers.image_processor import VaeImageProcessor

    # Load VAE config
    # ... (similar to post_process_func)

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def pre_process_func(images: list[Image.Image]):
        """
        Convert PIL images to latent tensors.

        Args:
            images: List of PIL images

        Returns:
            Preprocessed image tensor
        """
        return image_processor.preprocess(images)

    return pre_process_func
```

#### 2.5: Add Weight Loading Support

Add methods for automatic weight downloading and loading:

```python
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm.model_executor.models.utils import AutoWeightsLoader

class YourModelPipeline(nn.Module):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        # ... initialization code

        # Define weight sources for automatic loading
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load model weights.

        Args:
            weights: Iterable of (param_name, param_tensor) tuples

        Returns:
            Set of loaded parameter names
        """
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
```

#### 2.6: Complete Pipeline Example

Here's a complete minimal pipeline:

```python
# vllm_omni/diffusion/models/your_model_name/pipeline_your_model.py

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.your_model_name.your_model_transformer import (
    YourModelTransformer2DModel,
)


class YourModelPipeline(nn.Module):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Load components
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.vae = AutoencoderKL.from_pretrained(
            model, subfolder="vae", local_files_only=local_files_only
        ).to(self.device)
        self.transformer = YourModelTransformer2DModel(od_config=od_config)

        self.vae_scale_factor = 8
        self.default_sample_size = 128

    def encode_prompt(self, prompt: list[str]):
        """Encode text prompts."""
        # Implement text encoding
        # ...
        return prompt_embeds

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] = None,
        **kwargs,
    ) -> DiffusionOutput:
        # Extract parameters from request
        prompt = [p if isinstance(p, str) else p.get("prompt") for p in req.prompts] or prompt
        sampling_params = req.sampling_params

        # Encode prompt
        prompt_embeds = self.encode_prompt(prompt)

        # Prepare latents
        batch_size = len(prompt)
        latents = torch.randn(
            batch_size, 4,
            sampling_params.height // self.vae_scale_factor,
            sampling_params.width // self.vae_scale_factor,
        ).to(self.device)

        # Denoising loop
        self.scheduler.set_timesteps(sampling_params.num_inference_steps)
        for t in self.scheduler.timesteps:
            # Predict noise
            noise_pred = self.transformer(
                latents,
                encoder_hidden_states=prompt_embeds,
                timestep=t,
            )[0]

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents)[0]

        # Decode latents
        images = self.vae.decode(latents)[0]

        return DiffusionOutput(output=images)


def get_your_model_post_process_func(od_config: OmniDiffusionConfig):
    """Create post-processing function."""
    from diffusers.image_processor import VaeImageProcessor
    image_processor = VaeImageProcessor(vae_scale_factor=8)
    return lambda images: image_processor.postprocess(images, output_type="pil")
```

---

### Step 3: Register Model

Register your model in `vllm_omni/diffusion/registry.py` so vLLM-Omni can discover and load it.

#### 3.1: Register Pipeline Class

```python
# vllm_omni/diffusion/registry.py

_DIFFUSION_MODELS = {
    # Format: "PipelineClassName": (module_folder, module_file, class_name)

    # Existing models
    "QwenImagePipeline": ("qwen_image", "pipeline_qwen_image", "QwenImagePipeline"),
    "FluxPipeline": ("flux", "pipeline_flux", "FluxPipeline"),

    # Add your model
    "YourModelPipeline": (
        "your_model_name",           # Module folder name
        "pipeline_your_model",       # Python file name (without .py)
        "YourModelPipeline",         # Pipeline class name
    ),
}
```

**Naming:**
- **Key:** Pipeline class name (e.g., `"YourModelPipeline"`)
- **Folder:** Directory under `vllm_omni/diffusion/models/`
- **File:** Python file name without extension
- **Class:** Exact class name in the file

#### 3.2: Register Post-Processing Function

```python
# vllm_omni/diffusion/registry.py

_DIFFUSION_POST_PROCESS_FUNCS = {
    # Format: "PipelineClassName": "function_name"

    # Existing models
    "QwenImagePipeline": "get_qwen_image_post_process_func",
    "FluxPipeline": "get_flux_post_process_func",

    # Add your model
    "YourModelPipeline": "get_your_model_post_process_func",
}
```

**Function must:**
- Be defined in the same file as pipeline (`pipeline_your_model.py`)
- Accept `od_config: OmniDiffusionConfig` as parameter
- Return a callable that converts tensors to PIL images

#### 3.3: Register Pre-Processing Function (Optional)

For image editing pipelines:

```python
# vllm_omni/diffusion/registry.py

_DIFFUSION_PRE_PROCESS_FUNCS = {
    # Format: "PipelineClassName": "function_name"

    # Existing models
    "QwenImageEditPipeline": "get_qwen_image_edit_pre_process_func",

    # Add your editing pipeline
    "YourModelEditPipeline": "get_your_model_edit_pre_process_func",
}
```

#### 3.4: Export from Module

Create/update `__init__.py` to export your classes:

```python
# vllm_omni/diffusion/models/your_model_name/__init__.py

from .pipeline_your_model import (
    YourModelPipeline,
    get_your_model_post_process_func,
)
from .your_model_transformer import YourModelTransformer2DModel

__all__ = [
    "YourModelPipeline",
    "YourModelTransformer2DModel",
    "get_your_model_post_process_func",
]
```

---

### Step 4: Add Example Script

Provide a runnable example demonstrating how to use your model.

#### 4.1: Create Example File

```python
# examples/offline_inference/text_to_image/your_model_example.py

"""
Example: Text-to-Image Generation with Your Model

This script demonstrates how to use Your Model for text-to-image generation
with vLLM-Omni.

Usage:
    python your_model_example.py --model path/to/model --prompt "a cat"
"""

import argparse
from pathlib import Path

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    parser = argparse.ArgumentParser(description="Your Model Text-to-Image Example")
    parser.add_argument(
        "--model",
        type=str,
        default="your-org/your-model-name",
        help="Model name on HuggingFace Hub or local path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful landscape with mountains and a lake",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality",
        help="Negative prompt",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height in pixels",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width in pixels",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    print(f"Loading model: {args.model}")
    omni = Omni(
        model=args.model,
        trust_remote_code=True,  # If model uses custom code
    )

    # Create sampling parameters
    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    # Generate image
    print(f"Generating image with prompt: '{args.prompt}'")
    outputs = omni.generate(
        prompts=[args.prompt],
        negative_prompts=[args.negative_prompt] if args.negative_prompt else None,
        sampling_params=sampling_params,
    )

    # Save generated image
    for i, output in enumerate(outputs):
        image = output.outputs[0]  # Get PIL image
        output_path = output_dir / f"generated_{i}.png"
        image.save(output_path)
        print(f"Saved image to: {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
```

**Key elements:**
- Command-line arguments for all major parameters
- Clear usage documentation
- Output saving with informative filenames
- Progress messages for user feedback

#### 4.2: Add README Section

Update the main README or create a model-specific README:

```markdown
## Your Model

Text-to-image generation with Your Model.

### Quick Start

```bash
python examples/offline_inference/text_to_image/your_model_example.py \
    --model your-org/your-model-name \
    --prompt "a beautiful landscape" \
    --output-dir ./outputs
```

### Parameters

- `--model`: Model identifier or local path
- `--prompt`: Text description of desired image
- `--num-inference-steps`: Number of denoising steps (default: 50)
- `--guidance-scale`: CFG scale (default: 7.5)
- `--height`, `--width`: Image dimensions (default: 512x512)

### Examples

**High quality:**
```bash
python your_model_example.py --num-inference-steps 100 --guidance-scale 9.0
```

**Fast generation:**
```bash
python your_model_example.py --num-inference-steps 20 --guidance-scale 5.0
```
```

---

### Step 5: Test Your Implementation

Before submitting, thoroughly test your implementation.

#### 5.1: Basic Functionality Test

```python
# tests/e2e/offline_inference/test_your_model.py

import pytest
import torch
from PIL import Image

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


@pytest.mark.parametrize("model", ["your-org/your-model-name"])
def test_your_model_text_to_image(model):
    """Test basic text-to-image generation."""
    omni = Omni(model=model)

    outputs = omni.generate(
        prompts=["a cat"],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=20,
            height=512,
            width=512,
            seed=42,
        ),
    )

    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
    image = outputs[0].outputs[0]
    assert isinstance(image, Image.Image)
    assert image.size == (512, 512)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_your_model_batching(batch_size):
    """Test batched generation."""
    omni = Omni(model="your-org/your-model-name")

    prompts = [f"image {i}" for i in range(batch_size)]
    outputs = omni.generate(
        prompts=prompts,
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=20,
        ),
    )

    assert len(outputs) == batch_size
    for output in outputs:
        assert isinstance(output.outputs[0], Image.Image)


def test_your_model_guidance_scale():
    """Test different guidance scales."""
    omni = Omni(model="your-org/your-model-name")

    for guidance_scale in [1.0, 5.0, 10.0]:
        outputs = omni.generate(
            prompts=["a cat"],
            sampling_params=OmniDiffusionSamplingParams(
                num_inference_steps=20,
                guidance_scale=guidance_scale,
                seed=42,
            ),
        )
        assert len(outputs) == 1
```

#### 5.2: Visual Quality Check

Generate test images and manually verify quality:

```python
# scripts/test_visual_quality.py

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="your-org/your-model-name")

test_prompts = [
    "a photorealistic portrait of a person",
    "an abstract painting with vibrant colors",
    "a futuristic cityscape at night",
    "a cute puppy playing in grass",
]

for i, prompt in enumerate(test_prompts):
    outputs = omni.generate(
        prompts=[prompt],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=42,
        ),
    )
    outputs[0].outputs[0].save(f"test_output_{i}.png")
    print(f"Generated: {prompt}")
```

Compare outputs with Diffusers baseline to ensure correctness.

#### 5.3: Performance Benchmark

Compare inference speed with Diffusers:

```python
# scripts/benchmark_your_model.py

import time
from diffusers import DiffusionPipeline
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

model_name = "your-org/your-model-name"

# Benchmark Diffusers
pipe_diffusers = DiffusionPipeline.from_pretrained(model_name).to("cuda")
start = time.time()
_ = pipe_diffusers("a cat", num_inference_steps=50)
time_diffusers = time.time() - start

# Benchmark vLLM-Omni
omni = Omni(model=model_name)
start = time.time()
_ = omni.generate(
    prompts=["a cat"],
    sampling_params=OmniDiffusionSamplingParams(num_inference_steps=50),
)
time_vllm_omni = time.time() - start

print(f"Diffusers: {time_diffusers:.2f}s")
print(f"vLLM-Omni: {time_vllm_omni:.2f}s")
print(f"Speedup: {time_diffusers / time_vllm_omni:.2f}x")
```

---

## Advanced Features

Once basic implementation works, add advanced features for better performance.

### torch.compile Support

Enable automatic compilation for repeated blocks:

```python
# In your_model_transformer.py

class YourModelTransformer2DModel(nn.Module):
    # Specify which blocks can be compiled
    _repeated_blocks = ["YourTransformerBlock"]  # List of block class names

    def __init__(self, ...):
        super().__init__()
        # ... initialization
```

vLLM-Omni automatically compiles blocks in `_repeated_blocks` when `torch.compile` is available.

### Tensor Parallelism

Use vLLM's parallel linear layers:

```python
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,  # Split output dimension
    RowParallelLinear,     # Split input dimension
    QKVParallelLinear,     # Split Q, K, V heads
)

class YourAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ...):
        super().__init__()

        # Replace nn.Linear with parallel versions
        self.to_qkv = QKVParallelLinear(
            hidden_size,
            3 * hidden_size,  # Q, K, V concatenated
            bias=True,
        )

        self.to_out = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
        )
```

**Usage:** Set `tensor_parallel_size` when initializing:
```python
omni = Omni(model="your-model", tensor_parallel_size=2)
```

### CFG Parallelism

See detailed guide: [How to add CFG-Parallel support](../features/cfg_parallel.md)

**Quick setup:**
1. Create a CFG mixin inheriting from `CFGParallelMixin`
2. Implement `diffuse()` method
3. Inherit mixin in your pipeline class

### Sequence Parallelism

See detailed guide: [How to add Sequence Parallel support](../features/sequence_parallel.md)

**Quick setup:**
1. Add `_sp_plan` class attribute to transformer
2. Specify where to shard/gather tensors
3. Extract inline operations into submodules

### Cache Acceleration

#### TeaCache

See detailed guide: [How to add TeaCache support](../features/teacache.md)

**Quick setup:**
1. Write extractor function
2. Register in `EXTRACTOR_REGISTRY`
3. Add polynomial coefficients

#### Cache-DiT

See detailed guide: [How to add Cache-DiT support](../features/cache_dit.md)

**Quick setup:**
- For standard models: Works automatically!
- For complex architectures: Write custom enabler

---

## Troubleshooting

### Common Issues

#### Issue: ImportError when loading model

**Symptoms:** `ModuleNotFoundError` or `ImportError` when calling `Omni(model="your-model")`

**Causes:**
1. Model not registered in `registry.py`
2. Wrong class name in registry
3. Missing `__init__.py` exports

**Solutions:**
```python
# 1. Check registry.py
_DIFFUSION_MODELS = {
    "YourModelPipeline": ("your_model_name", "pipeline_your_model", "YourModelPipeline"),
    #                      ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^
    #                      Folder name        File name             Class name
}

# 2. Check __init__.py exports
# vllm_omni/diffusion/models/your_model_name/__init__.py
from .pipeline_your_model import YourModelPipeline

__all__ = ["YourModelPipeline"]
```

#### Issue: Shape mismatch in attention

**Symptoms:** `RuntimeError: shape mismatch` in attention forward

**Cause:** Incorrect tensor reshaping for vLLM-Omni's attention interface

**Solution:** Ensure correct shapes:
```python
# vLLM-Omni expects: [batch, seq_len, num_heads, head_dim]
query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
key = key.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim)
value = value.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim)

hidden_states = self.attn(query, key, value, attn_metadata=attn_metadata)

# Reshape back: [batch, seq_len, num_heads, head_dim] → [batch, seq_len, hidden_size]
hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
```

#### Issue: Different outputs compared to Diffusers

**Symptoms:** Generated images look different from Diffusers

**Causes:**
1. Attention backend differences (FlashAttention vs PyTorch)
2. Missing normalization or scaling
3. Incorrect scheduler configuration

**Solutions:**
```python
# 1. Use same random seed
torch.manual_seed(42)

# 2. Check attention scaling
self.attn = Attention(
    softmax_scale=1.0 / (self.head_dim ** 0.5),  # ← Important!
)

# 3. Verify scheduler matches Diffusers
self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(...)
```

#### Issue: Out of memory (OOM)

**Symptoms:** CUDA out of memory errors

**Solutions:**
1. **Reduce batch size:**
   ```python
   omni.generate(prompts=[...], max_batch_size=2)
   ```

2. **Use smaller image size:**
   ```python
   sampling_params = OmniDiffusionSamplingParams(height=512, width=512)
   ```

3. **Enable model offloading:**
   ```python
   omni = Omni(model="...", device_map="auto")
   ```

#### Issue: Slow inference

**Symptoms:** Generation slower than expected

**Solutions:**
1. **Enable torch.compile:**
   ```python
   # In transformer: _repeated_blocks = ["BlockClassName"]
   ```

2. **Use FlashAttention:**
   ```python
   # Automatically used if available, or install:
   # pip install flash-attn
   ```

3. **Enable cache acceleration:**
   ```python
   omni = Omni(
       model="...",
       cache_backend="cache_dit",  # or "tea_cache"
   )
   ```

---

## Pull Request Checklist

When submitting a PR to add your model, include:

### 1. Implementation Files

- ✅ Transformer model (`xxx_transformer.py`)
- ✅ Pipeline (`pipeline_xxx.py`)
- ✅ Registry entries in `registry.py`
- ✅ `__init__.py` with proper exports

### 2. Example and Tests

- ✅ Example script in `examples/`
- ✅ Test file in `tests/e2e/`
- ✅ README documentation

### 3. Verification Results

Include in PR description:

**Output Verification:**
```
Prompt: "a cat sitting on a windowsill"
Diffusers output: [attach image]
vLLM-Omni output: [attach image]
Visual similarity: ✅ Identical / ⚠️ Minor differences / ❌ Different
```

**Performance Benchmark:**
```
Hardware: A100 40GB
Batch size: 1
Inference steps: 50
Image size: 512x512

Diffusers: 3.2s
vLLM-Omni: 2.1s
Speedup: 1.52x
```

**Parallelism Support:**
```
✅ Tensor Parallel: Tested with tp_size=[1, 2, 4]
✅ CFG Parallel: Tested with cfg_parallel_size=2
✅ Sequence Parallel: Tested with sp_size=[1, 2]
⚠️ Patch VAE Parallel: Not yet supported
```

**Cache Acceleration:**
```
✅ TeaCache: Supported, 1.8x speedup at rel_l1_thresh=0.2
✅ Cache-DiT: Supported, 2.3x speedup with DBCache
```

### 4. Documentation

- ✅ Model-specific documentation (if needed)
- ✅ Usage examples
- ✅ Known limitations

### 5. Code Quality

- ✅ Passes linting (`ruff check`)
- ✅ Passes type checking (`mypy`)
- ✅ All tests pass
- ✅ No degradation in existing tests

**Run checks:**
```bash
# Linting
ruff check vllm_omni/diffusion/models/your_model_name/

# Type checking
mypy vllm_omni/diffusion/models/your_model_name/

# Tests
pytest tests/e2e/offline_inference/test_your_model.py -v
```

### 6. Model Recipe (Optional but Recommended)

Add a model recipe to [vllm-project/recipes](https://github.com/vllm-project/recipes) showing:
- Installation instructions
- Basic usage
- Advanced features
- Performance tips

---

## Reference Implementations

Study these complete examples:

| Model | Architecture | Key Features | Files |
|-------|--------------|--------------|-------|
| **Qwen-Image** | Dual-stream transformer | CFG-Parallel, SP, TeaCache | `vllm_omni/diffusion/models/qwen_image/` |
| **LongCat-Image** | FLUX-like MMDiT | Dual block lists, SP, Cache-DiT | `vllm_omni/diffusion/models/longcat_image/` |
| **Wan2.2** | Video transformer | Dual transformers, SP, temporal attention | `vllm_omni/diffusion/models/wan2_2/` |
| **Flux** | Image transformer | Single-stream, SP | `vllm_omni/diffusion/models/flux/` |
| **Z-Image** | Unified sequence | Concatenated input, SP | `vllm_omni/diffusion/models/z_image/` |

**Code locations:**
- **Transformers:** `vllm_omni/diffusion/models/{model_name}/{model_name}_transformer.py`
- **Pipelines:** `vllm_omni/diffusion/models/{model_name}/pipeline_{model_name}.py`
- **CFG Mixins:** `vllm_omni/diffusion/models/{model_name}/cfg_parallel.py`
- **Examples:** `examples/offline_inference/{task}/{model_name}_example.py`
- **Tests:** `tests/e2e/offline_inference/test_{model_name}.py`

---

## Summary

Adding a diffusion model to vLLM-Omni involves:

1. ✅ **Adapt transformer** - Replace attention, remove mixins, add config support
2. ✅ **Adapt pipeline** - Change interface, add request handling, extract processing
3. ✅ **Register model** - Add entries to `registry.py`
4. ✅ **Add examples** - Provide runnable scripts
5. ✅ **Test thoroughly** - Verify correctness and performance
6. ✅ **Add advanced features** - Enable parallelism and acceleration (optional)
7. ✅ **Submit PR** - Include verification results and documentation

**Need help?** Check reference implementations or ask in [slack.vllm.ai](https://slack.vllm.ai) or vLLM user forum at [discuss.vllm.ai](https://discuss.vllm.ai).
