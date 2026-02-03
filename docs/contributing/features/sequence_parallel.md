# How to add Sequence Parallel support for a new model

This section describes how to add Sequence Parallel (SP) to a diffusion **transformer model**. We use the Qwen-Image transformer (`vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py`) and Wan2.2 transformer as reference implementations.

Sequence Parallel distributes long sequences across multiple GPUs, enabling generation of high-resolution images and videos that wouldn't fit in a single GPU's memory. It provides **near-linear scaling** for sequence length, allowing models to process sequences **N×** longer with **N** GPUs.

---

## Overview

### What is Sequence Parallel?

**Terminology Note:** Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP) in the [diffusers library](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/_modeling_parallel.py). We use "Sequence Parallelism" to align with vLLM-Omni's terminology.

Diffusion transformers process long sequences of image patches or video frames. For high-resolution generation, these sequences can become very large:

- **1024×1024 image** (patch size 2): ~262K tokens
- **512×512×64 frame video** (patch size 2×2×1): ~4M tokens

**Problem:** Long sequences exceed single GPU memory capacity and slow down generation.

**Solution:** Sequence Parallel (SP) shards the sequence dimension across multiple GPUs:

```
┌─────────────────────────────────────────┐
│  Full Sequence (N tokens)               │
└─────────────────────────────────────────┘
              ↓ SP sharding
┌────────────┬────────────┬────────────────┐
│ GPU 0      │ GPU 1      │ GPU 2          │
│ tokens     │ tokens     │ tokens         │
│ [0:N/3]    │ [N/3:2N/3] │ [2N/3:N]       │
└────────────┴────────────┴────────────────┘
```

Each GPU processes only a portion of the sequence, with attention mechanisms (Ulysses/Ring) handling cross-GPU communication transparently.

### Key Benefits

- ✅ **Enables high-resolution generation** - 4K+ images, long videos
- ✅ **Near-linear memory scaling** - 2× GPUs = 2× max sequence length
- ✅ **Minimal communication overhead** - efficient attention algorithms
- ✅ **Zero accuracy loss** - mathematically identical to non-parallel
- ✅ **Non-intrusive** - use `_sp_plan` declaration, no forward() changes

### SP vs. Other Parallelism

| Type | What's sharded | Use case | Compatible with SP? |
|------|----------------|----------|-------------------|
| **Sequence Parallel (SP)** | Sequence dimension | Long sequences (high-res images/videos) | ✅ |
| **Tensor Parallel (TP)** | Model weights | Large models | ✅ Yes |
| **CFG Parallel** | Positive/negative branches | Speed up CFG | ✅ Yes |
| **Pipeline Parallel (PP)** | Model layers | Very large models | ⚠️ Complex |

---

## Architecture: Two Approaches

### Approach 1: Non-Intrusive `_sp_plan` (Recommended)

The `_sp_plan` mechanism allows SP **without modifying `forward()` logic**. The framework automatically registers hooks to shard inputs and gather outputs at module boundaries.

**How it works:**
1. Declare `_sp_plan` dict in your transformer class
2. Framework automatically applies hooks when `sequence_parallel_size > 1`
3. Hooks shard/gather tensors at specified module boundaries
4. Attention layers handle cross-GPU communication internally

**Requirements:**
- Tensor operations that need sharding/gathering must happen at **`nn.Module` boundaries**
- Inline Python operations (e.g., `torch.cat`, `pad_sequence`) **cannot be hooked**

**Solution for inline operations:** Extract into a submodule.

### Approach 2: Intrusive Modification (For Complex Cases)

For models with dynamic sharding logic that cannot be expressed via `_sp_plan`, manually insert shard/gather calls:

```python
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard, sp_gather

def forward(self, hidden_states, ...):
    if self.parallel_config.sequence_parallel_size > 1:
        hidden_states = sp_shard(hidden_states, dim=1)

    # ... computation ...

    if self.parallel_config.sequence_parallel_size > 1:
        output = sp_gather(output, dim=1)

    return output
```

**When to use:**
- Dynamic/conditional sharding logic
- Complex tensor manipulations that can't be encapsulated
- Temporary workaround during development

---

## Step-by-Step: Implementing `_sp_plan`

### Step 1: Understand Module Boundaries

Identify where tensors need to be sharded or gathered in your model's forward pass:

```python
class MyTransformer(nn.Module):
    def __init__(self):
        self.patch_embed = PatchEmbed()      # ← Boundary 1
        self.pos_embed = RoPE()              # ← Boundary 2
        self.blocks = nn.ModuleList([...])   # ← Boundary 3
        self.norm_out = LayerNorm()
        self.proj_out = Linear()             # ← Boundary 4

    def forward(self, x):
        x = self.patch_embed(x)              # ← Shard after this?
        pos = self.pos_embed(x)              # ← Shard RoPE outputs?
        for block in self.blocks:
            x = block(x, pos)                # ← Blocks process sharded x
        x = self.norm_out(x)
        output = self.proj_out(x)            # ← Gather before this?
        return output
```

**Key question:** At which boundaries should tensors be sharded or gathered?

### Step 2: Handle Inline Operations

If your `forward()` contains inline tensor operations, **extract them into submodules**.

**Example: Z-Image concatenates image + text features inline**

```python
# ❌ BAD: Inline operation - hooks cannot intercept
class ZImageTransformer(nn.Module):
    def forward(self, x, cap_feats):
        # This concatenation happens inline - _sp_plan can't shard it!
        unified = torch.cat([x, cap_feats], dim=1)

        for layer in self.layers:
            unified = layer(unified)

        return unified

# ✅ GOOD: Extract into submodule
class UnifiedPrepare(nn.Module):
    """Submodule to concatenate image and text features."""
    def forward(self, x, cap_feats):
        return torch.cat([x, cap_feats], dim=1)

class ZImageTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.unified_prepare = UnifiedPrepare()  # Now a module!
        self.layers = nn.ModuleList([...])

    def forward(self, x, cap_feats):
        # Now _sp_plan can shard the output of unified_prepare!
        unified = self.unified_prepare(x, cap_feats)

        for layer in self.layers:
            unified = layer(unified)

        return unified
```

**Other common cases:**
- `pad_sequence()` → `PadSequenceModule`
- `torch.cat()` → `ConcatModule`
- `tensor.reshape()` → `ReshapeModule`
- Complex preprocessing → `PreprocessModule`

### Step 3: Define `_sp_plan`

Create a class-level `_sp_plan` dictionary specifying where to shard/gather tensors.

**Type definitions:**

```python
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,   # For sharding (splitting) tensors
    SequenceParallelOutput,  # For gathering tensors
)
```

**SequenceParallelInput parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `split_dim` | int | Dimension to split (usually `1` for sequence) |
| `expected_dims` | int \| None | Expected tensor rank for validation (optional) |
| `split_output` | bool | `False`: shard **input** params; `True`: shard **output** tensors |
| `auto_pad` | bool | Auto-pad if sequence not divisible by world_size (default: False) |

**SequenceParallelOutput parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `gather_dim` | int | Dimension to gather (usually `1` for sequence) |
| `expected_dims` | int \| None | Expected tensor rank for validation (optional) |

**Module naming conventions:**

| Key | Meaning | Python equivalent |
|-----|---------|-------------------|
| `""` | Root model | `model` |
| `"blocks.0"` | First element of ModuleList | `model.blocks[0]` |
| `"blocks.*"` | All elements of ModuleList | `for b in model.blocks` |
| `"rope"` | Named submodule | `model.rope` |
| `"outputs.main"` | ModuleDict entry | `model.outputs["main"]` |

**Dictionary value types:**

| Key type | `split_output` | Description |
|----------|----------------|-------------|
| `"param_name"` (str) | `False` | Shard **input parameter** by name |
| `0`, `1`, ... (int) | `True` | Shard **output tuple** by index |

### Step 4: Write `_sp_plan` for Your Model

**Pattern 1: Shard at first block, gather at output projection**

Most common pattern for standard transformers:

```python
class StandardTransformer(nn.Module):
    _sp_plan = {
        # Shard hidden_states at first transformer block input
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        # Gather at final output projection
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

**Pattern 2: Shard RoPE embeddings separately**

When RoPE is computed in a separate module:

```python
class TransformerWithRoPE(nn.Module):
    _sp_plan = {
        # Shard RoPE module OUTPUTS (returns tuple of cos, sin)
        "rope": {
            0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # cos
            1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # sin
        },
        # Shard transformer block INPUT
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        # Gather at output
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

**Pattern 3: Variable sequence length support (auto_pad)**

For models that support variable sequence lengths:

```python
class VariableLengthTransformer(nn.Module):
    _sp_plan = {
        "blocks.0": {
            "hidden_states": SequenceParallelInput(
                split_dim=1,
                expected_dims=3,
                auto_pad=True,  # Auto-pad sequences not divisible by world_size
            ),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

---

## Complete Examples

### Example 1: Wan2.2 Transformer (Standard Pattern)

**Architecture:** Single-stream transformer with RoPE

```python
class Wan22Transformer2DModel(nn.Module):
    """
    Wan2.2 video generation transformer.

    Forward flow:
    1. patch_embedding: [B, C, F, H, W] → [B, seq, dim]
    2. rope: Compute RoPE embeddings [1, seq, 1, dim]
    3. blocks: Process with RoPE attention
    4. proj_out: Final projection
    5. unpatchify: [B, seq, dim] → [B, C, F, H, W]
    """

    _sp_plan = {
        # Shard RoPE embeddings after rope module computes them
        "rope": {
            0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # cos
            1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # sin
        },
        # Shard hidden_states at first transformer block input
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        # Gather at proj_out (before unpatchify)
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(self, ...):
        super().__init__()
        self.patch_embedding = PatchEmbed(...)
        self.rope = RotaryEmbedding(...)
        self.blocks = nn.ModuleList([Wan22TransformerBlock(...) for _ in range(num_layers)])
        self.proj_out = nn.Linear(...)

    def forward(self, hidden_states, ...):
        # Patchify: [B, C, F, H, W] → [B, seq, dim]
        hidden_states = self.patch_embedding(hidden_states)

        # Compute RoPE embeddings [1, seq, 1, dim]
        # _sp_plan shards these outputs via rope.{0,1}
        freqs_cos, freqs_sin = self.rope(hidden_states)

        # Process through transformer blocks
        # hidden_states is sharded at blocks.0 input
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                freqs_cos=freqs_cos,  # Sharded
                freqs_sin=freqs_sin,  # Sharded
            )

        # Final projection
        # Output is gathered at proj_out
        output = self.proj_out(hidden_states)

        # Unpatchify: [B, seq, dim] → [B, C, F, H, W]
        output = self.unpatchify(output, ...)

        return output
```

### Example 2: Qwen-Image Transformer (Dual-Stream with Preprocessing)

**Architecture:** Dual-stream (image + text) with RoPE and variable-length support

```python
class QwenImageTransformer2DModel(nn.Module):
    """
    Qwen-Image dual-stream transformer.

    Forward flow:
    1. image_rope_prepare: Project image + compute RoPE → (hidden_states, vid_freqs, txt_freqs)
    2. modulate_index_prepare: Create modulate_index for editing (optional)
    3. blocks: Dual-stream attention (image ↔ text)
    4. proj_out: Final projection
    """

    _sp_plan = {
        # Shard ImageRopePrepare outputs
        # (hidden_states and vid_freqs must be sharded together for RoPE)
        "image_rope_prepare": {
            # hidden_states: [B, img_seq, dim]
            0: SequenceParallelInput(
                split_dim=1,
                expected_dims=3,
                split_output=True,
                auto_pad=True,  # Support variable image sizes
            ),
            # vid_freqs: [img_seq, rope_dim] (image RoPE)
            1: SequenceParallelInput(
                split_dim=0,
                expected_dims=2,
                split_output=True,
                auto_pad=True,
            ),
            # txt_freqs (index 2) NOT sharded - kept replicated for dual-stream
        },
        # Shard ModulateIndexPrepare output (for editing models)
        "modulate_index_prepare": {
            # modulate_index: [B, seq_len] - must match hidden_states sharding
            1: SequenceParallelInput(
                split_dim=1,
                expected_dims=2,
                split_output=True,
                auto_pad=True,
            ),
        },
        # Gather at final projection
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(self, ...):
        super().__init__()

        # Create preprocessing submodules for _sp_plan
        self.image_rope_prepare = ImageRopePrepare(...)
        self.modulate_index_prepare = ModulateIndexPrepare(...)

        self.transformer_blocks = nn.ModuleList([
            QwenImageTransformerBlock(...) for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(...)

    def forward(
        self,
        hidden_states,           # Image latents
        encoder_hidden_states,   # Text embeddings
        timestep,
        img_shapes,
        txt_seq_lens,
        ...
    ):
        # Prepare image inputs + compute RoPE
        # _sp_plan shards outputs 0 (hidden_states) and 1 (vid_freqs)
        hidden_states, vid_freqs, txt_freqs = self.image_rope_prepare(
            hidden_states, img_shapes, txt_seq_lens
        )

        # Prepare modulate index (for editing models)
        timestep, modulate_index = self.modulate_index_prepare(
            timestep, img_shapes
        )

        # Process through dual-stream blocks
        # hidden_states is sharded, encoder_hidden_states is replicated
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,        # Sharded (image)
                encoder_hidden_states=encoder_hidden_states,  # Replicated (text)
                temb=timestep,
                image_rotary_emb=(vid_freqs, txt_freqs),  # vid sharded, txt replicated
                modulate_index=modulate_index,      # Sharded (if exists)
            )

        # Final projection (output gathered here)
        output = self.proj_out(hidden_states)

        return output
```

### Example 3: Z-Image Transformer (Unified Sequence)

**Architecture:** Concatenates image + text into unified sequence

```python
class ZImageTransformer2DModel(nn.Module):
    """
    Z-Image unified sequence transformer.

    Forward flow:
    1. noise_refiner: Refine image patches
    2. context_refiner: Refine text features
    3. unified_prepare: Concatenate image + text → unified sequence
    4. layers: Process unified sequence
    5. final_layer: Final projection
    """

    _sp_plan = {
        # Shard unified_prepare outputs
        # All 4 outputs must be sharded together
        "unified_prepare": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # unified
            1: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # unified_cos
            2: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # unified_sin
            3: SequenceParallelInput(split_dim=1, expected_dims=2, split_output=True),  # unified_attn_mask
        },
        # Gather at final layer (default patch_size=2, f_patch_size=1)
        "all_final_layer.2-1": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(self, ...):
        super().__init__()

        # Preprocessing modules
        self.noise_refiner = nn.ModuleList([...])
        self.context_refiner = nn.ModuleList([...])

        # Create submodule for concatenation (needed for _sp_plan)
        self.unified_prepare = UnifiedPrepare(...)

        # Main transformer layers
        self.layers = nn.ModuleList([ZImageBlock(...) for _ in range(n_layers)])

        # Final layers (ModuleDict for different patch sizes)
        self.all_final_layer = nn.ModuleDict({...})

    def forward(self, x, cap_feats, ...):
        # Refine image patches
        x_refined = x
        for layer in self.noise_refiner:
            x_refined = layer(x_refined)

        # Refine text features
        cap_refined = cap_feats
        for layer in self.context_refiner:
            cap_refined = layer(cap_refined)

        # Concatenate image + text into unified sequence
        # _sp_plan shards all 4 outputs of unified_prepare
        unified, unified_cos, unified_sin, unified_attn_mask = self.unified_prepare(
            x_refined, cap_refined
        )

        # Process unified sequence
        # All inputs are sharded along sequence dimension
        for layer in self.layers:
            unified = layer(
                unified,
                unified_attn_mask,
                unified_cos,
                unified_sin,
            )

        # Final projection (output gathered here)
        output = self.all_final_layer["2-1"](unified)

        return output


class UnifiedPrepare(nn.Module):
    """Submodule to prepare unified sequence (needed for _sp_plan)."""

    def forward(self, x_batched, cap_batched):
        # Concatenate image and caption sequences
        unified_list = []
        unified_cos_list = []
        unified_sin_list = []

        for i in range(batch_size):
            unified_list.append(
                torch.cat([x_batched[i], cap_batched[i]], dim=0)
            )
            # ... compute RoPE for unified sequence ...

        # Pad sequences
        unified = pad_sequence(unified_list, batch_first=True)
        unified_cos = pad_sequence(unified_cos_list, batch_first=True)
        unified_sin = pad_sequence(unified_sin_list, batch_first=True)

        # Create attention mask
        unified_attn_mask = create_attention_mask(...)

        return unified, unified_cos, unified_sin, unified_attn_mask
```

---

## Hook Flow and Execution

### How Hooks Work

When `sequence_parallel_size > 1`, the framework automatically registers hooks based on `_sp_plan`:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Forward Pass                             │
└─────────────────────────────────────────────────────────────────┘

Input (full sequence)
  │
  ├─ Pre-forward hook → Split into shards
  │
  ▼
[Module.forward()] ← Each GPU processes shard
  │
  ├─ Post-forward hook → Gather from all GPUs (if SequenceParallelOutput)
  │
  ▼
Output (full sequence or sharded)
```

### Hook Types

**SequenceParallelSplitHook:**
- Triggered: Before or after module forward (depending on `split_output`)
- Action: Shard tensors along `split_dim` across all SP ranks
- Communication: No communication (pure splitting)

**SequenceParallelGatherHook:**
- Triggered: After module forward
- Action: Gather sharded tensors from all SP ranks
- Communication: `all_gather` operation

### Attention Communication

Attention layers handle cross-GPU communication internally using:

**Ulysses Attention:** All-to-all communication for global attention
**Ring Attention:** Peer-to-peer communication for local attention

These are automatically used when `sequence_parallel_size > 1` - no code changes needed!

---

## Common Patterns

### Pattern 1: Standard Single-Stream Transformer

```python
_sp_plan = {
    "blocks.0": {
        "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
    },
    "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
}
```

**Use case:** Most image/video transformers (FLUX, Wan2.2, SD3)

### Pattern 2: Transformer with Separate RoPE Module

```python
_sp_plan = {
    "rope": {
        0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
        1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
    },
    "blocks.0": {
        "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
    },
    "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
}
```

**Use case:** Models with separate positional embedding computation

### Pattern 3: Dual-Stream Transformer (Image + Text)

```python
_sp_plan = {
    "image_rope_prepare": {
        0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # image
        1: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),  # image RoPE
        # text (index 2) NOT sharded - kept replicated
    },
    "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
}
```

**Use case:** Models with separate image and text streams (Qwen-Image, SD3)

### Pattern 4: Unified Sequence (Image + Text Concatenated)

```python
_sp_plan = {
    "unified_prepare": {
        0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # unified sequence
        1: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # unified RoPE cos
        2: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # unified RoPE sin
        3: SequenceParallelInput(split_dim=1, expected_dims=2, split_output=True),  # unified mask
    },
    "final_layer": SequenceParallelOutput(gather_dim=1, expected_dims=3),
}
```

**Use case:** Models that concatenate all inputs into one sequence (Z-Image)

### Pattern 5: Variable-Length Support (auto_pad)

```python
_sp_plan = {
    "blocks.0": {
        "hidden_states": SequenceParallelInput(
            split_dim=1,
            expected_dims=3,
            auto_pad=True,  # Automatically pad sequences
        ),
    },
    "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
}
```

**Use case:** Models that support variable image sizes or video lengths

---

## Troubleshooting

### Issue: SP not activating

**Symptoms:** Generation still using full sequence on single GPU, no memory savings.

**Causes & Solutions:**

1. **SP world size not set:**
   ```bash
   # Check SP configuration
   python -c "from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size; print(get_sequence_parallel_world_size())"

   # Should print N for N-way SP, 1 for no SP
   ```

   **Solution:** Initialize with `sequence_parallel_size=N`:
   ```python
   from vllm_omni.diffusion.distributed import initialize_model_parallel
   initialize_model_parallel(sequence_parallel_size=2)
   ```

2. **`_sp_plan` not defined:**

   **Solution:** Add `_sp_plan` class attribute to your transformer.

3. **Hooks not applied:**

   **Solution:** Verify hooks are registered:
   ```python
   # After model initialization
   print(model._sp_hooks_applied)  # Should be True
   ```

### Issue: Shape mismatch errors

**Symptoms:** `RuntimeError: shape mismatch` during forward pass.

**Causes & Solutions:**

1. **RoPE dimension mismatch:**

   **Problem:** RoPE embeddings not sharded, but hidden_states is sharded.

   **Solution:** Shard RoPE outputs in `_sp_plan`:
   ```python
   _sp_plan = {
       "rope": {
           0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
           1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
       },
       ...
   }
   ```

2. **Attention mask dimension mismatch:**

   **Problem:** Attention mask not sharded, but hidden_states is sharded.

   **Solution:** Shard mask in `_sp_plan`:
   ```python
   "unified_prepare": {
       0: SequenceParallelInput(...),  # sequence
       1: SequenceParallelInput(...),  # mask
   }
   ```

### Issue: Sequence length not divisible by world_size

**Symptoms:** `ValueError: Sequence length X not divisible by world_size Y`.

**Cause:** Sequence length cannot be evenly split across GPUs.

**Solution:** Use `auto_pad=True`:
```python
_sp_plan = {
    "blocks.0": {
        "hidden_states": SequenceParallelInput(
            split_dim=1,
            auto_pad=True,  # Automatically pad to make divisible
        ),
    },
}
```

### Issue: Inline operations not sharded

**Symptoms:** Some tensors remain full-sized, not sharded.

**Cause:** Operations happen inline in `forward()`, not at module boundaries.

**Example Problem:**
```python
def forward(self, x, cap):
    unified = torch.cat([x, cap], dim=1)  # ← Inline operation!
    # _sp_plan can't hook this
```

**Solution:** Extract into submodule:
```python
class ConcatModule(nn.Module):
    def forward(self, x, cap):
        return torch.cat([x, cap], dim=1)

class MyModel(nn.Module):
    def __init__(self):
        self.concat = ConcatModule()  # Now hookable!

    def forward(self, x, cap):
        unified = self.concat(x, cap)  # ← Can be sharded via _sp_plan
```

### Issue: Quality degradation with SP

**Symptoms:** Generated images/videos look different with SP enabled.

**Cause:** Attention implementation not handling sharding correctly.

**Solution:**
1. Verify attention uses Ulysses/Ring communication
2. Check attention mask is correctly sharded
3. Ensure LayerNorm uses correct statistics (not per-shard)

---

## Reference Implementations

Complete examples in the codebase:

| Model | Path | Pattern | Notes |
|-------|------|---------|-------|
| **Qwen-Image** | `vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py` | Dual-stream + preprocessing | auto_pad, separate RoPE |
| **Wan2.2** | `vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py` | Standard + RoPE | Video transformer |
| **Z-Image** | `vllm_omni/diffusion/models/z_image/z_image_transformer.py` | Unified sequence | Concatenated input |
| **SP Plan Types** | `vllm_omni/diffusion/distributed/sp_plan.py` | Type definitions | SequenceParallelInput/Output |
| **Hook Implementation** | `vllm_omni/diffusion/hooks/sequence_parallel.py` | Hook mechanics | How hooks work |
| **Tests** | `tests/diffusion/distributed/test_sp_plan_hooks.py` | Test examples | Validation patterns |

---

## Summary

Adding Sequence Parallel support to a transformer:

1. ✅ **Identify sharding boundaries** - Where should tensors be split/gathered?
2. ✅ **Extract inline operations** - Move `torch.cat`, `pad_sequence`, etc. to submodules
3. ✅ **Define `_sp_plan`** - Declare shard/gather points as class attribute
4. ✅ **Use `auto_pad` for variable lengths** - Support non-uniform sequences
5. ✅ **Shard RoPE embeddings together** - Keep hidden_states and RoPE dimensions aligned
6. ✅ **Test with different world_sizes** - Verify correctness and performance

**Key principle:** Tensors must be sharded at **`nn.Module` boundaries** - the framework handles everything else automatically!
