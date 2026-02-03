
# How to parallelize a new model for SP

NOTE: "Terminology: SP vs CP"
    Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP) in the [diffusers library](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/_modeling_parallel.py).
    We use "Sequence Parallelism" to align with vLLM-Omni's terminology.

---

## Non-intrusive `_sp_plan` (Recommended)

The `_sp_plan` mechanism allows SP without modifying `forward()` logic. The framework automatically registers hooks to shard inputs and gather outputs at module boundaries.

**Requirements for `forward()` function:**

- Tensor operations that need sharding/gathering must happen at **`nn.Module` boundaries** (not inline Python operations)
- If your `forward()` contains inline tensor operations (e.g., `torch.cat`, `pad_sequence`) that need sharding, **extract them into a submodule**

**When to create a submodule:**

```python
# ❌ BAD: Inline operations - hooks cannot intercept
def forward(self, x, cap_feats):
    unified = torch.cat([x, cap_feats], dim=1)  # Cannot be sharded via _sp_plan
    ...

# ✅ GOOD: Extract into a submodule
class UnifiedPrepare(nn.Module):
    def forward(self, x, cap_feats):
        return torch.cat([x, cap_feats], dim=1)  # Now can be sharded via _sp_plan

class MyModel(nn.Module):
    def __init__(self):
        self.unified_prepare = UnifiedPrepare()  # Submodule

    def forward(self, x, cap_feats):
        unified = self.unified_prepare(x, cap_feats)  # Hook can intercept here
```

---

## Defining `_sp_plan`

**Type definitions** (see [diffusers `_modeling_parallel.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/_modeling_parallel.py) for reference):

```python
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,   # Corresponds to diffusers' ContextParallelInput
    SequenceParallelOutput,  # Corresponds to diffusers' ContextParallelOutput
)
```

| Parameter | Description |
|-----------|-------------|
| `split_dim` | Dimension to split/gather (usually `1` for sequence) |
| `expected_dims` | Expected tensor rank for validation (optional) |
| `split_output` | `False`: shard **input** parameters; `True`: shard **output** tensors |
| `auto_pad` | Auto-pad if sequence not divisible by world_size (Ulysses only) |

**Key naming convention:**

| Key | Meaning | Python equivalent |
|-----|---------|-------------------|
| `""` | Root model | `model` |
| `"blocks.0"` | First element of ModuleList | `model.blocks[0]` |
| `"blocks.*"` | All elements of ModuleList | `for b in model.blocks` |
| `"outputs.main"` | ModuleDict entry | `model.outputs["main"]` |

**Dictionary key types:**

| Key type | `split_output` | Description |
|----------|----------------|-------------|
| `"param_name"` (str) | `False` | Shard **input parameter** by name |
| `0`, `1` (int) | `True` | Shard **output tuple** by index |

**Example** (similar to [diffusers `transformer_wan.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py)):

```python
class MyTransformer(nn.Module):
    _sp_plan = {
        # Shard rope module OUTPUTS (returns tuple)
        "rope": {
            0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # cos
            1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # sin
        },
        # Shard transformer block INPUT parameter
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        # Gather at final projection
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

---

## Hook flow

```
Input → [SequenceParallelSplitHook: pre_forward] → Module.forward() → [post_forward] → ...
                                                                              ↓
... → [SequenceParallelGatherHook: post_forward] → Output
```

1. **SplitHook** shards tensors before/after the target module
2. **Attention layers** handle Ulysses/Ring communication internally
3. **GatherHook** collects sharded outputs

The framework automatically applies these hooks when `sequence_parallel_size > 1`.

---

## Method 2: Intrusive modification (For complex cases)

For models with dynamic sharding logic that cannot be expressed via `_sp_plan`:

```python
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard, sp_gather

def forward(self, hidden_states, ...):
    if self.parallel_config.sequence_parallel_size > 1:
        hidden_states = sp_shard(hidden_states, dim=1)
        # ... computation ...
        output = sp_gather(output, dim=1)
    return output
```

---

## Choosing the right approach

| Scenario | Approach |
|----------|----------|
| Standard transformer | `_sp_plan` |
| Inline tensor ops need sharding | Extract to submodule + `_sp_plan` |
| Dynamic/conditional sharding | Intrusive modification |
