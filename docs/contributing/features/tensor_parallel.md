# How to add Tensor Parallel support for a new model

This section describes how to add Tensor Parallel (TP) to a diffusion **transformer model**. We use the Z-Image transformer (`vllm_omni/diffusion/models/z_image/z_image_transformer.py`) and FLUX transformer as reference implementations.

Tensor Parallel distributes model weights across multiple GPUs, enabling inference of large models that wouldn't fit in a single GPU's memory. It provides **near-linear speedup** for compute-bound operations while reducing per-GPU memory usage proportionally to the number of GPUs.

---

## Overview

### What is Tensor Parallel?

Tensor Parallel (TP) is a model parallelism technique that **shards model weights** across multiple GPUs. Each GPU holds only a portion of the model's parameters and computes only part of each layer's output.

Diffusion transformers contain large attention and MLP layers. We can use Tensor Parallel (TP) to shard the model dimension across multiple GPUs:

### Key API:

The major function for querying TP state `get_tensor_model_parallel_world_size()`:

```python
from vllm.distributed import get_tensor_model_parallel_world_size

# Get current tensor parallel world size
tp_size = get_tensor_model_parallel_world_size()
```

**Returns:**
- `1` - TP is not enabled (single GPU or before initialization)
- `N` - Running with N-way tensor parallelism (N GPUs)

---

## Tensor Parallel

### Parallel Layers

Tensor Parallel uses two fundamental strategies for sharding linear layers by splitting the weight matrix:

**1. `ColumnParallelLinear`**

The input tensor is replicated. The weight matrix is partitioned along the columns (output dimension). The result is partitioned along the column dimension. Typically used for the first FFN layer and the separated QKV transformation of the attention layer in the original Transformer.

**2. `RowParallelLinear`**

The input tensor is partitioned along the hidden dimension. The weight matrix is partitioned along the rows (input dimension). An all-reduce operation is performed after the matrix multiplication to reduce the results. Typically used for the second FFN layer and the output linear transformation of the attention layer.

**3. `QKVParallelLinear`**

Parallel linear layer for the query, key, and value projections of the multi-head and grouped-query attention mechanisms. When number of key/value heads are less than the world size, this class replicates the key/value heads properly. This class handles the weight loading and replication of the weight matrices.

**4. `ReplicatedLinear`**

Replicates the inputs and weights across multiple GPUs. No memory saving.

### Standard Patterns

**Pattern 1: MLP Block (Up-Down)**

Most transformer MLPs follow this pattern:

```python
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Column parallel: weight split by columns [hidden_dim/N, dim]
        self.w1 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.act = nn.GELU()

        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,  # Input already sharded from w1
        )

    def forward(self, x):
        # x: [batch, seq, dim] (replicated on all GPUs)
        # w1 outputs sharded [batch, seq, hidden_dim/N]
        x = self.w1(x)
        # act operates on sharded tensors (no communication)
        x = self.act(x)
        # w2 outputs full dim [batch, seq, dim] via all-reduce
        x = self.w2(x)
        return x
```

**Pattern 2: Attention (QKV-Out)**

Attention shards along the **head dimension**:

```python
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.head_dim = dim // num_heads

        # Column parallel: QKV weight split by columns
        # Each GPU gets num_heads/N heads
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
        )

        # Row parallel: output weight split by rows
        self.to_out = RowParallelLinear(
            dim,
            dim,
            bias=False,
            input_is_parallel=True,  # Input sharded from attention
        )

        self.attn = Attention(
            num_heads=num_heads // tp_size,  # Local heads per GPU
            head_size=self.head_dim,
        )

    def forward(self, x):
        # x: [batch, seq, dim] (replicated)
        # to_qkv outputs sharded [batch, seq, (q+k+v) * head_dim/N]
        qkv = self.to_qkv(x)
        # Split into Q, K, V (each sharded on heads)
        q, k, v = qkv.split([...], dim=-1)
        # Attention computed independently on each GPU
        out = self.attn(q, k, v)
        # to_out all-reduces to full dim
        out = self.to_out(out)
        return out
```

---

## Step-by-Step: Implementing Tensor Parallel

### Step 1: Identify Linear Layers

Find all `nn.Linear` layers in your transformer that need to be sharded:

**Key questions:**
1. Which layers should be column parallel (weight split by columns)?
2. Which layers should be row parallel (weight split by rows)?

### Step 2: Replace Linear Layers

Replace `nn.Linear` with parallel equivalents from `vllm.model_executor.layers.linear`:

## TP Constraints and Validation

### Required Divisibility Constraints

For correct TP operation, these dimensions **must be divisible** by `tensor_parallel_size`:

| Dimension | Reason | Example Error |
|-----------|--------|---------------|
| `hidden_dim` | Model dimension sharded by ColumnParallel | `hidden_dim=3840, tp=3` ❌ (3840 % 3 ≠ 0) |
| `num_heads` | Heads sharded by QKVParallelLinear | `num_heads=30, tp=4` ❌ (30 % 4 ≠ 0) |
| `num_kv_heads` | KV heads sharded by QKVParallelLinear | `num_kv_heads=30, tp=4` ❌ (30 % 4 ≠ 0) |

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
    --output "tp_enabled.png" \
    --tensor_parallel_size 2
```
Please record the "e2e_time_ms" in the log and the generated result, and compare them with the results of Tensor-Parallel not enabled. Please record the comparison results in your PR.


## Troubleshooting

### Issue: TP not activating

**Symptoms:** Model runs on single GPU, no memory savings or speedup.

**Causes & Solutions:**

1. **TP size not set:**

   **Check current TP size:**
   ```python
   from vllm.distributed import get_tensor_model_parallel_world_size

   # Check TP configuration
   tp_size = get_tensor_model_parallel_world_size()
   print(f"Current TP size: {tp_size}")
   # Should print N for N-way TP, 1 for no TP
   ```

   Or from command line:
   ```bash
   python -c "from vllm.distributed import get_tensor_model_parallel_world_size; print(get_tensor_model_parallel_world_size())"
   ```

   **Solution:** Initialize with `tensor_parallel_size=N`:
   ```python
   from vllm_omni.diffusion.data import DiffusionParallelConfig

   parallel_config = DiffusionParallelConfig(tensor_parallel_size=2)
   model = Omni(model="model_name", parallel_config=parallel_config)

   # Verify TP is active
   from vllm.distributed import get_tensor_model_parallel_world_size
   assert get_tensor_model_parallel_world_size() == 2, "TP not initialized correctly"
   ```

2. **Still using `nn.Linear`:**

   **Solution:** Replace with parallel layers:
   ```python
   # ❌ BAD
   self.proj = nn.Linear(dim, dim)

   # ✅ GOOD
   self.proj = RowParallelLinear(dim, dim, input_is_parallel=True)
   ```

### Issue: Dimension mismatch errors

**Symptoms:** `RuntimeError: shape mismatch` during forward pass.

**Causes & Solutions:**

1. **Missing `input_is_parallel=True`:**

   **Problem:** RowParallelLinear expects sharded input but receives full tensor.

   **Solution:** Set `input_is_parallel=True` when input comes from ColumnParallelLinear:
   ```python
   # ✅ GOOD: Correct pairing
   self.w1 = ColumnParallelLinear(dim, hidden_dim)
   self.w2 = RowParallelLinear(
       hidden_dim,
       dim,
       input_is_parallel=True,  # Input sharded from w1
   )
   ```

2. **Incorrect split dimensions:**

   **Problem:** QKV split sizes don't match sharded dimensions.

   **Solution:** Use `self.to_qkv.num_heads` (local heads per GPU):
   ```python
   # ❌ BAD: Uses total heads
   q_size = self.total_num_heads * self.head_dim

   # ✅ GOOD: Uses local heads
   q_size = self.to_qkv.num_heads * self.head_dim
   ```

---


## Reference Implementations

Complete examples in the codebase:

| Model | Path | Pattern | Notes |
|-------|------|---------|-------|
| **Z-Image** | `vllm_omni/diffusion/models/z_image/z_image_transformer.py` | Standard TP | Full implementation with validation |
| **FLUX** | `vllm_omni/diffusion/models/flux/flux_transformer.py` | Dual-stream | Image + text streams |
| **Qwen-Image** | `vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py` | Standard TP | With RoPE |
| **TP Tests** | `tests/e2e/offline_inference/test_zimage_tensor_parallel.py` | E2E testing | TP correctness and performance |
| **Constraint Tests** | `tests/diffusion/models/z_image/test_zimage_tp_constraints.py` | Unit testing | Validation logic |

---

## Summary

Adding Tensor Parallel support to a transformer:

1. ✅ **Identify linear layers** - Which layers should be sharded?
2. ✅ **Replace with parallel layers** - Use QKVParallelLinear, ColumnParallelLinear, RowParallelLinear
3. ✅ **Validate TP constraints** - Ensure dimensions divisible by TP size
4. ✅ **Test with valid `tp_size`* - Check the memory usage, inference speed, and generative quality.
