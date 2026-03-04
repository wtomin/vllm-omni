# Diffusion Dashboard — Parallelism Configuration Definition

This document defines the **parallelism configuration matrix** for the Diffusion Performance Dashboard. Both **Qwen-Image** (t2i) and **Wan2.2** (t2v) require monitoring the following server variants.

---

## 1. Configuration Overview

| Config ID | Description | `DiffusionParallelConfig` | GPU Count | Notes |
|-----------|-------------|---------------------------|-----------|-------|
| **baseline** | Single-GPU, no parallelism | default (all 1) | 1 | Reference for regression detection |
| **sp2-ulysses** | SP=2 (Ulysses mode) | `ulysses_degree=2`, `ring_degree=1` | 2 | Sequence split via All-to-All |
| **tp2** | Tensor Parallel = 2 | `tensor_parallel_size=2` | 2 | Weight sharding across 2 GPUs |
| **sp2-ring** | SP=2 (Ring mode) | `ulysses_degree=1`, `ring_degree=2` | 2 | Sequence split via Ring attention |

---

## 2. Parameter Definitions

### 2.1 Sequence Parallel (SP)

`sequence_parallel_size = ulysses_degree × ring_degree`

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequence_parallel_size` | int | Total SP degree; splits sequence dimension across GPUs |
| `ulysses_degree` | int | Ulysses-SP: All-to-All over Q/K/V heads |
| `ring_degree` | int | Ring-SP: K/V circulation in ring topology |

**SP=2 variants:**

- **sp2-ulysses**: `ulysses_degree=2`, `ring_degree=1` → `sequence_parallel_size=2` (Ulysses mode)
- **sp2-ring**: `ulysses_degree=1`, `ring_degree=2` → `sequence_parallel_size=2` (Ring mode)

### 2.2 Tensor Parallel (TP)

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor_parallel_size` | int | Shards model weights across GPUs; reduces per-GPU memory |

**tp2**: `tensor_parallel_size=2`

---

## 3. Stage Config / Engine Args Mapping

For `dashboard_configs.json`, each server variant is expressed via `server_params.update.stage_args`:

### 3.1 Baseline (no override)

```json
{
  "test_name": "test_qwen_image",
  "server_params": {
    "model": "Qwen/Qwen-Image",
    "stage_config_name": "qwen_image.yaml"
  }
}
```

### 3.2 SP=2 (Ulysses)

```json
{
  "server_params": {
    "model": "Qwen/Qwen-Image",
    "stage_config_name": "qwen_image.yaml",
    "update": {
      "stage_args": {
        "0": {
          "engine_args.parallel_config": {
            "ulysses_degree": 2,
            "ring_degree": 1,
            "sequence_parallel_size": 2,
            "tensor_parallel_size": 1
          }
        }
      }
    }
  }
}
```

### 3.3 TP=2

```json
{
  "server_params": {
    "model": "Qwen/Qwen-Image",
    "stage_config_name": "qwen_image.yaml",
    "update": {
      "stage_args": {
        "0": {
          "engine_args.parallel_config": {
            "tensor_parallel_size": 2,
            "ulysses_degree": 1,
            "ring_degree": 1,
            "sequence_parallel_size": 1
          }
        }
      }
    }
  }
}
```

### 3.4 SP=2 (Ring)

```json
{
  "server_params": {
    "model": "Qwen/Qwen-Image",
    "stage_config_name": "qwen_image.yaml",
    "update": {
      "stage_args": {
        "0": {
          "engine_args.parallel_config": {
            "ulysses_degree": 1,
            "ring_degree": 2,
            "sequence_parallel_size": 2,
            "tensor_parallel_size": 1
          }
        }
      }
    }
  }
}
```

---

## 4. Model Coverage

| Model | Task | baseline | sp2-ulysses | tp2 | sp2-ring |
|-------|------|:--------:|:------------:|:---:|:--------:|
| **Qwen/Qwen-Image** | t2i | ✅ | ✅ | ✅ | ✅ |
| **Wan2.2** | t2v | ✅ | ✅ | ✅ | ✅ |

Both models support Ulysses-SP, Ring-SP, and Tensor Parallel per [parallelism_acceleration.md](../user_guide/diffusion/parallelism_acceleration.md).

---

## 5. CLI / Offline Inference Equivalents

For manual runs or debugging:

**SP=2 (Ulysses):**
```bash
vllm serve Qwen/Qwen-Image --omni --usp 2 --ring 1
# or
python text_to_image.py --model Qwen/Qwen-Image --ulysses-degree 2 --ring-degree 1
```

**TP=2:**
```bash
vllm serve Qwen/Qwen-Image --omni --tensor-parallel-size 2
# or
python text_to_image.py --model Qwen/Qwen-Image --tensor-parallel-size 2
```

**SP=2 (Ring):**
```bash
vllm serve Qwen/Qwen-Image --omni --usp 1 --ring 2
# or
python text_to_image.py --model Qwen/Qwen-Image --ulysses-degree 1 --ring-degree 2
```

---

## 6. References

- `vllm_omni/diffusion/data.py` — `DiffusionParallelConfig` dataclass
- [Sequence Parallel Design](../design/feature/sequence_parallel.md)
- [Tensor Parallel Design](../design/feature/tensor_parallel.md)
- [Parallelism Acceleration Guide](../user_guide/diffusion/parallelism_acceleration.md)
- [RFC: Performance Dashboard for Diffusion Models](./perf_dashboard_diffusion.md)
