# Parallelism Acceleration Guide

This guide covers the parallelism methods in vLLM-Omni for speeding up diffusion model inference and reducing per-device memory requirements.

## Supported Methods

| Method | Description |
|--------|-------------|
| **[Tensor Parallelism](tensor_parallel.md)** | Shards DiT weights across GPUs to reduce per-GPU memory |
| **[Sequence Parallelism](sequence_parallel.md)** | Splits sequence dimension across GPUs (Ulysses-SP, Ring-Attention, or hybrid) for high-resolution images and videos |
| **[CFG-Parallel](cfg_parallel.md)** | Runs CFG positive/negative branches on separate GPUs for ~1.8x speedup on guided generation |
| **[VAE Patch Parallelism](vae_patch_parallel.md)** | Distributes VAE decode spatially across GPUs to reduce peak VAE memory |
| **[HSDP](hsdp.md)** | Shards full model weights via PyTorch FSDP2 to enable large-model inference on memory-constrained GPUs |
| **[Expert Parallelism](#expert-parallelism)** | Shards MoE expert blocks across GPUs for MoE models (e.g. HunyuanImage3.0) |

See [Supported Models](../diffusion_features.md#supported-models) for per-model compatibility.

---

## Expert Parallelism

Unlike Tensor Parallelism which shards every layer's weights, Expert Parallelism (EP) only shards the MoE expert MLP blocks. This significantly reduces the memory footprint of MoE models (e.g., HunyuanImage3.0) while maintaining constant dense-equivalent compute efficiency.

EP is enabled via `DiffusionParallelConfig.enable_expert_parallel`. The effective EP size equals `tp × sp × cfg × dp`, so at least one of TP/SP/CFG/DP must be set when EP is enabled.

### Offline Inference

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="tencent/HunyuanImage-3.0",
    parallel_config=DiffusionParallelConfig(
        tensor_parallel_size=8,
        enable_expert_parallel=True,
    ),
)

outputs = omni.generate(
    "A brown and white dog is running on the grass",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
        width=1024,
        height=1024,
    ),
)
```
