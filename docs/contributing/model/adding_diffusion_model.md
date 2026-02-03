# Adding a Diffusion Model
This guide walks through the process of adding a new Diffusion model to vLLM-Omni, using Qwen/Qwen-Image-Edit as a comprehensive example.

# Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Step-by-Step Implementation (Basic)](#step-by-step-implementation-basic)



# Overview
When add a new diffusion model into vLLM-Omni, additional adaptation work is required due to the following reasons:

+ New model must follow the framework’s parameter passing mechanisms and inference flow.

+ Replacing the model’s default implementations with optimized modules, which is necessary to achieve the better performance.

The diffusion execution flow as follow:
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/vllm-omni-diffusion-flow.png">
    <img alt="Diffusion Flow" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/vllm-omni-diffusion-flow.png" width=55%>
  </picture>
</p>


# Directory Structure
File Structure for Adding a New Diffusion Model

```
vllm_omni/
└── examples/
    └──offline_inference
        └── example script                # reuse existing if possible (e.g., image_edit.py)
    └──online_serving
        └── example script
└── diffusion/
    └── registry.py                       # Registry work
    ├── request.py                        # Request Info
    └── models/your_model_name/           # Model directory (e.g., qwen_image)
        ├── pipeline_xxx.py               # Pipeline implementation (e.g., pipeline_qwen_image_edit.py)
        └── xxx_transformer.py            # Transformer model implementation (e.g., qwen_image_transformer.py)
```

# Step-by-step-implementation (Basic)

This is the basic adaptation tutorial on how to add a diffusion model step by step. Following this tutorial, you will know how to adapt an existing model/pipeline in **HuggingFace Diffusers** to vLLM-Omni, and support the basic features (e.g., online/offline serving, batch request).

For the advanced adaptation tutorial for optimized features (e.g., parallelism, cache acceleration), please see

## Step 1: Model/Pipeline Implementation

### 1.1 Implement the Model class

We would recommend you to first copy the `xxx_transformer.py` from diffusers to `vllm_omni/diffusion/models/your_model_name/ `, and make the following revisions:

1. Remove Diffusers' `Mixin` inheritance

Diffusers' `Mixin` classes (e.g., `ModelMixin`, `AttentionModuleMixin`) are not necessary for `vLLM-Omni`. Please remove them. For example:
```diff
- class LongCatImageAttention(torch.nn.Module, AttentionModuleMixin):
+ class LongCatImageAttention(torch.nn.Module):
```

2. Replace Diffusers' Attention

Latest diffusers model implementation uses `dispatch_attention_fn` to run attention computation on different backends. A typical function call looks like:

```python
hidden_states = dispatch_attention_fn(
    query, # bs, seq_len, num_heads, head_dim
    key # bs, seq_len, num_heads, head_dim
    value, # bs, seq_len, num_heads, head_dim
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    backend=self._attention_backend,
    parallel_config=self._parallel_config,
)
```

vLLM-Omni has its own [`Attention`](../../design/module/dit_module.md#5-acceleration-components) interface, which can support various attention backends, and allow users to set different parallelism configuration (See in advanced tutorial). See more details about backends selection and attention parallelsim in [Attention Doc](../../design/module/dit_module.md#5-acceleration-components).

Here is an example of how to use vLLM-Omni's `Attention`:
```python
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
class xxxModule(torch.nn.Module)
    def __init__(...):
        self.attn = Attention(
        num_heads=self.num_heads,
        head_size=self.head_dim,
        softmax_scale=1.0 / (self.head_dim**0.5),
        causal=False,
        num_kv_heads=self.num_kv_heads,
    )

    def forward(...):
        attn_metadata =  AttentionMetadata(attn_mask=attention_mask)
        hidden_states = self.attn(
            query, # bs, seq_len, num_heads, head_dim
            key, # bs, seq_len, num_heads, head_dim
            value, # bs, seq_len, num_heads, head_dim
            attn_metadata=attn_metadata
        )
```

3. Miscellaneous Replacement

- Replace diffusers logger by vLLM's logger

```diff
- from diffusers.utils import logging
- logger = logging.get_logger(__name__)
+ from vllm.logger import init_logger
+ logger = init_logger(__name__)
```

- Replace by vLLM/vLLM-Omni custom Ops (if any)

```diff
+ from vllm.model_executor.layers.layernorm import RMSNorm
+ from vllm_omni.diffusion.layers.rope import RotaryEmbedding
+ from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
```

- Remove gradient_checkpoint related code (not used in inference)

```diff
    for index_block, block in enumerate(self.transformer_blocks):
-        if torch.is_grad_enabled() and self.gradient_checkpointing:
-            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
-               block,
-               hidden_states,
-               encoder_hidden_states,
-               None,  
-               temb,
-               image_rotary_emb,
-               block_attention_kwargs,
-               modulate_index,
-           )

```

So far, you have adapted the diffusers' model implementation to vLLM-Omni, with basic features, like various attention backends.


### 1.2 Implement the Pipeline class

We would recommend you to first copy the `pipeline_xxx.py` from diffusers to `vllm_omni/diffusion/models/your_model_name/`, and make the following revisions:

1. Remove Diffusers' pipeline inheritance

For example,
```diff
- class QwenImagePipeline(DiffusionPipeline, QwenImageLoraLoaderMixin):
+ class QwenImagePipeline(nn.Module):
```

2. Edit the pipeline's `__init__` function

An example pipeline `__init__` function in diffusers is like:

```python
class QwenImagePipeline(DiffusionPipeline, QwenImageLoraLoaderMixin):
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        transformer: QwenImageTransformer2DModel,
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

Since `self.register_modules` is no longer supported after removing `DiffusionPipeline` inheritance, we need to register each module manually:

```python
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
class QwenImagePipeline(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        model = od_config.model
        self.device = get_local_device()
        # Check if model is a local path
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        )
        self.vae = AutoencoderKLQwenImage.from_pretrained(model, subfolder="vae", local_files_only=local_files_only).to(
            self.device
        )
        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImageTransformer2DModel)
        self.transformer = QwenImageTransformer2DModel(od_config=od_config, **transformer_kwargs)

        self.tokenizer = Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

```

This will help the `QwenImagePipeline` to locate the local checkpoint directory and load the weight to each module.


3. Edit the pipeline's `__call__` function

First, you need to replace `__call__` function by `forward` function.

```diff
-    @torch.no_grad()
-    @replace_example_docstring(EXAMPLE_DOC_STRING)
-    def __call__(
+    def forward(
        self,
        ...
    ):
```

Then add the `OmniDiffusionRequest` as its' first argument, and `DiffusionOutput` as its output type:
```diff
+ from vllm_omni.diffusion.request import OmniDiffusionRequest
+ from vllm_omni.diffusion.data import DiffusionOutput
    def forward(
        self,
+       req: OmniDiffusionRequest,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,

-    ):
+    ) -> DiffusionOutput:
```

`OmniDiffusionRequest` contains the prompts and sampling parameters for the diffusion pipeline execution. All arguments can be passed via `OmniDiffusionRequest`. Therefore, the original arguments to pipeline's `forward` function need special handlings:

For example, prompt will be defined by `req.prompts` if `req.prompts` is not None:
```python
prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
```

Height and width will be defined by `req.sampling_params` if provided.
```python
height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
```

Image postprocessing should be extracted from the pipeline's forward function, and defined in a separate function.

```diff
    def forward(...):
        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
-           processed_image = self.image_processor.postprocess(image, output_type=output_type)

```

This separate function contains the image postprocessing steps, for example:
```python
def get_qwen_image_post_process_func(
    od_config: OmniDiffusionConfig,
):
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(
        images: torch.Tensor,
    ):
        return image_processor.postprocess(images)

    return post_process_func

```

For image editing pipelines, the `req.prompt` also contains `multi_modal_data` including images, which need pre-processing. Please also extract the pre-processing steps from `forward` function, and define it like `get_qwen_image_edit_pre_process_func`.

Finally, the output of pipeline's `forward` function should be wrapped by:

```diff
- return QwenImagePipelineOutput(images=image)
+ return DiffusionOutput(output=image)
```


4. Add HF weight helper

In order to run HF weight downloading and loading automatically, you need a helper `DiffusersPipelineLoader` and `AutoWeightsLoader`. An example usage:

```python
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm.model_executor.models.utils import AutoWeightsLoader
class QwenImagePipeline(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
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
        # customize the weight loading behavior
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

```


Sofar, you have adapted the diffusers' pipeline to vLLM-Omni. Good job!


## Step 2: Registry
+ registry diffusion model in `vllm_omni/diffusion/registry.py`
```python
_DIFFUSION_MODELS = {
    # arch:(mod_folder, mod_relname, cls_name)
    ...
    "QwenImageEditPipeline": (
        "qwen_image",
        "pipeline_qwen_image_edit",
        "QwenImageEditPipeline",
    ),
    ...
}
```
+ registry pre-process get function (if any)
```python
_DIFFUSION_PRE_PROCESS_FUNCS = {
    # arch: pre_process_func
    ...
    "QwenImageEditPipeline": "get_qwen_image_edit_pre_process_func",
    ...
}
```

+ registry post-process get function
```python
_DIFFUSION_POST_PROCESS_FUNCS = {
    # arch: post_process_func
    ...
    "QwenImageEditPipeline": "get_qwen_image_edit_post_process_func",
    ...
}
```

## Step 3: Add an Example Script
For each newly integrated model, we need to provide examples script under the examples/ to demonstrate how to initialize the pipeline with Omni, pass in user inputs, and generate outputs.
Key point for writing the example:

+ Use the Omni entrypoint to load the model and construct the pipeline.

+ Show how to format user inputs and pass them via omni.generate(...).

+ Demonstrate the common runtime arguments, such as:

    + model path or model name

    + input image(s) or prompt text

    + key diffusion parameters (e.g., inference steps, guidance scale)

+ Save or display the generated results so users can validate the integration.


# Step-by-step-implementation (Advanced)

This is the advanced adaptation tutorial how to implement an existing model/pipeline with various features, like sequence parallelism, quantization, cache acceleration, compilation, etc.

## torch.compile

`torch.compile` is automatically enabled on `transformer._repeated_blocks`. Please add one single line of code to define it:
```diff
class Flux2Transformer2DModel(nn.Module):
+    _repeated_blocks = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]
```

## Tensor Parallelism

Tensor parallelism is supported via vLLM interface. For example, in `flux_transformer.py`:

```diff
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear

```

See detailed instruction in

## CFG parallelism

See instructions in [how to parallelize a pipeline for CFG parallel](../features/cfg_parallel.md)

## Sequence Parallelism

See instructions in [How to parallelize a new model for SP](../features/sequence_parallel.md)

## Patch VAE Parallelism

coming soon.

## What's included in Your Pull Request

When submitting a pull request to add support for a new model, please include the following information in the PR description:

+ Output verification: provide generation outputs to verify correctness and model behavior.

+ Inference speed: provide a comparison with the corresponding implementation in Diffusers.

+ Parallelism support: specify the supported parallel sizes and any relevant limitations.

+ Cache acceleration: check whether the model can be accelerated using Cache-Dit or not.

Providing these details helps reviewers evaluate correctness, performance improvements, and parallel scalability of the new model integration.

Test scripts are recommended for good maintainous. If your PR introduces some changes to the internal/external interface, please add test script. For comprehensive testing guidelines, please refer to the [Test File Structure and Style Guide](../ci/tests_style.md).


## Adding a Model Recipe
After implementing and testing your model, please add a model recipe to the [vllm-project/recipes](https://github.com/vllm-project/recipes) repository. This helps other users understand how to use your model with vLLM-Omni.
