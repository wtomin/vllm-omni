# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents, PipelineParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import DenoiseProgressMixin
from vllm_omni.diffusion.lora.loader import WanLoraLoaderMixin
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler import WanEulerScheduler
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni.diffusion.postprocess import interpolate_video_tensor
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState

logger = logging.getLogger(__name__)
DEBUG_PERF = False
WAN_SAMPLE_SOLVER_CHOICES = {"unipc", "euler"}


def build_wan_scheduler(sample_solver: str, flow_shift: float) -> Any:
    if sample_solver == "unipc":
        return FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )
    if sample_solver == "euler":
        return WanEulerScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
        )

    raise ValueError(
        f"Unsupported Wan sample_solver: {sample_solver}. Expected one of: {sorted(WAN_SAMPLE_SOLVER_CHOICES)}"
    )


def resolve_wan_sample_solver(req: OmniDiffusionRequest, default: str = "unipc") -> str:
    return resolve_wan_sample_solver_from_sampling(req.sampling_params, default=default)


def resolve_wan_sample_solver_from_sampling(
    sampling_params: OmniDiffusionSamplingParams,
    default: str = "unipc",
) -> str:
    extra_args = getattr(sampling_params, "extra_args", {}) or {}
    raw = extra_args.get("sample_solver", default)
    sample_solver = str(raw).strip().lower()
    if sample_solver not in WAN_SAMPLE_SOLVER_CHOICES:
        raise ValueError(f"Invalid sample_solver={raw!r}. Expected one of: {sorted(WAN_SAMPLE_SOLVER_CHOICES)}")
    return sample_solver


def resolve_wan_flow_shift(req: OmniDiffusionRequest, od_config: OmniDiffusionConfig) -> float:
    return resolve_wan_flow_shift_from_sampling(req.sampling_params, od_config)


def resolve_wan_flow_shift_from_sampling(
    sampling_params: OmniDiffusionSamplingParams,
    od_config: OmniDiffusionConfig,
) -> float:
    extra_args = getattr(sampling_params, "extra_args", {}) or {}
    raw_flow_shift = extra_args.get("flow_shift")
    if raw_flow_shift is None:
        raw_flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0

    try:
        return float(raw_flow_shift)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid flow_shift={raw_flow_shift!r}. flow_shift must be a float.") from exc


def resolve_wan_guidance_scales(
    sampling_params: OmniDiffusionSamplingParams,
    default_guidance_scale: float,
) -> tuple[float, float]:
    guidance_scale = (
        sampling_params.guidance_scale if sampling_params.guidance_scale_provided else default_guidance_scale
    )
    guidance_low = guidance_scale if isinstance(guidance_scale, (int, float)) else guidance_scale[0]
    guidance_high = (
        sampling_params.guidance_scale_2
        if sampling_params.guidance_scale_2_provided and sampling_params.guidance_scale_2 is not None
        else (
            guidance_scale[1] if isinstance(guidance_scale, (list, tuple)) and len(guidance_scale) > 1 else guidance_low
        )
    )
    return guidance_low, guidance_high


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    """Retrieve latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        # Try to download config from HF Hub
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(
    config: dict, quant_config: QuantizationConfig | None = None, prefix: str = ""
) -> WanTransformer3DModel:
    """Create WanTransformer3DModel from config dict."""
    kwargs: dict = {}

    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    if "num_attention_heads" in config:
        kwargs["num_attention_heads"] = config["num_attention_heads"]
    if "attention_head_dim" in config:
        kwargs["attention_head_dim"] = config["attention_head_dim"]
    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
    if "out_channels" in config:
        kwargs["out_channels"] = config["out_channels"]
    if "text_dim" in config:
        kwargs["text_dim"] = config["text_dim"]
    if "freq_dim" in config:
        kwargs["freq_dim"] = config["freq_dim"]
    if "ffn_dim" in config:
        kwargs["ffn_dim"] = config["ffn_dim"]
    if "num_layers" in config:
        kwargs["num_layers"] = config["num_layers"]
    if "cross_attn_norm" in config:
        kwargs["cross_attn_norm"] = config["cross_attn_norm"]
    if "eps" in config:
        kwargs["eps"] = config["eps"]
    if "image_dim" in config:
        kwargs["image_dim"] = config["image_dim"]
    if "added_kv_proj_dim" in config:
        kwargs["added_kv_proj_dim"] = config["added_kv_proj_dim"]
    if "rope_max_seq_len" in config:
        kwargs["rope_max_seq_len"] = config["rope_max_seq_len"]
    if "pos_embed_seq_len" in config:
        kwargs["pos_embed_seq_len"] = config["pos_embed_seq_len"]

    if "quantization_config" in config:
        from vllm_omni.quantization.factory import resolve_quant_config_from_disk

        quant_config = resolve_quant_config_from_disk(quant_config, config["quantization_config"])

    if quant_config is not None:
        kwargs["quant_config"] = quant_config
    if prefix:
        kwargs["prefix"] = prefix

    return WanTransformer3DModel(**kwargs)


def get_wan22_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
        sampling_params=None,
    ):
        if sampling_params is not None and sampling_params.output_type is not None:
            output_type = sampling_params.output_type
        if output_type == "latent":
            return video
        video_metadata = {}
        if sampling_params is not None and getattr(sampling_params, "enable_frame_interpolation", False):
            video, multiplier = interpolate_video_tensor(
                video,
                exp=sampling_params.frame_interpolation_exp,
                scale=sampling_params.frame_interpolation_scale,
                model_path=sampling_params.frame_interpolation_model_path,
            )
            video_metadata["video_fps_multiplier"] = multiplier
        return {
            "payload": {"video": video_processor.postprocess_video(video, output_type=output_type)},
            "metadata": {"video": video_metadata} if video_metadata else {},
        }

    return post_process_func


def get_wan22_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for Wan2.2: optionally load and resize input image for I2V mode."""
    import numpy as np

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        has_image = raw_image is not None and (not isinstance(raw_image, list) or bool(raw_image))
        request.batch_compatibility_key = ("wan22_image_condition", has_image)
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        if "additional_information" not in prompt:
            prompt["additional_information"] = {}

        if raw_image is None:
            request.prompt = prompt
            return request

        if not isinstance(raw_image, (str, PIL.Image.Image)):
            raise TypeError(
                f"""Unsupported image format {raw_image.__class__}.""",
                """Please correctly set `"multi_modal_data": {"image": <an image object or file path>, …}`""",
            )
        image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image

        # Calculate dimensions based on aspect ratio if not provided
        if request.sampling_params.height is None or request.sampling_params.width is None:
            # Default max area for 720P
            max_area = 720 * 1280
            aspect_ratio = image.height / image.width

            # Calculate dimensions maintaining aspect ratio
            mod_value = 16  # Must be divisible by 16
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

            if request.sampling_params.height is None:
                request.sampling_params.height = height
            if request.sampling_params.width is None:
                request.sampling_params.width = width

        # Resize image to target dimensions
        image = image.resize(
            (request.sampling_params.width, request.sampling_params.height),  # type: ignore # Above has ensured that width & height are not None
            PIL.Image.Resampling.LANCZOS,
        )
        prompt["multi_modal_data"]["image"] = image  # type: ignore # key existence already checked above

        request.prompt = prompt
        return request

    return pre_process_func


class Wan22Pipeline(
    nn.Module,
    PipelineParallelMixin,
    CFGParallelMixin,
    ProgressBarMixin,
    DenoiseProgressMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
    WanLoraLoaderMixin,
):
    supports_request_batch = True
    supports_step_execution: ClassVar[bool] = True
    supports_pp_latent_static_layout: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = ["transformer", "transformer_2"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Read model_index.json to detect expand_timesteps mode (for TI2V-5B)
        self.expand_timesteps = False
        self.has_transformer_2 = False
        if local_files_only:
            model_index_path = os.path.join(model, "model_index.json")
            if os.path.exists(model_index_path):
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
            # Check if this is a two-stage model (MoE with transformer_2)
            transformer_2_path = os.path.join(model, "transformer_2")
            self.has_transformer_2 = os.path.exists(transformer_2_path)
        else:
            # For remote models, download and read model_index.json
            try:
                from huggingface_hub import hf_hub_download

                model_index_path = hf_hub_download(repo_id=model, filename="model_index.json")
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
                    # Check transformer_2 from model_index
                    transformer_2_info = model_index.get("transformer_2", [None, None])
                    self.has_transformer_2 = transformer_2_info[0] is not None
            except Exception:
                pass

        self.boundary_ratio = od_config.boundary_ratio

        # Determine which transformers to load based on boundary_ratio
        # boundary_ratio=1.0: only load transformer_2 (low-noise stage only)
        # boundary_ratio=0.0: only load transformer (high-noise stage only)
        # otherwise: load both transformers
        load_transformer = self.boundary_ratio != 1.0 if self.boundary_ratio is not None else True
        load_transformer_2 = self.has_transformer_2 and (
            self.boundary_ratio != 0.0 if self.boundary_ratio is not None else True
        )

        # Set up weights sources for transformer(s)
        self.weights_sources = []
        if load_transformer:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer",
                    revision=None,
                    prefix="transformer.",
                    fall_back_to_pt=True,
                )
            )
        if load_transformer_2:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer_2",
                    revision=None,
                    prefix="transformer_2.",
                    fall_back_to_pt=True,
                )
            )

        # See ``hub_prefetch.py`` for the transformers v5 subfolder race.
        component_subfolders = ["tokenizer", "text_encoder", "vae"]
        prefetch_subfolders(
            model,
            component_subfolders,
            local_files_only=local_files_only,
        )

        # ``from_pretrained_with_prefetch`` re-prefetches and retries if the
        # cache is still half-written (the missing-shard ``OSError`` and the
        # default-``UMT5Config`` size-mismatch ``RuntimeError`` seen on multi
        # -worker HSDP / ring launches), instead of crashing the worker.
        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            model,
            subfolder="tokenizer",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
        )
        self.text_encoder = from_pretrained_with_prefetch(
            UMT5EncoderModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        # Initialize transformers with correct config (weights loaded via load_weights)
        if load_transformer:
            transformer_config = load_transformer_config(model, "transformer", local_files_only)
            self.transformer = self._create_transformer(transformer_config)
        else:
            self.transformer = None

        if load_transformer_2:
            transformer_2_config = load_transformer_config(model, "transformer_2", local_files_only)
            self.transformer_2 = self._create_transformer(transformer_2_config)
        else:
            self.transformer_2 = None

        # Store the active transformer config
        if load_transformer:
            self.transformer_config = self.transformer.config
        elif load_transformer_2:
            self.transformer_config = self.transformer_2.config
        else:
            raise RuntimeError("No transformer loaded")

        self._sample_solver = "unipc"
        self._flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0
        self.scheduler = build_wan_scheduler(self._sample_solver, self._flow_shift)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8

        self._guidance_scale = None
        self._guidance_scale_2 = None
        self._num_timesteps = None
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _create_transformer(self, config: dict) -> WanTransformer3DModel:
        """Create a transformer from a config dict. Respects od_config.quantization_config."""
        quant_config = getattr(self.od_config, "quantization_config", None)
        return create_transformer_from_config(config, quant_config=quant_config)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def _active_dtype(self) -> torch.dtype:
        if self.transformer is not None:
            return self.transformer.dtype
        if self.transformer_2 is not None:
            return self.transformer_2.dtype
        return self.text_encoder.dtype

    @staticmethod
    def _extract_step_prompt_fields(
        prompt: OmniTextPrompt | None,
    ) -> tuple[str, str | None, torch.Tensor | None, torch.Tensor | None, PIL.Image.Image | torch.Tensor | None]:
        if isinstance(prompt, str):
            return prompt, None, None, None, None
        if prompt is None:
            return "", None, None, None, None

        multi_modal_data = prompt.get("multi_modal_data", {})
        raw_image = multi_modal_data.get("image")
        if isinstance(raw_image, list):
            if len(raw_image) > 1:
                logger.warning("Received multiple images for one Wan request; using only the first image.")
            raw_image = raw_image[0] if raw_image else None
        if isinstance(raw_image, str):
            raw_image = PIL.Image.open(raw_image)

        return (
            prompt.get("prompt") or "",
            prompt.get("negative_prompt"),
            prompt.get("prompt_embeds"),
            prompt.get("negative_prompt_embeds"),
            cast(PIL.Image.Image | torch.Tensor | None, raw_image),
        )

    @staticmethod
    def _ensure_prompt_batch(embeds: torch.Tensor) -> torch.Tensor:
        return embeds.unsqueeze(0) if embeds.ndim == 2 else embeds

    def _resolve_boundary_timestep(self, sampling: OmniDiffusionSamplingParams) -> float:
        boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else sampling.boundary_ratio
        if boundary_ratio is None:
            boundary_ratio = 0.875
            logger.warning("boundary_ratio is required for T2V generation. using default value 0.875")
        return boundary_ratio * self.scheduler.config.num_train_timesteps

    def _select_step_model_and_guidance(
        self,
        timestep: torch.Tensor,
        *,
        boundary_timestep: float | None,
        guidance_low: float,
        guidance_high: float,
    ) -> tuple[nn.Module, float]:
        if boundary_timestep is not None and timestep < boundary_timestep:
            current_guidance_scale = guidance_high
            if self.transformer_2 is not None:
                current_model = self.transformer_2
            elif self.transformer is not None:
                current_model = self.transformer
            else:
                raise RuntimeError("No transformer available for low-noise stage")
        else:
            current_guidance_scale = guidance_low
            if self.transformer is not None:
                current_model = self.transformer
            elif self.transformer_2 is not None:
                current_model = self.transformer_2
            else:
                raise RuntimeError("No transformer available for high-noise stage")
        return current_model, current_guidance_scale

    def _prepare_step_image_condition(
        self,
        image: PIL.Image.Image | torch.Tensor,
        *,
        height: int,
        width: int,
        num_frames: int,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from diffusers.video_processor import VideoProcessor

        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        if isinstance(image, PIL.Image.Image):
            image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
            image_tensor = video_processor.preprocess(image, height=height, width=width)
        else:
            image_tensor = image.unsqueeze(0) if image.ndim == 3 else image
        image_tensor = image_tensor.repeat_interleave(batch_size, dim=0)

        image_tensor = image_tensor.unsqueeze(2).to(device=device, dtype=self.vae.dtype)
        latent_condition = retrieve_latents(self.vae.encode(image_tensor), sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latent_condition.device, latent_condition.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latent_condition.device, latent_condition.dtype
        )
        latent_condition = ((latent_condition - latents_mean) * latents_std).to(torch.float32)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        first_frame_mask = torch.ones(
            batch_size,
            1,
            num_latent_frames,
            latent_height,
            latent_width,
            dtype=torch.float32,
            device=device,
        )
        first_frame_mask[:, :, 0] = 0
        return latent_condition, first_frame_mask

    def prepare_encode(
        self,
        state: StepRequestState,
        **kwargs: Any,
    ) -> StepRequestState:
        del kwargs
        sampling = state.sampling
        prompt, negative_prompt, prompt_embeds, negative_prompt_embeds, image = self._extract_step_prompt_fields(
            state.prompt
        )
        if prompt_embeds is None and not prompt:
            raise ValueError("Prompt is required for Wan2.2 generation when prompt_embeds are not provided.")

        height = sampling.height or 480
        width = sampling.width or 832
        num_frames = sampling.num_frames or 81

        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        num_steps = 40 if sampling.num_inference_steps is None else sampling.num_inference_steps
        output_type = sampling.output_type or "np"
        num_outputs_per_prompt = sampling.num_outputs_per_prompt or 1

        guidance_low, guidance_high = resolve_wan_guidance_scales(sampling, default_guidance_scale=4.0)
        self._guidance_scale = guidance_low
        self._guidance_scale_2 = guidance_high
        boundary_timestep = self._resolve_boundary_timestep(sampling)

        self.check_inputs(
            prompt=prompt if prompt_embeds is None else None,
            negative_prompt=negative_prompt if negative_prompt_embeds is None else None,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale_2=guidance_high,
            boundary_ratio=boundary_timestep / self.scheduler.config.num_train_timesteps,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        dtype = self._active_dtype()
        do_classifier_free_guidance = guidance_low > 1.0 or guidance_high > 1.0
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=num_outputs_per_prompt,
                max_sequence_length=sampling.max_sequence_length or 512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = self._ensure_prompt_batch(prompt_embeds).to(device=device, dtype=dtype)
            prompt_embeds = prompt_embeds.repeat_interleave(num_outputs_per_prompt, dim=0)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = self._ensure_prompt_batch(negative_prompt_embeds).to(
                    device=device, dtype=dtype
                )
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_outputs_per_prompt, dim=0)
            elif do_classifier_free_guidance:
                _, negative_prompt_embeds = self.encode_prompt(
                    prompt="",
                    negative_prompt=negative_prompt,
                    do_classifier_free_guidance=True,
                    num_videos_per_prompt=num_outputs_per_prompt,
                    max_sequence_length=sampling.max_sequence_length or 512,
                    device=device,
                    dtype=dtype,
                )

        sample_solver = resolve_wan_sample_solver_from_sampling(sampling, default=self._sample_solver)
        flow_shift = resolve_wan_flow_shift_from_sampling(sampling, self.od_config)
        req_scheduler = build_wan_scheduler(sample_solver, flow_shift)
        req_scheduler.set_timesteps(num_steps, device=device)
        timesteps = req_scheduler.timesteps
        self._num_timesteps = len(timesteps)

        generator = sampling.generator
        request_latents = getattr(sampling, "latents", None)
        latent_condition = None
        first_frame_mask = None
        if self.expand_timesteps and image is not None:
            num_channels_latents = self.transformer_config.out_channels
            batch_size = prompt_embeds.shape[0]
            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=request_latents,
            )
            latent_condition, first_frame_mask = self._prepare_step_image_condition(
                image,
                height=height,
                width=width,
                num_frames=num_frames,
                batch_size=batch_size,
                device=device,
            )
        else:
            num_channels_latents = self.transformer_config.in_channels
            latents = self.prepare_latents(
                batch_size=prompt_embeds.shape[0],
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=request_latents,
            )

        state.prompt_embeds = prompt_embeds
        state.negative_prompt_embeds = negative_prompt_embeds
        state.latents = latents
        state.timesteps = timesteps
        state.step_index = sampling.step_index or 0
        state.scheduler = req_scheduler
        state.do_true_cfg = do_classifier_free_guidance and negative_prompt_embeds is not None
        state.extra.update(
            {
                "attention_kwargs": {},
                "boundary_timestep": boundary_timestep,
                "dtype": dtype,
                "first_frame_mask": first_frame_mask,
                "guidance_high": guidance_high,
                "guidance_low": guidance_low,
                "height": height,
                "latent_condition": latent_condition,
                "num_frames": num_frames,
                "output_type": output_type,
                "width": width,
            }
        )
        return state

    @staticmethod
    def _require_step_tensor(state: StepRequestState, field_name: str) -> torch.Tensor:
        value = getattr(state, field_name)
        if value is None:
            raise ValueError(f"{field_name} is not initialized on request {state.request_id}.")
        return value

    @staticmethod
    def _gather_state_tensors(states: Sequence[StepRequestState], field_name: str) -> torch.Tensor | None:
        values = [getattr(state, field_name) for state in states]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Mixed {field_name} presence in one Wan step batch is not supported.")
        return torch.cat(cast(list[torch.Tensor], values), dim=0)

    @staticmethod
    def _gather_state_extra_tensors(states: Sequence[StepRequestState], field_name: str) -> torch.Tensor | None:
        values = [state.extra.get(field_name) for state in states]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Mixed {field_name} presence in one Wan step batch is not supported.")
        return torch.cat(cast(list[torch.Tensor], values), dim=0)

    @staticmethod
    def _gather_state_timesteps(states: Sequence[StepRequestState]) -> torch.Tensor:
        timestep_values: list[torch.Tensor] = []
        for state in states:
            timestep = state.current_timestep
            if timestep is None:
                raise ValueError(f"current_timestep is not initialized on request {state.request_id}.")
            if not torch.is_tensor(timestep):
                raise ValueError("Wan step batching expects tensor timesteps.")
            latents = Wan22Pipeline._require_step_tensor(state, "latents")
            timestep = timestep.to(device=latents.device)
            if timestep.ndim == 0:
                timestep = timestep.expand(latents.shape[0])
            elif timestep.shape[0] != latents.shape[0]:
                raise ValueError(
                    f"timestep rows for request {state.request_id} do not match latent rows: "
                    f"{timestep.shape[0]} vs {latents.shape[0]}."
                )
            timestep_values.append(timestep)
        return torch.cat(timestep_values, dim=0)

    def _build_wan_step_model_input(
        self,
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        dtype: torch.dtype,
        latent_condition: torch.Tensor | None,
        first_frame_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.expand_timesteps and latent_condition is not None and first_frame_mask is not None:
            latent_model_input = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
            latent_model_input = latent_model_input.to(dtype)

            patch_size = self.transformer_config.patch_size
            patch_height = latents.shape[3] // patch_size[1]
            patch_width = latents.shape[4] // patch_size[2]
            patch_mask = first_frame_mask[:, :, :, :: patch_size[1], :: patch_size[2]]
            patch_mask = patch_mask[:, :, :, :patch_height, :patch_width]
            timestep = (patch_mask[:, 0] * timesteps.reshape(-1, 1, 1, 1)).flatten(1)
            return latent_model_input, timestep

        return latents.to(dtype), timesteps

    def _denoise_step_for_states(self, states: Sequence[StepRequestState]) -> torch.Tensor:
        first_state = states[0]
        first_timestep = first_state.current_timestep
        if first_timestep is None or not torch.is_tensor(first_timestep):
            raise ValueError(f"current_timestep is not initialized on request {first_state.request_id}.")
        current_model, current_guidance_scale = self._select_step_model_and_guidance(
            first_timestep,
            boundary_timestep=first_state.extra.get("boundary_timestep"),
            guidance_low=first_state.extra["guidance_low"],
            guidance_high=first_state.extra["guidance_high"],
        )
        dtype = cast(torch.dtype, first_state.extra["dtype"])
        latents = self._gather_state_tensors(states, "latents")
        prompt_embeds = self._gather_state_tensors(states, "prompt_embeds")
        negative_prompt_embeds = self._gather_state_tensors(states, "negative_prompt_embeds")
        if latents is None or prompt_embeds is None:
            raise ValueError("latents and prompt_embeds must be initialized before Wan denoise_step.")
        timesteps = self._gather_state_timesteps(states)
        latent_condition = self._gather_state_extra_tensors(states, "latent_condition")
        first_frame_mask = self._gather_state_extra_tensors(states, "first_frame_mask")
        latent_model_input, timestep = self._build_wan_step_model_input(
            latents=latents,
            timesteps=timesteps,
            dtype=dtype,
            latent_condition=latent_condition,
            first_frame_mask=first_frame_mask,
        )

        do_true_cfg = current_guidance_scale > 1.0 and negative_prompt_embeds is not None
        attention_kwargs = first_state.extra.get("attention_kwargs") or {}
        positive_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "attention_kwargs": attention_kwargs,
            "return_dict": False,
            "current_model": current_model,
        }
        negative_kwargs = (
            {
                "hidden_states": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": negative_prompt_embeds,
                "attention_kwargs": attention_kwargs,
                "return_dict": False,
                "current_model": current_model,
            }
            if do_true_cfg
            else None
        )
        return self.predict_noise_maybe_with_cfg(
            do_true_cfg=do_true_cfg,
            true_cfg_scale=current_guidance_scale,
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
        )

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        del input_batch, kwargs
        if getattr(self, "interrupt", False):
            return None
        if states is None:
            raise ValueError("Wan step execution requires request states.")
        states = list(states)
        if not states:
            raise ValueError("Cannot run Wan denoise_step with no states.")

        groups: dict[tuple[int, float, bool], list[StepRequestState]] = {}
        for state in states:
            timestep = state.current_timestep
            if timestep is None or not torch.is_tensor(timestep):
                raise ValueError(f"current_timestep is not initialized on request {state.request_id}.")
            current_model, current_guidance_scale = self._select_step_model_and_guidance(
                timestep,
                boundary_timestep=state.extra.get("boundary_timestep"),
                guidance_low=state.extra["guidance_low"],
                guidance_high=state.extra["guidance_high"],
            )
            do_true_cfg = current_guidance_scale > 1.0 and state.negative_prompt_embeds is not None
            groups.setdefault((id(current_model), float(current_guidance_scale), do_true_cfg), []).append(state)

        outputs_by_request: dict[str, torch.Tensor] = {}
        for grouped_states in groups.values():
            group_output = self._denoise_step_for_states(grouped_states)
            if group_output is None:
                return None
            row_offset = 0
            for state in grouped_states:
                row_count = self._require_step_tensor(state, "latents").shape[0]
                outputs_by_request[state.request_id] = group_output[row_offset : row_offset + row_count]
                row_offset += row_count

        return torch.cat([outputs_by_request[state.request_id] for state in states], dim=0)

    def step_scheduler(
        self,
        state: StepRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        del kwargs
        t = state.current_timestep
        if t is None:
            raise ValueError(f"current_timestep is not initialized on request {state.request_id}.")
        _, current_guidance_scale = self._select_step_model_and_guidance(
            t,
            boundary_timestep=state.extra.get("boundary_timestep"),
            guidance_low=state.extra["guidance_low"],
            guidance_high=state.extra["guidance_high"],
        )
        do_true_cfg = current_guidance_scale > 1.0 and state.negative_prompt_embeds is not None
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            t,
            state.latents,
            do_true_cfg,
            per_request_scheduler=state.scheduler,
        )
        state.step_index += 1

    def post_decode(
        self,
        state: StepRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        del kwargs
        self._current_timestep = None
        latents = self._require_step_tensor(state, "latents")
        latent_condition = state.extra.get("latent_condition")
        first_frame_mask = state.extra.get("first_frame_mask")
        if self.expand_timesteps and latent_condition is not None and first_frame_mask is not None:
            latents = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()

        output_type = state.extra.get("output_type") or state.sampling.output_type or "np"
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]
        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_low: float,
        guidance_high: float,
        boundary_timestep: float | None,
        dtype: torch.dtype,
        attention_kwargs: dict[str, Any],
        latent_condition: torch.Tensor | None = None,
        first_frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | AsyncLatents:
        if attention_kwargs is None:
            attention_kwargs = {}
        with self.progress_bar(total=len(timesteps)) as pbar:
            for step_idx, t in enumerate(timesteps):
                self._current_timestep = t
                self.record_denoise_step(step_idx, t)

                # Select model based on timestep and boundary_ratio
                # High noise stage (t >= boundary_timestep): use transformer
                # Low noise stage (t < boundary_timestep): use transformer_2
                if boundary_timestep is not None and t < boundary_timestep:
                    # Low noise stage - always use guidance_high for this stage
                    current_guidance_scale = guidance_high
                    if self.transformer_2 is not None:
                        current_model = self.transformer_2
                    elif self.transformer is not None:
                        # Fallback to transformer if transformer_2 not loaded
                        current_model = self.transformer
                    else:
                        raise RuntimeError("No transformer available for low-noise stage")
                else:
                    # High noise stage - always use guidance_low for this stage
                    current_guidance_scale = guidance_low
                    if self.transformer is not None:
                        current_model = self.transformer
                    elif self.transformer_2 is not None:
                        # Fallback to transformer_2 if transformer not loaded
                        current_model = self.transformer_2
                    else:
                        raise RuntimeError("No transformer available for high-noise stage")

                if self.expand_timesteps and latent_condition is not None:
                    # I2V mode: blend condition with latents using mask
                    latent_model_input = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(dtype)

                    # Expand timesteps per patch - use floor division to match patch embedding
                    patch_size = self.transformer_config.patch_size
                    patch_height = latents.shape[3] // patch_size[1]
                    patch_width = latents.shape[4] // patch_size[2]

                    # Create mask at patch resolution (same as hidden states sequence length)
                    patch_mask = first_frame_mask[:, :, :, :: patch_size[1], :: patch_size[2]]
                    patch_mask = patch_mask[:, :, :, :patch_height, :patch_width]  # Ensure correct dimensions
                    temp_ts = (patch_mask[0][0] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    # T2V mode: standard forward
                    latent_model_input = latents.to(dtype)
                    timestep = t.expand(latents.shape[0])

                do_true_cfg = current_guidance_scale > 1.0 and negative_prompt_embeds is not None
                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                    "current_model": current_model,
                }
                if do_true_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                        "current_model": current_model,
                    }
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=current_guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                pbar.update()

        return latents

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        sampling_params_list = req.sampling_params_list
        common = sampling_params_list[0]
        prompt_texts = [prompt if isinstance(prompt, str) else (prompt.get("prompt") or "") for prompt in req.prompts]
        negative_prompts = [
            None if isinstance(prompt, str) else prompt.get("negative_prompt") for prompt in req.prompts
        ]
        prompt_fields = DiffusionRequestBatch.collate_prompt_field_map(
            req.prompts,
            {
                "prompt_embeds": None,
                "negative_prompt_embeds": None,
            },
        )
        prompt_embeds = prompt_fields["prompt_embeds"]
        negative_prompt_embeds = prompt_fields["negative_prompt_embeds"]
        prompt: list[str] | None = prompt_texts if prompt_embeds is None else None
        negative_prompt: list[str] | None = None
        if negative_prompt_embeds is None and any(value is not None for value in negative_prompts):
            negative_prompt = [value or "" for value in negative_prompts]

        if prompt is not None and not all(prompt):
            raise ValueError("Prompt is required for Wan2.2 generation when prompt_embeds are not provided.")

        height = common.height or 480
        width = common.width or 832
        num_frames = common.num_frames or 81

        # Ensure dimensions are compatible with VAE and patch size
        # For expand_timesteps mode, we need latent dims to be even (divisible by patch_size)
        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]  # 16*2=32 for TI2V, 8*2=16 for I2V
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        num_steps = 40 if common.num_inference_steps is None else common.num_inference_steps

        output_type = common.output_type or "np"
        num_outputs_per_prompt = common.num_outputs_per_prompt or 1
        attention_kwargs: dict | None = None

        guidance_low, guidance_high = resolve_wan_guidance_scales(common, default_guidance_scale=4.0)

        # record guidance for properties
        self._guidance_scale = guidance_low
        self._guidance_scale_2 = guidance_high

        # Prefer engine-configured boundary_ratio, but allow per-request fallback.
        boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else common.boundary_ratio

        if boundary_ratio is None:
            boundary_ratio = 0.875
            logger.warning("boundary_ratio is required for T2V generation. using default value 0.875")

        # validate shapes
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale_2=guidance_high if boundary_ratio is not None else None,
            boundary_ratio=boundary_ratio,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        # Get dtype from whichever transformer is loaded
        if self.transformer is not None:
            dtype = self.transformer.dtype
        elif self.transformer_2 is not None:
            dtype = self.transformer_2.dtype
        else:
            # Fallback to text_encoder dtype if no transformer loaded
            dtype = self.text_encoder.dtype

        generator = req.collate_request_generators(num_outputs_per_prompt, None)
        request_latents = req.collate_request_tensors("latents", None)

        if DEBUG_PERF:
            # Sync GPU before timing to ensure accurate measurements
            current_omni_platform.synchronize()
            _t_pipeline_start = time.perf_counter()
            _t_text_enc_start = _t_pipeline_start
        do_classifier_free_guidance = guidance_low > 1.0 or guidance_high > 1.0
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=num_outputs_per_prompt,
                max_sequence_length=common.max_sequence_length or 512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            prompt_embeds = prompt_embeds.repeat_interleave(num_outputs_per_prompt, dim=0)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_outputs_per_prompt, dim=0)
            elif do_classifier_free_guidance:
                _, negative_prompt_embeds = self.encode_prompt(
                    prompt=[""] * req.num_reqs,
                    negative_prompt=negative_prompt,
                    do_classifier_free_guidance=True,
                    num_videos_per_prompt=num_outputs_per_prompt,
                    max_sequence_length=common.max_sequence_length or 512,
                    device=device,
                    dtype=dtype,
                )

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_text_enc_ms = (time.perf_counter() - _t_text_enc_start) * 1000

        first_request = req.requests[0]
        sample_solver = resolve_wan_sample_solver(first_request, default=self._sample_solver)
        flow_shift = resolve_wan_flow_shift(first_request, self.od_config)
        if sample_solver != self._sample_solver or abs(flow_shift - self._flow_shift) > 1e-6:
            self.scheduler = build_wan_scheduler(sample_solver, flow_shift)
            self._sample_solver = sample_solver
            self._flow_shift = flow_shift

        # Timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        boundary_timestep = None
        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.config.num_train_timesteps

        if DEBUG_PERF:
            _t_latent_prep_start = time.perf_counter()
        images: list[PIL.Image.Image | torch.Tensor | None] = []
        for request_prompt in req.prompts:
            multi_modal_data = request_prompt.get("multi_modal_data", {}) if not isinstance(request_prompt, str) else {}
            raw_image = multi_modal_data.get("image")
            if isinstance(raw_image, list):
                if len(raw_image) > 1:
                    logger.warning("Received multiple images for one Wan request; using only the first image.")
                raw_image = raw_image[0] if raw_image else None
            if isinstance(raw_image, str):
                raw_image = PIL.Image.open(raw_image)
            images.append(cast(PIL.Image.Image | torch.Tensor | None, raw_image))

        latent_condition = None
        first_frame_mask = None

        if self.expand_timesteps and any(image is not None for image in images):
            if not all(image is not None for image in images):
                raise ValueError("Cannot batch Wan requests with a mix of provided and missing image conditions.")
            # I2V mode: encode image and prepare condition
            from diffusers.video_processor import VideoProcessor

            video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

            image_tensors = []
            for image in images:
                assert image is not None
                if isinstance(image, PIL.Image.Image):
                    image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
                    image_tensor = video_processor.preprocess(image, height=height, width=width)
                else:
                    image_tensor = image.unsqueeze(0) if image.ndim == 3 else image
                image_tensors.append(image_tensor)
            image_tensor = DiffusionRequestBatch.collate_tensors(image_tensors, "image condition", None)
            assert image_tensor is not None
            image_tensor = image_tensor.repeat_interleave(num_outputs_per_prompt, dim=0)

            # Use out_channels for noise latents (not in_channels which includes condition)
            num_channels_latents = self.transformer_config.out_channels
            batch_size = prompt_embeds.shape[0]

            # Prepare noise latents
            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=request_latents,
            )

            # Encode image condition
            num_latent_frames = latents.shape[2]
            latent_height = latents.shape[3]
            latent_width = latents.shape[4]

            image_tensor = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]
            image_tensor = image_tensor.to(device=device, dtype=self.vae.dtype)
            latent_condition = retrieve_latents(self.vae.encode(image_tensor), sample_mode="argmax")

            # Normalize condition latents
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent_condition.device, latent_condition.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latent_condition.device, latent_condition.dtype
            )
            latent_condition = (latent_condition - latents_mean) * latents_std
            latent_condition = latent_condition.to(torch.float32)

            # Create mask: 0 for first frame (condition), 1 for rest (to denoise)
            first_frame_mask = torch.ones(
                batch_size, 1, num_latent_frames, latent_height, latent_width, dtype=torch.float32, device=device
            )
            first_frame_mask[:, :, 0] = 0
        else:
            # T2V mode: standard latent preparation
            num_channels_latents = self.transformer_config.in_channels
            latents = self.prepare_latents(
                batch_size=prompt_embeds.shape[0],
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=request_latents,
            )
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_latent_prep_ms = (time.perf_counter() - _t_latent_prep_start) * 1000

        if attention_kwargs is None:
            attention_kwargs = {}

        if DEBUG_PERF:
            _t_denoise_start = time.perf_counter()
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            boundary_timestep=boundary_timestep,
            dtype=dtype,
            attention_kwargs=attention_kwargs,
            latent_condition=latent_condition,
            first_frame_mask=first_frame_mask,
        )

        # Wan2.2 is prone to out of memory errors when predicting large videos
        # so we empty the cache here to avoid OOM before vae decoding.
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_denoise_ms = (time.perf_counter() - _t_denoise_start) * 1000

        # For I2V mode: blend final latents with condition
        if self.expand_timesteps and latent_condition is not None:
            latents = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents

        if DEBUG_PERF:
            _t_decode_start = time.perf_counter()
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_decode_ms = (time.perf_counter() - _t_decode_start) * 1000
            _t_pipeline_wall_ms = (time.perf_counter() - _t_pipeline_start) * 1000
            _t_stages_sum = _t_text_enc_ms + _t_latent_prep_ms + _t_denoise_ms + _t_decode_ms

            if _is_rank_zero():
                logger.info(
                    "Pipeline stage timing summary: "
                    "TextEncoding=%.2f ms, LatentPreparation=%.2f ms, "
                    "Denoising=%.2f ms (%d steps), Decoding=%.2f ms, "
                    "StagesSum=%.2f ms, PipelineWall=%.2f ms, Unaccounted=%.2f ms",
                    _t_text_enc_ms,
                    _t_latent_prep_ms,
                    _t_denoise_ms,
                    len(timesteps),
                    _t_decode_ms,
                    _t_stages_sum,
                    _t_pipeline_wall_ms,
                    _t_pipeline_wall_ms - _t_stages_sum,
                )

        return split_diffusion_output_by_request(
            DiffusionOutput(
                output=output,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            ),
            req,
            num_outputs_per_prompt=num_outputs_per_prompt,
        )

    def predict_noise(
        self,
        current_model: nn.Module | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Forward pass through transformer to predict noise.

        Args:
            current_model: The transformer model to use (transformer or transformer_2)
            **kwargs: Arguments to pass to the transformer

        Returns:
            Predicted noise tensor or IntermediateTensors on non-last PP stages.
        """
        if current_model is None:
            current_model = self.transformer
        result = current_model(**kwargs)
        return result if isinstance(result, IntermediateTensors) else result[0]

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} does not match batch size {batch_size}.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        guidance_scale_2=None,
        boundary_ratio=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and "
                f"`negative_prompt_embeds`: {negative_prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when `boundary_ratio` is set.")


# ---------------------------------------------------------------------------
# DMD2-distilled variant
# ---------------------------------------------------------------------------


class WanT2VDMD2Pipeline(DMD2PipelineMixin, Wan22Pipeline):
    """Wan 2.x T2V pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
