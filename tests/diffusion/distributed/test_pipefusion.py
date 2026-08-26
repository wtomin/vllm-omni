# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU unit tests for PipeFusion.

These tests intentionally avoid real distributed process groups. GPU-backed
communication correctness for the PP labels PipeFusion uses lives in
test_pipeline_parallel.py.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

import vllm_omni.diffusion.distributed.pipefusion.offload as pf_offload
import vllm_omni.diffusion.distributed.pipefusion.pipefusion as pf_pipeline
import vllm_omni.diffusion.distributed.pipefusion.pipefusion_conv as pf_conv
import vllm_omni.diffusion.distributed.pipefusion.pipefusion_runtime as pf_runtime
import vllm_omni.diffusion.distributed.pipefusion.pipefusion_scheduler as pf_scheduler
import vllm_omni.diffusion.distributed.pipefusion.pipefusion_transformer as pf_transformer
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.diffusion_kv.manager import DiffusionKVCacheManager
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata, DiffusionKVSequenceMetadata
from vllm_omni.diffusion.diffusion_kv.model_runner_backend import DiffusionKVModelRunnerBackend
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipefusion.dense_kv_store import (
    PipeFusionDenseKVStore,
    PipeFusionPatchLayout,
)
from vllm_omni.diffusion.distributed.pipefusion.offload import (
    PipeFusionKVOffloadManager,
    PipeFusionKVResidency,
)
from vllm_omni.diffusion.distributed.pipefusion.pipefusion import PipeFusionPipelineMixin
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_conv import PipeFusionConvMixin
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_runtime import (
    PipeFusionRuntime,
    attach_pipefusion_kv_requests,
    build_pipefusion_kv_requests,
    get_pipefusion_token_capacity,
)
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_scheduler import PipeFusionSchedulerMixin
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_transformer import (
    PipeFusionRotaryEmbeddingMixin,
    PipeFusionSelfAttentionMixin,
    PipeFusionTransformerMixin,
)
from vllm_omni.diffusion.distributed.pipeline_parallel import PipelineParallelMixin

pytestmark = [pytest.mark.core_model, pytest.mark.parallel, pytest.mark.cpu]


class FakePPGroup:
    def __init__(self):
        self.device = torch.device("cpu")
        self.reset_calls = 0
        self.dtype = None

    def reset_buffer(self):
        self.reset_calls += 1

    def set_config(self, dtype: torch.dtype):
        self.dtype = dtype


@pytest.fixture(autouse=True)
def reset_pipefusion_runtime(monkeypatch):
    monkeypatch.setattr(pf_runtime, "_PF_RUNTIME", None)


def _set_patch_idx(runtime: PipeFusionRuntime, patch_idx: int) -> None:
    runtime.pipeline_patch_idx = patch_idx
    runtime.patch_idx_tensor = torch.tensor(patch_idx, dtype=torch.int64)


class TestPipeFusionRuntime:
    def test_build_managed_kv_rows_uses_contiguous_branch_rows(self):
        assert get_pipefusion_token_capacity((5, 8, 6), (1, 2, 2)) == 60

        requests = build_pipefusion_kv_requests(
            "req-a",
            token_capacity=60,
            sequence_count=2,
            cfg_branches=("inputs", "inputs_uncond"),
        )

        assert [request.sequence_id for request in requests] == [0, 1, 2, 3]
        assert [(request.request_id, request.logical_sequence_id, request.cache_branch) for request in requests] == [
            ("req-a:pf:0:inputs", 0, "inputs"),
            ("req-a:pf:0:inputs_uncond", 0, "inputs_uncond"),
            ("req-a:pf:1:inputs", 1, "inputs"),
            ("req-a:pf:1:inputs_uncond", 1, "inputs_uncond"),
        ]

    def test_default_local_cfg_reserves_unconditional_row_without_negative_prompt(self):
        request = SimpleNamespace(
            request_id="req-a",
            prompt={"prompt": "a cat"},
            sampling_params=SimpleNamespace(height=64, width=64, num_frames=5),
            diffusion_kv_requests=(),
        )
        od_config = SimpleNamespace(
            dtype=torch.bfloat16,
            model_config={"vae_scale_factor_temporal": 4, "vae_scale_factor_spatial": 8},
            tf_model_config={
                "patch_size": (1, 2, 2),
                "num_layers": 2,
                "num_attention_heads": 2,
                "attention_head_dim": 4,
            },
            diffusion_kv_max_rows_per_request=2,
            parallel_config=SimpleNamespace(
                enable_pipefusion=True,
                cfg_parallel_size=1,
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
            ),
        )

        attach_pipefusion_kv_requests(request, od_config)

        assert [row.cache_branch for row in request.diffusion_kv_requests] == ["inputs", "inputs_uncond"]
        assert all(row.estimated_bytes > 0 for row in request.diffusion_kv_requests)

    def test_set_run_config_validates_warmup_steps(self):
        runtime = PipeFusionRuntime()

        with pytest.raises(ValueError, match="warmup_steps must be a positive integer"):
            runtime.set_run_config(warmup_steps=0)

        with pytest.raises(ValueError, match="warmup_steps must be a positive integer"):
            runtime.set_run_config(warmup_steps=1.5)

        runtime.set_run_config(warmup_steps=3)
        assert runtime.warmup_steps == 3

    def test_set_run_config_validates_split_dim(self):
        runtime = PipeFusionRuntime()

        with pytest.raises(ValueError, match="Invalid PipeFusion split_dim"):
            runtime.set_run_config(split_dim="width")

        runtime.set_run_config(split_dim="temporal")
        assert runtime.split_dim == "temporal"

    def test_set_cache_key_validates_cfg_branch(self):
        runtime = PipeFusionRuntime()

        runtime.set_cache_key("inputs_uncond")
        assert runtime.cache_key == "inputs_uncond"
        assert runtime.cache_identity == (None, 0, "inputs_uncond")

        with pytest.raises(ValueError, match="Invalid PipeFusion cache key"):
            runtime.set_cache_key("bad-key")

    def test_request_context_validates_request_and_sequence(self):
        runtime = PipeFusionRuntime()

        runtime.set_request_context("req-a", 3)
        assert runtime.cache_identity == ("req-a", 3, "inputs")

        runtime.clear_request_context()
        assert runtime.cache_identity == (None, 0, "inputs")

        with pytest.raises(ValueError, match="request_id must be a non-empty string"):
            runtime.set_request_context("", 0)

        with pytest.raises(ValueError, match="sequence_id must be a non-negative integer"):
            runtime.set_request_context("req-a", -1)

    def test_managed_row_binding_validates_runtime_capacity(self):
        runtime = PipeFusionRuntime()
        runtime.set_request_context("req-a", 0)
        runtime.ppf, runtime.pph, runtime.ppw = 2, 3, 4
        runtime.set_kv_row_binding_resolver(
            lambda request_id, sequence_id, branch: SimpleNamespace(
                row_index=7,
                max_seq_len=24,
            )
        )
        runtime.validate_kv_row_binding()

        runtime.set_kv_row_binding_resolver(
            lambda request_id, sequence_id, branch: SimpleNamespace(
                row_index=7,
                max_seq_len=23,
            )
        )
        with pytest.raises(RuntimeError, match="runtime requires 24 tokens"):
            runtime.validate_kv_row_binding()
        with pytest.raises(RuntimeError, match="runtime requires 25 tokens"):
            runtime.validate_kv_row_binding(observed_seq_len=25)

        def missing_binding(request_id, sequence_id, branch):
            raise KeyError((request_id, sequence_id, branch))

        runtime.set_kv_row_binding_resolver(missing_binding)
        with pytest.raises(KeyError):
            runtime.validate_kv_row_binding()

    def test_height_split_patch_metadata_and_recv_buffer_reset(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.patch_size = (1, 2, 2)
        runtime.split_dim = "height"
        pp_group = FakePPGroup()

        monkeypatch.setattr(pf_runtime, "get_pipeline_parallel_world_size", lambda: 3)
        monkeypatch.setattr(pf_runtime, "get_pp_group", lambda: pp_group)

        latents = torch.zeros(1, 4, 2, 10, 6)
        runtime.set_input_parameters(latents, torch.float32)

        assert runtime.num_pipeline_patch == 3
        assert runtime.latent_split_dim == -2
        assert runtime.pp_patches_post_height == [1, 1, 3]
        assert runtime.pp_patches_height == [2, 2, 6]
        assert runtime.pp_patches_start_end_idx == [(0, 2), (2, 4), (4, 10)]
        assert runtime.pp_patches_token_num == [6, 6, 18]
        assert runtime.pp_patches_token_start_end_idx == [(0, 6), (6, 12), (12, 30)]
        assert pp_group.reset_calls == 1
        assert pp_group.dtype is torch.float32
        assert runtime.patch_idx_tensor.device.type == "cpu"

    def test_temporal_split_patch_metadata(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.patch_size = (1, 2, 2)
        runtime.split_dim = "temporal"

        monkeypatch.setattr(pf_runtime, "get_pipeline_parallel_world_size", lambda: 2)

        latents = torch.zeros(1, 4, 5, 6, 4)
        runtime._calc_patches_metadata(latents)

        assert runtime.latent_split_dim == -3
        assert runtime.pp_patches_post_frames == [2, 3]
        assert runtime.pp_patches_height == [2, 3]
        assert runtime.pp_patches_start_end_idx == [(0, 2), (2, 5)]
        assert runtime.pp_patches_token_num == [12, 18]
        assert runtime.pp_patches_token_start_end_idx == [(0, 12), (12, 30)]

    def test_next_patch_wraps_only_in_patch_mode(self):
        runtime = PipeFusionRuntime()
        runtime.num_pipeline_patch = 3

        runtime.set_patched_mode(True)
        assert runtime.pipeline_patch_idx == 0

        runtime.next_patch()
        assert runtime.pipeline_patch_idx == 1
        assert runtime.patch_idx_tensor.item() == 1

        runtime.next_patch()
        assert runtime.pipeline_patch_idx == 2

        runtime.next_patch()
        assert runtime.pipeline_patch_idx == 0

        runtime.set_patched_mode(False)
        runtime.pipeline_patch_idx = 2
        runtime.next_patch()
        assert runtime.pipeline_patch_idx == 0


class TestPipeFusionDenseKVStore:
    def test_cpu_offload_state_and_byte_accounting(self):
        manager = PipeFusionKVOffloadManager(enabled=True, pin_memory=False)
        identity = ("req-a", 0, "inputs", "blocks.0.attn1")
        key = torch.arange(4, dtype=torch.float32)
        value = key + 10

        manager.put(identity, key, value)
        assert manager.residency(identity) is PipeFusionKVResidency.GPU
        assert manager.gpu_bytes == 32

        assert manager.offload(identity) is True
        assert manager.residency(identity) is PipeFusionKVResidency.CPU
        assert manager.gpu_bytes == 0
        assert manager.cpu_bytes == 32

        restored_key, restored_value = manager.get(identity)
        assert manager.residency(identity) is PipeFusionKVResidency.GPU
        torch.testing.assert_close(restored_key, key)
        torch.testing.assert_close(restored_value, value)

    def test_cpu_offload_budget_is_enforced_before_copy(self, monkeypatch):
        monkeypatch.setattr(pf_offload, "PIPEFUSION_KV_MAX_CPU_BYTES", 31)
        manager = PipeFusionKVOffloadManager(enabled=True, pin_memory=False)
        identity = ("req-a", 0, "inputs", "blocks.0.attn1")
        manager.put(identity, torch.ones(4), torch.ones(4))

        with pytest.raises(RuntimeError, match="CPU offload budget exceeded"):
            manager.offload(identity)

        assert manager.residency(identity) is PipeFusionKVResidency.GPU

    def test_interleaved_requests_keep_patch_updates_isolated_with_offload(self):
        manager = PipeFusionKVOffloadManager(enabled=True, pin_memory=False)
        store = PipeFusionDenseKVStore(manager)
        layout = PipeFusionPatchLayout(
            split_dim="temporal",
            ppf=2,
            pph=1,
            ppw=1,
            num_pipeline_patch=2,
        )
        layer_id = "blocks.0.attn1"
        req_a = ("req-a", 0, "inputs")
        req_b = ("req-b", 0, "inputs")
        store.put_full(req_a, layer_id, torch.tensor([[[[1.0]], [[2.0]]]]), torch.tensor([[[[11.0]], [[12.0]]]]))
        store.put_full(req_b, layer_id, torch.tensor([[[[3.0]], [[4.0]]]]), torch.tensor([[[[13.0]], [[14.0]]]]))
        assert store.offload(req_a, layer_id)
        assert store.offload(req_b, layer_id)

        key_a, value_a = store.update_patch(
            req_a,
            layer_id,
            0,
            torch.tensor([[[[100.0]]]]),
            torch.tensor([[[[110.0]]]]),
            layout=layout,
        )
        key_b, value_b = store.update_patch(
            req_b,
            layer_id,
            1,
            torch.tensor([[[[200.0]]]]),
            torch.tensor([[[[210.0]]]]),
            layout=layout,
        )

        torch.testing.assert_close(key_a.flatten(), torch.tensor([100.0, 2.0]))
        torch.testing.assert_close(value_a.flatten(), torch.tensor([110.0, 12.0]))
        torch.testing.assert_close(key_b.flatten(), torch.tensor([3.0, 200.0]))
        torch.testing.assert_close(value_b.flatten(), torch.tensor([13.0, 210.0]))

    def test_prefetch_window_restores_following_layer(self):
        manager = PipeFusionKVOffloadManager(enabled=True, pin_memory=False, prefetch_layers=1)
        runtime = PipeFusionRuntime()
        runtime._kv_offload_manager = manager
        runtime.register_kv_layer("blocks.0.attn1")
        runtime.register_kv_layer("blocks.1.attn1")
        next_identity = ("req-a", 0, "inputs", "blocks.1.attn1")
        manager.put(next_identity, torch.ones(1), torch.ones(1))
        manager.offload(next_identity)

        runtime.prefetch_following_layers(("req-a", 0, "inputs"), "blocks.0.attn1")

        assert manager.residency(next_identity) is PipeFusionKVResidency.GPU


class TestPipeFusionManagedRows:
    def test_logical_manager_reserves_and_releases_rows(self):
        manager = DiffusionKVCacheManager(
            None,
            max_model_len=64,
            scheduler_block_size=1,
            hash_block_size=1,
            max_logical_rows=2,
        )
        requests = build_pipefusion_kv_requests(
            "req-a",
            token_capacity=32,
            cfg_branches=("inputs", "inputs_uncond"),
        )

        metadata = manager.reserve_request("req-a", requests)
        assert metadata is not None
        assert [sequence.cache_branch for sequence in metadata.sequences] == ["inputs", "inputs_uncond"]
        assert (
            manager.reserve_request(
                "req-b",
                build_pipefusion_kv_requests("req-b", token_capacity=32),
            )
            is None
        )

        manager.free_request("req-a")
        assert (
            manager.reserve_request(
                "req-b",
                build_pipefusion_kv_requests("req-b", token_capacity=32),
            )
            is not None
        )

    def test_logical_manager_admits_and_releases_estimated_bytes(self):
        manager = DiffusionKVCacheManager(
            None,
            max_model_len=64,
            scheduler_block_size=1,
            hash_block_size=1,
            max_logical_rows=2,
            max_logical_bytes=128,
        )
        request = build_pipefusion_kv_requests(
            "req-a",
            token_capacity=32,
            estimated_bytes_per_row=96,
        )
        assert manager.reserve_request("req-a", request) is not None
        assert (
            manager.reserve_request(
                "req-b",
                build_pipefusion_kv_requests(
                    "req-b",
                    token_capacity=32,
                    estimated_bytes_per_row=64,
                ),
            )
            is None
        )

        manager.free_request("req-a")
        assert (
            manager.reserve_request(
                "req-b",
                build_pipefusion_kv_requests(
                    "req-b",
                    token_capacity=32,
                    estimated_bytes_per_row=64,
                ),
            )
            is not None
        )

    def test_worker_row_binding_isolated_by_cfg_branch(self):
        od_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                enable_pipefusion=True,
            ),
            diffusion_kv_max_rows_per_request=2,
            max_num_seqs=1,
        )
        backend = DiffusionKVModelRunnerBackend(
            vllm_config=SimpleNamespace(),
            od_config=od_config,
            device=torch.device("cpu"),
        )
        metadata = DiffusionKVMetadata(
            request_id="req-a",
            allocation_generation=1,
            sequences=(
                DiffusionKVSequenceMetadata(
                    sequence_id=0,
                    logical_sequence_id=0,
                    cache_branch="inputs",
                    prefix_len=0,
                    target_len=32,
                    seq_len=32,
                    block_ids=(),
                ),
                DiffusionKVSequenceMetadata(
                    sequence_id=1,
                    logical_sequence_id=0,
                    cache_branch="inputs_uncond",
                    prefix_len=0,
                    target_len=32,
                    seq_len=32,
                    block_ids=(),
                ),
            ),
        )

        assert backend.install_diffusion_kv_metadata(metadata) is True
        positive = backend.get_pipefusion_kv_row_binding("req-a", 0, "inputs")
        negative = backend.get_pipefusion_kv_row_binding("req-a", 0, "inputs_uncond")
        assert positive.row_index != negative.row_index
        assert positive.max_seq_len == negative.max_seq_len == 32
        assert backend.remove_diffusion_kv_requests(["req-a"]) == 2
        with pytest.raises(KeyError):
            backend.get_pipefusion_kv_row_binding("req-a", 0, "inputs")


class DummyScheduler(PipeFusionSchedulerMixin):
    _pipefusion_patch_cache_spec = [
        ("model_outputs", "list"),
        ("last_sample", "tensor"),
    ]

    def __init__(self):
        self._init_patch_caches()
        self.model_outputs = [torch.arange(6, dtype=torch.float32).view(1, 1, 1, 6, 1), None]
        self.last_sample = torch.arange(100, 106, dtype=torch.float32).view(1, 1, 1, 6, 1)
        self._step_index = 4
        self.lower_order_nums = 2
        self.this_order = 1
        self.timestep_list = ["warmup"]


def _scheduler_step_impl(self: DummyScheduler, sample: torch.Tensor):
    self.model_outputs[0] = sample + 10
    self.last_sample = sample + 20
    self._step_index += 1
    self.lower_order_nums += 1
    self.this_order += 1
    self.timestep_list.append(f"patch-{pf_scheduler.get_pipefusion_runtime().pipeline_patch_idx}")
    return self.last_sample


class TestPipeFusionSchedulerMixin:
    def test_split_caches_for_patches_splits_list_and_tensor_entries(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.num_pipeline_patch = 2
        monkeypatch.setattr(pf_scheduler, "get_pipefusion_runtime", lambda: runtime)

        scheduler = DummyScheduler()
        scheduler.split_caches_for_patches([2, 4], dim=-2)

        assert set(scheduler._pf_patch_caches) == {"model_outputs", "last_sample"}
        assert scheduler._pf_patch_caches["model_outputs"][0][0].shape[-2] == 2
        assert scheduler._pf_patch_caches["model_outputs"][1][0].shape[-2] == 4
        assert scheduler._pf_patch_caches["model_outputs"][0][1] is None
        assert scheduler._pf_patch_caches["last_sample"][0].shape[-2] == 2
        assert scheduler._pf_patch_caches["last_sample"][1].shape[-2] == 4

    def test_scheduler_step_restores_shared_and_first_patch_only_state(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.num_pipeline_patch = 2
        runtime.set_patched_mode(True)
        monkeypatch.setattr(pf_scheduler, "get_pipefusion_runtime", lambda: runtime)

        scheduler = DummyScheduler()
        scheduler.split_caches_for_patches([2, 4], dim=-2)

        _set_patch_idx(runtime, 0)
        first_patch_sample = torch.ones(1, 1, 1, 2, 1)
        first_result = scheduler._pipefusion_scheduler_step(_scheduler_step_impl, first_patch_sample)

        torch.testing.assert_close(first_result, first_patch_sample + 20)
        assert scheduler._step_index == 4
        assert scheduler.lower_order_nums == 2
        assert scheduler.this_order == 1
        assert scheduler.timestep_list == ["warmup", "patch-0"]
        torch.testing.assert_close(scheduler._pf_patch_caches["last_sample"][0], first_patch_sample + 20)

        _set_patch_idx(runtime, 1)
        second_patch_sample = torch.ones(1, 1, 1, 4, 1) * 2
        second_result = scheduler._pipefusion_scheduler_step(_scheduler_step_impl, second_patch_sample)

        torch.testing.assert_close(second_result, second_patch_sample + 20)
        assert scheduler._step_index == 5
        assert scheduler.lower_order_nums == 3
        assert scheduler.this_order == 2
        assert scheduler.timestep_list == ["warmup", "patch-0"]
        torch.testing.assert_close(scheduler._pf_patch_caches["last_sample"][1], second_patch_sample + 20)

    def test_clear_patch_caches(self):
        scheduler = DummyScheduler()
        scheduler._pf_patch_caches = {"last_sample": [torch.ones(1)]}

        scheduler.clear_patch_caches()

        assert scheduler._pf_patch_caches is None


class DummyRotary(PipeFusionRotaryEmbeddingMixin):
    patch_size = (1, 2, 2)


class DummyAttention(PipeFusionSelfAttentionMixin):
    pass


class DummyTransformer(PipeFusionTransformerMixin):
    def __init__(self):
        self.config = SimpleNamespace(patch_size=(1, 2, 2))

    def _unpatchify(self, hidden_states: torch.Tensor, dims: tuple[int, int, int, int, int]):
        return hidden_states


class TestPipeFusionTransformerHelpers:
    def test_rotary_embedding_slice_height(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.num_pipeline_patch = 2
        runtime.split_dim = "height"
        _set_patch_idx(runtime, 1)
        monkeypatch.setattr(pf_transformer, "get_pipefusion_runtime", lambda: runtime)

        rotary = DummyRotary()
        cos = torch.arange(1 * 16 * 1 * 2, dtype=torch.float32).reshape(1, 16, 1, 2)
        sin = cos + 100

        sliced_cos, sliced_sin = rotary.pipefusion_slice_rotary_emb((cos, sin), 2, 8, 4)

        expected_cos = cos.reshape(2, 4, 2, 2).narrow(1, 2, 2).reshape(1, 8, 1, 2)
        expected_sin = sin.reshape(2, 4, 2, 2).narrow(1, 2, 2).reshape(1, 8, 1, 2)
        torch.testing.assert_close(sliced_cos, expected_cos)
        torch.testing.assert_close(sliced_sin, expected_sin)

    def test_rotary_embedding_slice_temporal(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.num_pipeline_patch = 2
        runtime.split_dim = "temporal"
        _set_patch_idx(runtime, 0)
        monkeypatch.setattr(pf_transformer, "get_pipefusion_runtime", lambda: runtime)

        rotary = DummyRotary()
        cos = torch.arange(1 * 16 * 1 * 2, dtype=torch.float32).reshape(1, 16, 1, 2)
        sin = cos + 100

        sliced_cos, sliced_sin = rotary.pipefusion_slice_rotary_emb((cos, sin), 4, 4, 4)

        expected_cos = cos.reshape(4, 2, 2, 2).narrow(0, 0, 2).reshape(1, 8, 1, 2)
        expected_sin = sin.reshape(4, 2, 2, 2).narrow(0, 0, 2).reshape(1, 8, 1, 2)
        torch.testing.assert_close(sliced_cos, expected_cos)
        torch.testing.assert_close(sliced_sin, expected_sin)

    def test_kv_cache_height_patch_update_uses_5d_slice(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.patch_mode = False
        runtime.cache_key = "inputs"
        runtime.split_dim = "height"
        runtime.num_pipeline_patch = 2
        runtime.ppf = 2
        runtime.pph = 4
        runtime.ppw = 2
        _set_patch_idx(runtime, 1)
        monkeypatch.setattr(pf_transformer, "get_pipefusion_runtime", lambda: runtime)

        attention = DummyAttention()
        attention.pipefusion_reset_cache()
        full_key = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1, 1)
        full_value = full_key + 100
        attention._pipefusion_update_kv_cache(full_key.clone(), full_value.clone())

        runtime.patch_mode = True
        patch_key = torch.full((1, 8, 1, 1), 1000.0)
        patch_value = torch.full((1, 8, 1, 1), 2000.0)
        updated_key, updated_value = attention._pipefusion_update_kv_cache(patch_key, patch_value)

        expected_key = full_key.clone().view(1, 2, 4, 2, 1, 1)
        expected_value = full_value.clone().view(1, 2, 4, 2, 1, 1)
        expected_key.narrow(2, 2, 2).copy_(patch_key.reshape(1, 2, 2, 2, 1, 1))
        expected_value.narrow(2, 2, 2).copy_(patch_value.reshape(1, 2, 2, 2, 1, 1))
        torch.testing.assert_close(updated_key, expected_key.reshape(1, 16, 1, 1))
        torch.testing.assert_close(updated_value, expected_value.reshape(1, 16, 1, 1))

    def test_kv_cache_temporal_patch_update_uses_contiguous_slice(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.patch_mode = False
        runtime.cache_key = "inputs_uncond"
        runtime.split_dim = "temporal"
        runtime.num_pipeline_patch = 2
        runtime.ppf = 4
        runtime.pph = 2
        runtime.ppw = 2
        _set_patch_idx(runtime, 1)
        monkeypatch.setattr(pf_transformer, "get_pipefusion_runtime", lambda: runtime)

        attention = DummyAttention()
        attention.pipefusion_reset_cache()
        full_key = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1, 1)
        full_value = full_key + 100
        attention._pipefusion_update_kv_cache(full_key.clone(), full_value.clone())

        runtime.patch_mode = True
        patch_key = torch.full((1, 8, 1, 1), 3000.0)
        patch_value = torch.full((1, 8, 1, 1), 4000.0)
        updated_key, updated_value = attention._pipefusion_update_kv_cache(patch_key, patch_value)

        expected_key = full_key.clone()
        expected_value = full_value.clone()
        expected_key.narrow(1, 8, 8).copy_(patch_key)
        expected_value.narrow(1, 8, 8).copy_(patch_value)
        torch.testing.assert_close(updated_key, expected_key)
        torch.testing.assert_close(updated_value, expected_value)

    def test_kv_cache_requires_warmup_cache_before_patch_mode(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.patch_mode = True
        runtime.cache_key = "inputs"
        monkeypatch.setattr(pf_transformer, "get_pipefusion_runtime", lambda: runtime)

        attention = DummyAttention()
        attention.pipefusion_reset_cache()

        with pytest.raises(RuntimeError, match="Run at least one warmup step"):
            attention._pipefusion_update_kv_cache(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1))

    def test_kv_cache_isolated_by_request_sequence_and_cfg_branch(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.patch_mode = False
        runtime.split_dim = "temporal"
        runtime.num_pipeline_patch = 2
        runtime.ppf = 2
        runtime.pph = 1
        runtime.ppw = 1
        _set_patch_idx(runtime, 0)
        monkeypatch.setattr(pf_transformer, "get_pipefusion_runtime", lambda: runtime)

        attention = DummyAttention()
        attention.pipefusion_reset_cache()

        runtime.set_request_context("req-a", 0)
        runtime.set_cache_key("inputs")
        req_a_seq0_key = torch.full((1, 2, 1, 1), 10.0)
        req_a_seq0_value = req_a_seq0_key + 100
        attention._pipefusion_update_kv_cache(req_a_seq0_key.clone(), req_a_seq0_value.clone())

        runtime.set_request_context("req-a", 1)
        req_a_seq1_key = torch.full((1, 2, 1, 1), 20.0)
        req_a_seq1_value = req_a_seq1_key + 100
        attention._pipefusion_update_kv_cache(req_a_seq1_key.clone(), req_a_seq1_value.clone())

        runtime.set_request_context("req-a", 0)
        runtime.set_cache_key("inputs_uncond")
        req_a_uncond_key = torch.full((1, 2, 1, 1), 30.0)
        req_a_uncond_value = req_a_uncond_key + 100
        attention._pipefusion_update_kv_cache(req_a_uncond_key.clone(), req_a_uncond_value.clone())

        assert set(attention._kv_caches) == {
            ("req-a", 0, "inputs"),
            ("req-a", 1, "inputs"),
            ("req-a", 0, "inputs_uncond"),
        }

        runtime.patch_mode = True
        runtime.set_request_context("req-b", 0)
        runtime.set_cache_key("inputs")
        with pytest.raises(RuntimeError, match=r"\('req-b', 0, 'inputs'\)"):
            attention._pipefusion_update_kv_cache(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1))

        runtime.set_request_context("req-a", 1)
        updated_key, updated_value = attention._pipefusion_update_kv_cache(
            torch.full((1, 1, 1, 1), 200.0),
            torch.full((1, 1, 1, 1), 300.0),
        )
        expected_key = req_a_seq1_key.clone()
        expected_value = req_a_seq1_value.clone()
        expected_key.narrow(1, 0, 1).fill_(200.0)
        expected_value.narrow(1, 0, 1).fill_(300.0)
        torch.testing.assert_close(updated_key, expected_key)
        torch.testing.assert_close(updated_value, expected_value)

    def test_kv_cache_reset_can_target_one_request_sequence(self, monkeypatch):
        runtime = PipeFusionRuntime()
        monkeypatch.setattr(pf_transformer, "get_pipefusion_runtime", lambda: runtime)

        attention = DummyAttention()
        attention.pipefusion_reset_cache()
        for request_id, sequence_id, branch in [
            ("req-a", 0, "inputs"),
            ("req-a", 0, "inputs_uncond"),
            ("req-a", 1, "inputs"),
            ("req-b", 0, "inputs"),
        ]:
            runtime.set_request_context(request_id, sequence_id)
            runtime.set_cache_key(branch)
            attention._pipefusion_update_kv_cache(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1))

        attention.pipefusion_reset_cache("req-a", 0)

        assert set(attention._kv_caches) == {
            ("req-a", 1, "inputs"),
            ("req-b", 0, "inputs"),
        }

    def test_unpatchify_uses_patch_local_height(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.split_dim = "height"
        runtime.num_pipeline_patch = 2
        _set_patch_idx(runtime, 1)
        monkeypatch.setattr(pf_transformer, "get_pipefusion_runtime", lambda: runtime)

        transformer = DummyTransformer()
        hidden_states = torch.arange(1 * 8 * 4, dtype=torch.float32).reshape(1, 8, 4)
        output = transformer.pipefusion_unpatchify(hidden_states, (1, 4, 2, 8, 4))

        assert output.shape == (1, 1, 2, 4, 4)

    def test_unpatchify_uses_patch_local_frames(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.split_dim = "temporal"
        runtime.num_pipeline_patch = 2
        _set_patch_idx(runtime, 0)
        monkeypatch.setattr(pf_transformer, "get_pipefusion_runtime", lambda: runtime)

        transformer = DummyTransformer()
        hidden_states = torch.arange(1 * 8 * 4, dtype=torch.float32).reshape(1, 8, 4)
        output = transformer.pipefusion_unpatchify(hidden_states, (1, 4, 4, 4, 4))

        assert output.shape == (1, 1, 2, 4, 4)


class DummyConv(PipeFusionConvMixin):
    def __init__(self):
        self.kernel_size = (1, 3, 1)
        self.stride = (1, 1, 1)
        self.padding = (0, 1, 0)
        self.dilation = (1, 1, 1)
        self.groups = 1
        self.weight = torch.ones(1, 1, 1, 3, 1)
        self.bias = None

    def orig_forward(self, x: torch.Tensor, dims=None) -> torch.Tensor:
        return F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class TestPipeFusionConvMixin:
    def test_conv3d_enabled_only_for_overlapping_patch_mode(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.patch_mode = True
        runtime.num_pipeline_patch = 2
        monkeypatch.setattr(pf_conv, "get_pipefusion_runtime", lambda: runtime)

        conv = DummyConv()
        assert conv.pipefusion_conv3d_enabled() is True

        conv.stride = conv.kernel_size
        assert conv.pipefusion_conv3d_enabled() is False

        runtime.patch_mode = False
        conv.stride = (1, 1, 1)
        assert conv.pipefusion_conv3d_enabled() is False

    def test_conv3d_forward_caches_activations_and_returns_patch_slice(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.patch_mode = True
        runtime.num_pipeline_patch = 2
        runtime.split_dim = "height"
        runtime.pp_patches_start_end_idx = [(0, 2), (2, 6)]
        runtime.pp_patches_post_start_end_idx = [(0, 2), (2, 6)]
        monkeypatch.setattr(pf_conv, "get_pipefusion_runtime", lambda: runtime)

        conv = DummyConv()
        full_input = torch.arange(6, dtype=torch.float32).view(1, 1, 1, 6, 1)

        runtime.pipeline_patch_idx = 0
        _ = conv.pipefusion_conv3d_forward(full_input[:, :, :, :2, :], dims=full_input.shape)

        runtime.pipeline_patch_idx = 1
        patch_output = conv.pipefusion_conv3d_forward(full_input[:, :, :, 2:, :], dims=full_input.shape)
        full_output = conv.orig_forward(full_input)

        torch.testing.assert_close(patch_output, full_output[:, :, :, 2:6, :])
        torch.testing.assert_close(conv.activation_cache, full_input)

    def test_conv3d_forward_supports_temporal_split(self, monkeypatch):
        runtime = PipeFusionRuntime()
        runtime.patch_mode = True
        runtime.num_pipeline_patch = 2
        runtime.split_dim = "temporal"
        runtime.pp_patches_start_end_idx = [(0, 2), (2, 6)]
        runtime.pp_patches_post_start_end_idx = [(0, 2), (2, 6)]
        monkeypatch.setattr(pf_conv, "get_pipefusion_runtime", lambda: runtime)

        conv = DummyConv()
        conv.kernel_size = (3, 1, 1)
        conv.stride = (1, 1, 1)
        conv.padding = (1, 0, 0)
        conv.weight = torch.ones(1, 1, 3, 1, 1)

        full_input = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1, 1)

        runtime.pipeline_patch_idx = 0
        _ = conv.pipefusion_conv3d_forward(full_input[:, :, :2, :, :], dims=full_input.shape)

        runtime.pipeline_patch_idx = 1
        patch_output = conv.pipefusion_conv3d_forward(full_input[:, :, 2:, :, :], dims=full_input.shape)
        full_output = conv.orig_forward(full_input)

        torch.testing.assert_close(patch_output, full_output[:, :, 2:6, :, :])
        torch.testing.assert_close(conv.activation_cache, full_input)


class TestPipeFusionPipelineMixin:
    def test_pipefusion_pipeline_requires_pipeline_parallel_mixin(self):
        with pytest.raises(TypeError, match="inherits PipeFusionPipelineMixin but not PipelineParallelMixin"):

            class _MissingPP(PipeFusionPipelineMixin):
                def prepare_model_kwargs(self, latents, timestep, **extra_kwargs):
                    return {}, None, False, 1.0

    def test_pipefusion_pipeline_requires_mro_before_pipeline_parallel_mixin(self):
        with pytest.raises(TypeError, match="must inherit PipeFusionPipelineMixin before PipelineParallelMixin"):

            class _WrongOrder(PipelineParallelMixin, PipeFusionPipelineMixin, CFGParallelMixin):
                def prepare_model_kwargs(self, latents, timestep, **extra_kwargs):
                    return {}, None, False, 1.0

    def test_configure_pipefusion_run_reads_sampling_params(self, monkeypatch):
        runtime = PipeFusionRuntime()
        monkeypatch.setattr(pf_pipeline, "get_pipefusion_runtime", lambda: runtime)
        req = SimpleNamespace(
            sampling_params=SimpleNamespace(
                pipefusion_warmup_steps=4,
                pipefusion_split_dim="temporal",
            )
        )

        PipeFusionPipelineMixin._configure_pipefusion_run(req)

        assert runtime.warmup_steps == 4
        assert runtime.split_dim == "temporal"


class TestDiffusionParallelConfigPipeFusion:
    def test_pipefusion_disabled_by_default(self):
        config = DiffusionParallelConfig()

        assert config.enable_pipefusion is False

    def test_pipefusion_requires_pipeline_parallelism(self):
        with pytest.raises(ValueError, match="PipeFusion requires pipeline_parallel_size > 1"):
            DiffusionParallelConfig(enable_pipefusion=True)

    def test_pipefusion_valid_with_pipeline_parallelism(self):
        config = DiffusionParallelConfig(pipeline_parallel_size=2, enable_pipefusion=True)

        assert config.enable_pipefusion is True
        assert config.pipeline_parallel_size == 2
