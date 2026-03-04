# RFC: Performance Dashboard for Diffusion Models

- **Author(s):** (your name)
- **Status:** Draft
- **Created:** 2026-03-02
- **Last Updated:** 2026-03-02 (rev2: aligned config matrix with Omni model test.json pattern)

---

## 1. Summary

This RFC proposes a structured, reproducible **performance dashboard** for diffusion model serving in vLLM-Omni. The dashboard will track key serving metrics — latency percentiles, throughput, SLO attainment, and peak GPU memory — for two target models:

| Model | Task |
|-------|------|
| `Qwen/Qwen-Image` | Text-to-Image (t2i) |
| `Wan2.2` (`WanPipeline`) | Text-to-Video (t2v) |

The goal is to give the team a single, authoritative source of truth for performance regressions and improvements across code changes, hardware configurations, and parallelism strategies.

---

## 2. Motivation and Goals

### 2.1 Problem

Diffusion model serving has fundamentally different performance characteristics from LLM inference:

- The unit of work is a **full image or video clip**, not a token stream.
- Latency scales with resolution (`width × height`), frame count, and number of inference steps — not sequence length.
- There is no existing CI gate or dashboard that tracks these metrics for `Qwen-Image` or `Wan2.2`.

Without a dashboard:
- Performance regressions can silently land in main.
- There is no baseline to compare against when tuning parallelism (e.g., sequence parallel, tensor parallel).
- SLO targets are informally discussed and never validated at scale.

### 2.2 Goals

1. **Regression detection** — catch latency or throughput regressions before they reach main.
2. **Reproducibility** — all benchmark runs are driven from a versioned config file; results are saved as structured JSON.
3. **SLO visibility** — report SLO attainment rate alongside raw latency numbers.
4. **Multi-config coverage** — cover the most common production-like configurations (resolution, concurrency, parallelism).
5. **Low maintenance overhead** — reuse the existing `benchmarks/diffusion/diffusion_benchmark_serving.py` infrastructure; the dashboard is just a config matrix + result aggregation layer on top.

### 2.3 Non-Goals

- This RFC does **not** propose a web-based visualization UI (out of scope for now; results are JSON + printed tables).
- This RFC does **not** cover LLM / audio-generation benchmarking (tracked separately via `tests/perf/scripts/run_benchmark.py`).
- This RFC does **not** propose automated hardware provisioning.

---

## 3. Background

### 3.1 Existing Infrastructure

`benchmarks/diffusion/diffusion_benchmark_serving.py` already provides:

- Async HTTP request dispatch via `aiohttp`.
- Three dataset modes: `vbench`, `trace`, `random`.
- Per-request SLO tracking (`slo_ms`, `slo_achieved`).
- Warmup-based SLO inference (linear scaling by area × frames × steps).
- Structured JSON output via `--output-file`.

Metrics currently emitted:

| Field | Description |
|-------|-------------|
| `throughput_qps` | Successful requests per second |
| `latency_mean` | Mean end-to-end latency (s) |
| `latency_median` | Median latency (s) |
| `latency_p50` | P50 latency (s) |
| `latency_p99` | P99 latency (s) |
| `peak_memory_mb_max` | Max peak GPU memory across requests (MB) |
| `peak_memory_mb_mean` | Mean peak GPU memory (MB) |
| `slo_attainment_rate` | Fraction of requests meeting `slo_ms` |
| `slo_met_success` | Absolute count of SLO-met requests |

### 3.2 Target Models

**`Qwen/Qwen-Image`** (`QwenImagePipeline`)
- Task: t2i
- Backend: `vllm-omni` (`/v1/chat/completions`)
- Representative resolutions: 512×512, 1024×1024

**`Wan2.2`** (`WanPipeline`)
- Task: t2v
- Backend: `vllm-omni` (`/v1/chat/completions`)
- Representative resolution: 480×640, 720×1280
- Representative frame counts: 16, 49, 81 frames at 16 fps

---

## 4. Proposed Design

### 4.1 Configuration Matrix

Each model × task combination is benchmarked as a **concurrency sweep**, mirroring the pattern used for Omni models in `tests/perf/tests/test.json`. The `num_prompts` array scales proportionally with `max_concurrency` so that each concurrency slot has a sustained load throughout the run (roughly 5 requests per slot for t2i, 3 for t2v). The matrix is stored as a versioned JSON file at `benchmarks/diffusion/dashboard_configs.json`.

#### Design Principles (aligned with Omni model `test.json`)

| Dimension | Omni models | Diffusion models |
|-----------|-------------|------------------|
| Concurrency sweep | `[1, 4, 10]` | `[1, 4, 10]` (same) |
| `num_prompts` | `[10, 40, 100]` | `[5, 20, 50]` (t2i) / `[3, 12, 30]` (t2v, slower) |
| Dataset variants | `random` + `random-mm` | `vbench` (fixed res) + `trace` (heterogeneous res) |
| Server variants | standard + `async_chunk` | baseline + SP=2 + TP=2 + SP=2&ring=2 (see [Parallelism Config](./diffusion_dashboard_parallelism_config.md)) |
| Baseline strategy | Loose smoke-test thresholds | Loose initially; tighten after real data collected |

#### Parallelism Variants (Qwen-Image & Wan2.2)

Both models require monitoring **four** server variants per [Parallelism Config](./diffusion_dashboard_parallelism_config.md):

| Variant | Description | `parallel_config` |
|---------|-------------|------------------|
| **baseline** | Single-GPU | default |
| **sp2-ulysses** | SP=2 (Ulysses) | `ulysses_degree=2`, `ring_degree=1` |
| **tp2** | TP=2 | `tensor_parallel_size=2` |
| **sp2-ring** | SP=2 (Ring) | `ulysses_degree=1`, `ring_degree=2` |

#### Qwen-Image (t2i) — Configuration Matrix

Two dataset variants × four parallelism variants × one concurrency sweep each.

| Config ID | Dataset | Resolution | Steps | Parallelism | `num_prompts` | `max_concurrency` | SLO Scale |
|-----------|---------|------------|-------|-------------|---------------|-------------------|-----------|
| `qi-512-vbench` | vbench | 512×512 | 20 | baseline | [5, 20, 50] | [1, 4, 10] | 3.0 |
| `qi-1024-vbench` | vbench | 1024×1024 | 20 | baseline | [5, 20, 50] | [1, 4, 10] | 3.0 |
| `qi-512-trace` | trace (sd3_trace.txt) | per-request | per-request | baseline | [5, 20, 50] | [1, 4, 10] | 3.0 |
| `qi-512-vbench-sp2-ulysses` | vbench | 512×512 | 20 | sp2-ulysses | [5, 20, 50] | [1, 4, 10] | 3.0 |
| `qi-512-vbench-tp2` | vbench | 512×512 | 20 | tp2 | [5, 20, 50] | [1, 4, 10] | 3.0 |
| `qi-512-vbench-sp2-ring` | vbench | 512×512 | 20 | sp2-ring | [5, 20, 50] | [1, 4, 10] | 3.0 |

#### Wan2.2 (t2v) — Configuration Matrix

Two dataset variants × four parallelism variants × one concurrency sweep each.

| Config ID | Dataset | Resolution | Frames | Steps | Parallelism | `num_prompts` | `max_concurrency` | SLO Scale |
|-----------|---------|------------|--------|-------|-------------|---------------|-------------------|-----------|
| `wan-480-vbench` | vbench | 480×640 | 49 | 20 | baseline | [3, 12, 30] | [1, 4, 10] | 3.0 |
| `wan-720-vbench` | vbench | 720×1280 | 49 | 20 | baseline | [3, 12, 30] | [1, 4, 10] | 3.0 |
| `wan-480-trace` | trace (cogvideox_trace.txt) | per-request | per-request | baseline | [3, 12, 30] | [1, 4, 10] | 3.0 |
| `wan-480-vbench-sp2-ulysses` | vbench | 480×640 | 49 | 20 | sp2-ulysses | [3, 12, 30] | [1, 4, 10] | 3.0 |
| `wan-480-vbench-tp2` | vbench | 480×640 | 49 | 20 | tp2 | [3, 12, 30] | [1, 4, 10] | 3.0 |
| `wan-480-vbench-sp2-ring` | vbench | 480×640 | 49 | 20 | sp2-ring | [3, 12, 30] | [1, 4, 10] | 3.0 |

### 4.2 Config File Format

The schema aligns with `tests/perf/tests/test.json`: `num_prompts` and `max_concurrency` are parallel arrays, and each top-level entry groups all sweep points for one (model, task, dataset, server-variant) combination. The runner iterates the arrays in lockstep, running one benchmark per `(num_prompts[i], max_concurrency[i])` pair.

```json
[
  {
    "test_name": "test_qwen_image",
    "server_params": {
      "model": "Qwen/Qwen-Image",
      "stage_config_name": "qwen_image.yaml"
    },
    "benchmark_params": [
      {
        "dataset_name": "vbench",
        "task": "t2i",
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "num_prompts": [5, 20, 50],
        "max_concurrency": [1, 4, 10],
        "slo": true,
        "slo_scale": 3.0,
        "warmup_requests": 2,
        "warmup_num_inference_steps": 1,
        "baseline": {
          "throughput_qps": 0.01,
          "latency_p99": 999.0,
          "slo_attainment_rate": 0.0
        }
      },
      {
        "dataset_name": "vbench",
        "task": "t2i",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 20,
        "num_prompts": [5, 20, 50],
        "max_concurrency": [1, 4, 10],
        "slo": true,
        "slo_scale": 3.0,
        "warmup_requests": 2,
        "warmup_num_inference_steps": 1,
        "baseline": {
          "throughput_qps": 0.01,
          "latency_p99": 999.0,
          "slo_attainment_rate": 0.0
        }
      },
      {
        "dataset_name": "trace",
        "task": "t2i",
        "num_prompts": [5, 20, 50],
        "max_concurrency": [1, 4, 10],
        "slo": true,
        "slo_scale": 3.0,
        "warmup_requests": 2,
        "warmup_num_inference_steps": 1,
        "baseline": {
          "throughput_qps": 0.01,
          "latency_p99": 999.0,
          "slo_attainment_rate": 0.0
        }
      }
    ]
  },
  {
    "test_name": "test_wan2",
    "server_params": {
      "model": "Wan2.2",
      "stage_config_name": "wan2_2.yaml"
    },
    "benchmark_params": [
      {
        "dataset_name": "vbench",
        "task": "t2v",
        "width": 640,
        "height": 480,
        "num_frames": 49,
        "fps": 16,
        "num_inference_steps": 20,
        "num_prompts": [3, 12, 30],
        "max_concurrency": [1, 4, 10],
        "slo": true,
        "slo_scale": 3.0,
        "warmup_requests": 1,
        "warmup_num_inference_steps": 1,
        "baseline": {
          "throughput_qps": 0.01,
          "latency_p99": 9999.0,
          "slo_attainment_rate": 0.0
        }
      },
      {
        "dataset_name": "trace",
        "task": "t2v",
        "num_prompts": [3, 12, 30],
        "max_concurrency": [1, 4, 10],
        "slo": true,
        "slo_scale": 3.0,
        "warmup_requests": 1,
        "warmup_num_inference_steps": 1,
        "baseline": {
          "throughput_qps": 0.01,
          "latency_p99": 9999.0,
          "slo_attainment_rate": 0.0
        }
      }
    ]
  },
  {
    "test_name": "test_wan2_sp2",
    "server_params": {
      "model": "Wan2.2",
      "stage_config_name": "wan2_2.yaml",
      "update": {
        "stage_args": {
          "0": { "engine_args.sequence_parallel_size": 2 }
        }
      }
    },
    "benchmark_params": [
      {
        "dataset_name": "vbench",
        "task": "t2v",
        "width": 640,
        "height": 480,
        "num_frames": 49,
        "fps": 16,
        "num_inference_steps": 20,
        "num_prompts": [3, 12, 30],
        "max_concurrency": [1, 4, 10],
        "slo": true,
        "slo_scale": 3.0,
        "warmup_requests": 1,
        "warmup_num_inference_steps": 1,
        "baseline": {
          "throughput_qps": 0.01,
          "latency_p99": 9999.0,
          "slo_attainment_rate": 0.0
        }
      }
    ]
  }
]
```

> **Note on baseline values**: All `baseline` entries are intentionally set to trivially loose thresholds (`throughput_qps: 0.01`, `latency_p99: 999.0`, `slo_attainment_rate: 0.0`) in Phase 1. These act as smoke-test guards (ensure the server does not crash) and will be tightened in Phase 3 after real numbers are collected on reference hardware — following the same approach used in `tests/perf/tests/test.json`.

### 4.3 Dashboard Runner Script

A new lightweight runner at `benchmarks/diffusion/run_dashboard.py` will:

1. Parse `dashboard_configs.json`.
2. For each config entry, invoke `diffusion_benchmark_serving.py` as a subprocess (mirroring the pattern in `tests/perf/scripts/run_benchmark.py`).
3. Collect each `--output-file` JSON result.
4. Aggregate all results into a single `dashboard_results_<timestamp>.json`.
5. Print a summary table to stdout.
6. Optionally assert each result against the `baseline` section (for CI use).

```
Usage:
  python benchmarks/diffusion/run_dashboard.py \
    --config benchmarks/diffusion/dashboard_configs.json \
    --base-url http://localhost:8091 \
    --output-dir ./dashboard_results \
    [--assert-baselines]
```

### 4.4 Metrics Collected Per Run

For every config entry the following metrics are recorded in the output JSON:

#### Latency

| Metric | Unit | Notes |
|--------|------|-------|
| `latency_mean` | seconds | Mean end-to-end generation latency |
| `latency_median` | seconds | P50 |
| `latency_p99` | seconds | Tail latency |

#### Throughput

| Metric | Unit | Notes |
|--------|------|-------|
| `throughput_qps` | req/s | Successful requests per second over the full benchmark window |

#### SLO

| Metric | Unit | Notes |
|--------|------|-------|
| `slo_attainment_rate` | fraction [0,1] | Fraction of requests that finished within `slo_ms` |
| `slo_met_success` | count | Absolute number of SLO-met requests |
| `slo_scale` | multiplier | The `slo_ms = expected_latency × slo_scale` factor used |

#### Memory

| Metric | Unit | Notes |
|--------|------|-------|
| `peak_memory_mb_max` | MB | Reported by server in response body; 0 if not available |
| `peak_memory_mb_mean` | MB | Mean across successful requests |

#### Reliability

| Metric | Unit | Notes |
|--------|------|-------|
| `completed_requests` | count | Number of successful requests |
| `failed_requests` | count | Number of failed requests |
| `duration` | seconds | Total wall-clock time of the benchmark run |

### 4.5 Output Format

Each `(num_prompts[i], max_concurrency[i])` sweep point produces one result file at `dashboard_results/<test_name>_<dataset>_<concurrency>_<num_prompts>_<timestamp>.json`, mirroring the naming convention in `run_benchmark.py`:

```json
{
  "test_name": "test_qwen_image",
  "dataset_name": "vbench",
  "task": "t2i",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 20,
  "max_concurrency": 4,
  "num_prompts": 20,
  "results": {
    "duration": 62.4,
    "completed_requests": 20,
    "failed_requests": 0,
    "throughput_qps": 0.80,
    "latency_mean": 4.82,
    "latency_median": 4.71,
    "latency_p50": 4.71,
    "latency_p99": 6.30,
    "slo_attainment_rate": 0.96,
    "slo_met_success": 19,
    "slo_scale": 3.0,
    "peak_memory_mb_max": 18432.0,
    "peak_memory_mb_mean": 17980.5
  },
  "baseline_pass": true
}
```

A rolled-up `dashboard_summary_<timestamp>.json` contains one entry per sweep point with `pass/fail` status, making it easy to spot the concurrency level at which performance degrades:</p>

```json
[
  { "test_name": "test_qwen_image", "dataset": "vbench-512", "concurrency": 1,  "num_prompts": 5,  "throughput_qps": 0.95, "latency_p99": 4.1,  "slo_attainment_rate": 1.00, "pass": true  },
  { "test_name": "test_qwen_image", "dataset": "vbench-512", "concurrency": 4,  "num_prompts": 20, "throughput_qps": 0.80, "latency_p99": 6.3,  "slo_attainment_rate": 0.96, "pass": true  },
  { "test_name": "test_qwen_image", "dataset": "vbench-512", "concurrency": 10, "num_prompts": 50, "throughput_qps": 0.55, "latency_p99": 12.8, "slo_attainment_rate": 0.82, "pass": true  },
  { "test_name": "test_wan2",       "dataset": "vbench-480", "concurrency": 1,  "num_prompts": 3,  "throughput_qps": 0.10, "latency_p99": 85.0, "slo_attainment_rate": 1.00, "pass": true  },
  ...
]
```

---

## 5. Implementation Plan

### Phase 1 — Config Matrix & Baseline Collection (Week 1–2)

- [ ] Define `dashboard_configs.json` with the full config matrix for both models.
- [ ] Run the existing `diffusion_benchmark_serving.py` manually for each config on reference hardware to establish initial `baseline` values.
- [ ] Check in the config file and baseline values to the repo.

### Phase 2 — Runner Script (Week 2–3)

- [ ] Implement `benchmarks/diffusion/run_dashboard.py`.
- [ ] Support `--assert-baselines` flag: fails if `throughput_qps < baseline.throughput_qps` or `latency_p99 > baseline.latency_p99` or `slo_attainment_rate < baseline.slo_attainment_rate`.
- [ ] Add unit tests for the runner's config-parsing and assertion logic.

### Phase 3 — CI Integration (Week 3–4)

- [ ] Add a new CI job (Level 4 or Level 5 per `docs/contributing/ci/CI_5levels.md`) that:
  - Starts the vLLM-Omni server for each model.
  - Invokes `run_dashboard.py --assert-baselines`.
  - Uploads `dashboard_results/` as a CI artifact.
- [ ] Gate merges on this job for PRs touching `vllm_omni/diffusion/`.

### Phase 4 — Parallelism Coverage (Week 4–5)

- [ ] Extend the config matrix with sequence-parallel configurations (SP=2, SP=4) for Wan2.2 (t2v), reusing the `--extra-body` mechanism to pass SP settings.
- [ ] Document the expected throughput scaling behavior per SP degree.

---

## 6. Open Questions

1. **Baseline hardware** — Which GPU SKU(s) should be the reference for baseline numbers? A8000, H100, or both? The baseline values are hardware-specific and need to be clearly labeled.

2. **SLO definition for t2v** — For video generation, should `slo_ms` be defined as absolute wall-clock latency, or normalized by `num_frames` (i.e., latency-per-frame SLO)? The current implementation uses absolute latency; per-frame SLO may be more meaningful for Wan2.2.

3. **Trace vs. VBench dataset** — Should the dashboard use `vbench` (fixed resolution per run) or `trace` (heterogeneous resolutions in a single run)? Trace-based runs are more realistic but make baseline comparisons harder. Recommendation: use `vbench` for baselines, `trace` for stress tests.

4. **Frequency of runs** — Should the dashboard run on every PR, nightly, or only on tagged releases? Given the long warm-up time for diffusion servers (multi-minute model load), a nightly cadence may be more practical.

5. **Memory metric availability** — `peak_memory_mb` is currently only reported if the server includes it in the response body. Should we standardize this field in the API response contract for both `Qwen-Image` and `Wan2.2`?

---

## 7. Alternatives Considered

### 7.1 Extend `tests/perf/scripts/run_benchmark.py` for diffusion models

`run_benchmark.py` dispatches to `vllm bench serve`, which is token-centric and does not support diffusion-specific parameters (resolution, frames, steps, SLO-by-latency). Extending it for diffusion would require forking the entire metric pipeline. Using `diffusion_benchmark_serving.py` directly is cleaner.

### 7.2 Build a separate Grafana / Prometheus dashboard

This would require persistent metric storage infrastructure that the project does not currently have. Deferring to JSON files + CI artifacts is sufficient for the near term.

### 7.3 Use the `trace` dataset exclusively

Trace-based runs produce more realistic load patterns but require maintaining separate trace files per model. Starting with `vbench` + fixed configs is simpler to reason about and easier to reproduce.

---

## 8. References

- [Diffusion Dashboard Parallelism Config](./diffusion_dashboard_parallelism_config.md) — defines SP=2, TP=2, SP=2&ring=2 configurations
- `benchmarks/diffusion/diffusion_benchmark_serving.py` — primary benchmark entrypoint
- `benchmarks/diffusion/backends.py` — `RequestFuncInput`, `RequestFuncOutput`, backend dispatch
- `benchmarks/diffusion/README.md` — usage documentation
- `vllm_omni/diffusion/registry.py` — `QwenImagePipeline`, `WanPipeline` registration
- `tests/perf/scripts/run_benchmark.py` — reference pattern for config-driven benchmark runners
- `tests/perf/tests/test.json` — reference pattern for benchmark config schema
- `docs/contributing/ci/CI_5levels.md` — CI level definitions
