# PR Review: #1920 — [Docs] Add Wan2.1-T2V-1.3B as supported video generation model

**Repository:** vllm-project/vllm-omni  
**PR URL:** https://github.com/vllm-project/vllm-omni/pull/1920  
**Author:** SamitHuang  
**Reviewed:** 2026-03-17  
**Verdict:** 🔴 REQUEST CHANGES

---

## Summary

This is a documentation-only PR that registers `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` as a supported video generation model. It modifies two files: the VideoGen parallelism table in `parallelism_acceleration.md` and the text-to-video usage example in `text_to_video.md`. No production code is changed.

The PR body is well-structured and includes a generation script, latency numbers, and VRAM usage. However, four issues must be resolved before merge: a missing sample output, two incomplete documentation tables, and unvalidated parallelism feature claims.

---

## Gate Checks

| Check | Status |
|-------|--------|
| DCO | ✅ SUCCESS |
| pre-commit | ✅ SUCCESS |
| Mergeability | ✅ MERGEABLE |
| ReadTheDocs | ✅ SUCCESS |

All gates pass. Full review proceeds.

---

## Dimension 1: PR Body Evidence

| Item | Status | Notes |
|------|--------|-------|
| vLLM-Omni generation script | ✅ Present | Offline `text_to_video.py` with all parameters shown |
| Generated sample output | 🔴 **Missing** | No video clip, GIF, or frame attached or linked |
| vLLM-Omni e2e latency | ✅ Present | 13.1s on single H800, 20 steps, 480×832, 33 frames |
| vLLM-Omni peak VRAM | ✅ Present | 13.7 GiB reported at load time |
| diffusers generation script | 💡 Absent | Recommended for quality comparison baseline |
| diffusers sample output | 💡 Absent | Recommended |
| diffusers latency | 💡 Absent | Recommended |
| diffusers VRAM | 💡 Absent | Recommended |

### Blocking: Missing Sample Output

The PR includes latency and VRAM numbers but no actual generated video is attached or linked. A sample output is required for all video model PRs regardless of whether they add new code.

```
🔴 Missing sample output.

Please attach or link at least one generated video (MP4, GIF, or animated preview) 
for `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`. The generation script and numbers are 
present, but reviewers cannot verify output quality without a visual sample.
```

---

## Dimension 2: Code Review

This PR introduces no production code changes. All code-level checks (diffusers mixin cleanup, acceleration feature implementation, memory optimization implementation) are N/A.

### Observation: Parallelism Claims Are Inferred, Not Validated

The PR adds all-✅ entries for Wan2.1 in the VideoGen parallelism table (Ulysses-SP, Ring-Attention, TP, HSDP, VAE-Patch-Parallel), identical to Wan2.2. The PR body justifies this by stating the model shares the same `WanPipeline` architecture and `WanTransformer3DModel`.

However, the test plan ran only basic offline inference using `--enforce-eager` with no parallelism enabled:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/offline_inference/text_to_video/text_to_video.py \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --enforce-eager \
    ...
```

None of the five parallelism features claimed as ✅ were directly tested. The `--enforce-eager` flag also disables CUDA graph execution, meaning the test does not validate standard production inference behavior.

While architectural parity with Wan2.2 is a reasonable inference, documenting ✅ for untested configurations creates a risk of directing users to unsupported configurations. The PR should either:
1. Run at least one parallelism mode (e.g., Ulysses-SP=2) and report the result, or
2. Mark unvalidated features as `❓` (untested/assumed via architecture parity) until confirmed.

This is a non-blocking observation. If the project convention is to inherit feature support documentation from architecturally identical pipelines without re-running full test suites, this can be waived with an explicit note.

### Observation: Online Serving Not Verified

The test plan covers only offline inference. For completeness, a brief online serving check (`vllm serve Wan-AI/Wan2.1-T2V-1.3B-Diffusers --omni`) is expected. The PR body does not mention whether the model was tested in online mode.

---

## Dimension 3: Documentation

### Files Changed

| File | Changed | Status |
|------|---------|--------|
| `docs/user_guide/diffusion/parallelism_acceleration.md` | ✅ | Wan2.1 row added to VideoGen table |
| `docs/user_guide/examples/offline_inference/text_to_video.md` | ✅ | Wan2.1 listed with VRAM and CLI flag notes |
| `docs/models/supported_models.md` | 🔴 **Not updated** | WanPipeline row missing Wan2.1 model ID |
| `docs/user_guide/diffusion_acceleration.md` | 🔴 **Not updated** | VideoGen table missing Wan2.1 row |

---

### Blocking: `docs/models/supported_models.md` Not Updated

The `docs/models/supported_models.md` VideoGen table currently reads:

```
| `WanPipeline` | Wan2.2-T2V, Wan2.2-TI2V | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
```

`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` uses the same `WanPipeline` and should be added as a recognized model identifier in this table. This is the canonical model registry for users discovering which model IDs are supported.

```
🔴 `docs/models/supported_models.md` not updated.

Please add `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` to the `WanPipeline` row in the 
VideoGen section. This is the primary supported-model registry users consult to 
determine which HuggingFace IDs are valid.
```

---

### Blocking: `docs/user_guide/diffusion_acceleration.md` VideoGen Table Not Updated

`diffusion_acceleration.md` contains a comprehensive multi-method support table (TeaCache, Cache-DiT, Ulysses-SP, Ring-Attention, CFG-Parallel, HSDP, VAE-Patch-Parallel) for all models. The current VideoGen section is:

```
| **Wan2.2** | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **LTX-2**  | `Lightricks/LTX-2`                  | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
```

Wan2.1 is absent from this table. Since this file is where users look for a unified view of all acceleration and memory features, omitting Wan2.1 here creates an inconsistency with the update to `parallelism_acceleration.md`.

```
🔴 `docs/user_guide/diffusion_acceleration.md` VideoGen table not updated.

Please add a row for `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`. Based on architecture 
parity with Wan2.2, the expected values are:

| **Wan2.1** | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

(Columns: TeaCache | Cache-DiT | Ulysses-SP | Ring-Attention | CFG-Parallel | HSDP | VAE-Patch-Parallel)

Confirm or adjust the Cache-DiT and CFG-Parallel values based on testing.
```

---

### Changes Validated in Diff

**`parallelism_acceleration.md` (VideoGen table):**

The added row is correctly formatted and consistent with the existing table structure. The model identifier `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` matches the official HuggingFace ID. All feature columns mirror Wan2.2, which is consistent with the stated architecture parity claim.

**`text_to_video.md`:**

The change from a Wan2.2-specific sentence to a general "Wan T2V pipeline" introduction with a two-item list is a clean improvement. The VRAM hints (`~48GB` for Wan2.2, `~16GB` for Wan2.1) are useful for users selecting between models. The `--flow-shift 3.0 --boundary-ratio 0.0` flags are correctly surfaced as required CLI differences.

---

## Dimension 4: Test Coverage

This is a documentation-only PR. No new test files are required. The existing `text_to_video.py` example script functions as the offline inference test and was run by the author.

| Test Type | Status |
|-----------|--------|
| e2e online serving test | 💡 Not added (docs-only; online serving not verified) |
| Offline inference test | ✅ Validated via existing script (as described in PR body) |

---

## Recommended Additions (Non-Blocking)

### diffusers Baseline Comparison

Adding a diffusers comparison would allow users to understand the performance and memory tradeoff of using vLLM-Omni for this smaller 1.3B model versus direct diffusers usage.

```
💡 Recommended: diffusers baseline comparison

For a 1.3B model, the overhead/benefit ratio of vLLM-Omni vs. direct diffusers 
is less obvious than for the 14B Wan2.2. Adding diffusers latency and VRAM 
numbers would help users decide when to use vLLM-Omni for this model.

- diffusers generation script (same prompt, same resolution, same steps)
- diffusers e2e latency
- diffusers peak VRAM
```

---

## Blocking Issue Summary

| # | File | Issue | Severity |
|---|------|-------|----------|
| 1 | PR body | No generated video sample attached or linked | 🔴 Blocking |
| 2 | `docs/models/supported_models.md` | Wan2.1 model ID not added to WanPipeline row | 🔴 Blocking |
| 3 | `docs/user_guide/diffusion_acceleration.md` | VideoGen table not updated with Wan2.1 row | 🔴 Blocking |
| 4 | PR body / parallelism table | Five parallelism features claimed ✅ but none tested directly | ⚠️ Conditional |

---

## What Was Validated

- All CI gates pass (DCO, pre-commit, ReadTheDocs)
- Generation script is runnable and includes correct parameters
- Latency (13.1s) and VRAM (13.7 GiB) numbers are present and plausible for a 1.3B model on H800
- The `--flow-shift 3.0 --boundary-ratio 0.0` CLI flags are correctly documented as required overrides for Wan2.1
- The two changed files are internally consistent
- No code regressions are possible since no production code is modified

## What Must Change Before Approval

1. Attach or link at least one generated video sample to the PR body
2. Update `docs/models/supported_models.md` to add `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` under the `WanPipeline` entry
3. Update `docs/user_guide/diffusion_acceleration.md` VideoGen table with a Wan2.1 row (confirm Cache-DiT and CFG-Parallel values)
