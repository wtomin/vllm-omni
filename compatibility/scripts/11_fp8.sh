#!/usr/bin/env bash
# ── Row 12: FP8 Quant ────────────────────────────────────────────────────────
# Compatibility matrix row 12: fp8 as baseline, test combinations with all remaining features.
# FP8 itself requires only 1 GPU; combined with parallel features the gpu_multiplier is multiplied accordingly.
#
# Addon combinations to test:
#   teacache           — FP8 + TeaCache                    ✅ verified
#   cache_dit          — FP8 + Cache-DiT                   ✅ verified
#   ulysses            — FP8 + Ulysses-SP                  ✅ verified (requires 2 GPUs)
#   ring               — FP8 + Ring-Attn                   ✅ verified (requires 2 GPUs)
#   cfg_parallel       — FP8 + CFG-Parallel                ✅ verified (requires 2 GPUs)
#   tp                 — FP8 + Tensor Parallel             ✅ verified (requires 2 GPUs)
#   hsdp               — FP8 + HSDP                       ❓ not yet verified (requires 2 GPUs)
#   layerwise_offload  — FP8 + Layerwise Offload           ✅ verified (single-GPU; multi-GPU combos auto-skipped)
#   cpu_offload        — FP8 + CPU Offload (Module-wise)   ✅ verified
#   vae_patch_parallel — FP8 + VAE Patch Parallel          ✅ verified (requires parallel baseline providing >1 rank)
#
# Note: multi-parallel combinations (e.g. fp8+ulysses+ring) may require 4 GPUs;
#       the framework auto-marks SKIP (GPU) when GPUs are insufficient.
#
# Usage:
#   bash compatibility/scripts/11_fp8.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen-Image}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
STEPS="${STEPS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-./compat_results}"
CHARTS="${CHARTS:-1}"

echo "======================================================================"
echo "Row 12 | baseline: fp8"
echo "        addons: teacache cache_dit ulysses ring cfg_parallel tp hsdp"
echo "                layerwise_offload cpu_offload vae_patch_parallel"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature fp8 \
    --addons teacache cache_dit ulysses ring cfg_parallel tp hsdp \
             layerwise_offload cpu_offload vae_patch_parallel \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: fp8 ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/fp8" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
