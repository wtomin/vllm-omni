#!/usr/bin/env bash
# ── Row 8: Expert Parallel ───────────────────────────────────────────────────
# Compatibility matrix row 8: expert_parallel as baseline, test combinations
# with cache acceleration and other parallelism features.
# Requires ≥2 GPUs (expert_parallel gpu_multiplier=2); multi-feature combinations
# may require 4 GPUs.
#
# NOTE: The 'expert_parallel' feature must be registered in FEATURE_REGISTRY
# inside run_compat_test.py before this script can run successfully.
#
# Addon combinations to test (all ❓ — compatibility not yet verified):
#   teacache      — Expert Parallel + TeaCache
#   cache_dit     — Expert Parallel + Cache-DiT
#   ulysses       — Expert Parallel + Ulysses-SP      (requires 4 GPUs)
#   ring          — Expert Parallel + Ring-Attn       (requires 4 GPUs)
#   cfg_parallel  — Expert Parallel + CFG-Parallel    (requires 4 GPUs)
#   tp            — Expert Parallel + Tensor Parallel (requires 4 GPUs)
#   hsdp          — Expert Parallel + HSDP            (requires 4 GPUs)
#
# Note: the framework auto-marks SKIP (GPU) when GPUs are insufficient.
#
# Usage:
#   bash compatibility/scripts/12_expert_parallel.sh
#   MODEL=Tongyi-MAI/Z-Image-Turbo bash compatibility/scripts/12_expert_parallel.sh
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
echo "Row 8 | baseline: expert_parallel"
echo "        addons: teacache cache_dit ulysses ring cfg_parallel tp hsdp"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature expert_parallel \
    --addons teacache cache_dit ulysses ring cfg_parallel tp hsdp \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: expert_parallel ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/expert_parallel" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
