#!/usr/bin/env bash
# ── Row 4: Ring-Attn ─────────────────────────────────────────────────────────
# Compatibility matrix row 4: ring as baseline, test combinations with cache and sequence parallel features.
# Requires ≥2 GPUs (ring gpu_multiplier=2); ring+ulysses requires 4 GPUs.
#
# Addon combinations to test:
#   teacache   — Ring + TeaCache            ✅ verified compatible
#   cache_dit  — Ring + Cache-DiT           ✅ verified compatible
#   ulysses    — Ring + Ulysses-SP          ✅ verified compatible (requires 4 GPUs)
#
# Note: ring+ulysses combination requires 4 GPUs; the framework auto-marks SKIP (GPU) if insufficient.
#
# Usage:
#   bash compatibility/scripts/04_ring.sh
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
echo "Row 4 | baseline: ring | addons: teacache cache_dit ulysses"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature ring \
    --addons teacache cache_dit ulysses \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: ring ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/ring" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
