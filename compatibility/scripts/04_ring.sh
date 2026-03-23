#!/usr/bin/env bash
# ── Row 4: Ring-Attn ─────────────────────────────────────────────────────────
# 兼容性矩阵第 4 行：以 ring 为基线，测试与缓存加速特性的组合。
# 需要 ≥2 GPU（ring gpu_multiplier=2）。
#
# ✅ 已验证: ring + ulysses（矩阵中标记为 ✅，不在本脚本重跑）
#
# 🙋 待测组合（addons）:
#   teacache   — Ring + TeaCache
#   cache_dit  — Ring + Cache-DiT
#
# 用法:
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
echo "Row 4 | baseline: ring | addons: teacache cache_dit"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature ring \
    --addons teacache cache_dit \
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
