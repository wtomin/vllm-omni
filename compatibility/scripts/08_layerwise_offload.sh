#!/usr/bin/env bash
# ── Row 8: CPU Offloading (Layerwise) ────────────────────────────────────────
# 兼容性矩阵第 8 行：以 layerwise_offload 为基线，测试与缓存加速特性的组合。
# 仅支持单卡（SINGLE_CARD_ONLY）；所有多卡特性均已标记 ❌，框架会自动跳过。
#
# 🙋 待测组合（addons）:
#   teacache   — Layerwise Offload + TeaCache
#   cache_dit  — Layerwise Offload + Cache-DiT
#
# ❌ 已知冲突（自动跳过）: layerwise_offload + ulysses/ring/cfg_parallel/tp/hsdp
#
# 用法:
#   bash compatibility/scripts/08_layerwise_offload.sh
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
echo "Row 8 | baseline: layerwise_offload | addons: teacache cache_dit"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature layerwise_offload \
    --addons teacache cache_dit \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: layerwise_offload ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/layerwise_offload" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
