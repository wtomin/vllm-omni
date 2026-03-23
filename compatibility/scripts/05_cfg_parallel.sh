#!/usr/bin/env bash
# ── Row 5: CFG-Parallel ──────────────────────────────────────────────────────
# 兼容性矩阵第 5 行：以 cfg_parallel 为基线，测试与缓存加速和序列并行特性的组合。
# 需要 ≥2 GPU（cfg_parallel gpu_multiplier=2）。
#
# 🙋 待测组合（addons）:
#   teacache   — CFG-Parallel + TeaCache
#   cache_dit  — CFG-Parallel + Cache-DiT
#   ulysses    — CFG-Parallel + Ulysses-SP      (需 4 GPU)
#   ring       — CFG-Parallel + Ring-Attn       (需 4 GPU)
#
# 注：ulysses / ring 组合需要 4 GPU；GPU 不足时框架会自动标记 SKIP (GPU)。
#
# 用法:
#   bash compatibility/scripts/05_cfg_parallel.sh
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
echo "Row 5 | baseline: cfg_parallel | addons: teacache cache_dit ulysses ring"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature cfg_parallel \
    --addons teacache cache_dit ulysses ring \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: cfg_parallel ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/cfg_parallel" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
