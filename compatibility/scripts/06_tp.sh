#!/usr/bin/env bash
# ── Row 6: Tensor Parallel ───────────────────────────────────────────────────
# 兼容性矩阵第 6 行：以 tp 为基线，测试与缓存加速、序列并行、CFG 并行的组合。
# 需要 ≥2 GPU（tp gpu_multiplier=2）；多组合时可能需要 4 GPU。
#
# 🙋 待测组合（addons）:
#   teacache      — TP + TeaCache
#   cache_dit     — TP + Cache-DiT
#   ulysses       — TP + Ulysses-SP      (需 4 GPU)
#   ring          — TP + Ring-Attn       (需 4 GPU)
#   cfg_parallel  — TP + CFG-Parallel    (需 4 GPU)
#
# ❌ 已知冲突（自动跳过）: tp + hsdp
#
# 注：GPU 不足时框架自动标记 SKIP (GPU)。
#
# 用法:
#   bash compatibility/scripts/06_tp.sh
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
echo "Row 6 | baseline: tp | addons: teacache cache_dit ulysses ring cfg_parallel"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature tp \
    --addons teacache cache_dit ulysses ring cfg_parallel \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: tp ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/tp" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
