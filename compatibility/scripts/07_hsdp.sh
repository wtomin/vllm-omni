#!/usr/bin/env bash
# ── Row 7: HSDP ──────────────────────────────────────────────────────────────
# 兼容性矩阵第 7 行：以 hsdp 为基线，测试与缓存加速、序列并行、CFG 并行的组合。
# 需要 ≥2 GPU（hsdp gpu_multiplier=2）；多组合时可能需要 4 GPU。
#
# 🙋 待测组合（addons）:
#   teacache      — HSDP + TeaCache
#   cache_dit     — HSDP + Cache-DiT
#   ulysses       — HSDP + Ulysses-SP      (需 4 GPU)
#   ring          — HSDP + Ring-Attn       (需 4 GPU)
#   cfg_parallel  — HSDP + CFG-Parallel    (需 4 GPU)
#
# ❌ 已知冲突（自动跳过）: hsdp + tp
#
# 注：GPU 不足时框架自动标记 SKIP (GPU)。
#
# 用法:
#   bash compatibility/scripts/07_hsdp.sh
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
echo "Row 7 | baseline: hsdp | addons: teacache cache_dit ulysses ring cfg_parallel"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature hsdp \
    --addons teacache cache_dit ulysses ring cfg_parallel \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: hsdp ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/hsdp" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
