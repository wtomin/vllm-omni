#!/usr/bin/env bash
# ── Row 9: VAE Patch Parallel ─────────────────────────────────────────────────
# 兼容性矩阵第 9 行：vae_patch_parallel 为 addon-only 特性，不能单独作为 baseline。
#
# 本脚本将 vae_patch_parallel 叠加在每一个 🙋 并行基线上，分轮运行；
# 每轮结束后立即对该轮结果调用 analyze_compat_results.py。
# 每轮还附带 teacache / cache_dit 作为附加 addon，覆盖三方组合。
#
# 🙋 待测组合（vae_patch_parallel 作为 addon）:
#   cfg_parallel  + vae_patch_parallel [+ teacache / cache_dit]
#   ulysses       + vae_patch_parallel [+ teacache / cache_dit]
#   ring          + vae_patch_parallel [+ teacache / cache_dit]
#   tp            + vae_patch_parallel [+ teacache / cache_dit]
#   hsdp          + vae_patch_parallel [+ teacache / cache_dit]
#
# ❌ 已知冲突（自动跳过）: vae_patch_parallel + layerwise_offload
#
# 注：所有并行基线均需 ≥2 GPU；多方并行组合可能需要 4 GPU。
#
# 用法:
#   bash compatibility/scripts/09_vae_patch_parallel.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen-Image}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
STEPS="${STEPS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-./compat_results}"
CHARTS="${CHARTS:-1}"

PARALLEL_BASELINES=(cfg_parallel ulysses ring tp hsdp)

for BASELINE in "${PARALLEL_BASELINES[@]}"; do
    RUN_OUTPUT_DIR="${OUTPUT_DIR}/vae_patch_parallel_on_${BASELINE}"

    echo "======================================================================"
    echo "Row 9 | baseline: ${BASELINE} | addons: vae_patch_parallel teacache cache_dit"
    echo "======================================================================"

    python "${SCRIPT_DIR}/../run_compat_test.py" \
        --baseline-feature "${BASELINE}" \
        --addons vae_patch_parallel teacache cache_dit \
        --model "${MODEL}" \
        --num-prompts "${NUM_PROMPTS}" \
        --steps "${STEPS}" \
        --output-dir "${RUN_OUTPUT_DIR}"

    echo ""
    echo "--- Analyzing results: ${BASELINE} (vae_patch_parallel run) ---"
    python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
        --results-dir "${RUN_OUTPUT_DIR}/${BASELINE}" \
        ${CHARTS:+--charts} \
        || echo "[WARN] analyze_compat_results returned non-zero (check results above)"

    echo ""
done

echo "======================================================================"
echo "Row 9 complete — all vae_patch_parallel combination runs finished."
echo "======================================================================"
