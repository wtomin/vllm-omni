#!/usr/bin/env bash
# ── CPU Offload (Module-level) ────────────────────────────────────────────────
# 以 cpu_offload（模块级 CPU 卸载，DiT + 文本编码器整体卸载）为基线，
# 测试与缓存加速、并行特性的组合兼容性。
#
# 注意：cpu_offload 未列入 SINGLE_CARD_ONLY，框架不会自动排除多卡组合；
#       实际能否运行取决于硬件，GPU 不足时框架自动标记 SKIP (GPU)。
#
# 🙋 待测组合（addons）:
#   teacache      — CPU Offload + TeaCache
#   cache_dit     — CPU Offload + Cache-DiT
#   ulysses       — CPU Offload + Ulysses-SP      (需 2 GPU)
#   ring          — CPU Offload + Ring-Attn       (需 2 GPU)
#   cfg_parallel  — CPU Offload + CFG-Parallel    (需 2 GPU)
#   tp            — CPU Offload + Tensor Parallel (需 2 GPU)
#   hsdp          — CPU Offload + HSDP            (需 2 GPU)
#
# ❌ 已知冲突（自动跳过）: cpu_offload + layerwise_offload
#
# 用法:
#   bash compatibility/scripts/11_cpu_offload.sh
#   MODEL=Tongyi-MAI/Z-Image-Turbo bash compatibility/scripts/11_cpu_offload.sh
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
echo "CPU Offload (module-level) | baseline: cpu_offload"
echo "addons: teacache cache_dit ulysses ring cfg_parallel tp hsdp"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature cpu_offload \
    --addons teacache cache_dit ulysses ring cfg_parallel tp hsdp \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: cpu_offload ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/cpu_offload" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
