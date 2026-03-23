#!/usr/bin/env bash
# ── Row 10: FP8 Quant ────────────────────────────────────────────────────────
# 兼容性矩阵第 10 行：以 fp8 为基线，测试与所有其余 🙋 特性的组合。
# FP8 本身仅需 1 GPU；搭配并行特性时按各特性的 gpu_multiplier 累乘。
#
# 🙋 待测组合（addons）:
#   teacache           — FP8 + TeaCache
#   cache_dit          — FP8 + Cache-DiT
#   ulysses            — FP8 + Ulysses-SP          (需 2 GPU)
#   ring               — FP8 + Ring-Attn            (需 2 GPU)
#   cfg_parallel       — FP8 + CFG-Parallel         (需 2 GPU)
#   tp                 — FP8 + Tensor Parallel       (需 2 GPU)
#   hsdp               — FP8 + HSDP                 (需 2 GPU)
#   layerwise_offload  — FP8 + Layerwise Offload     (单卡；多卡组合自动跳过)
#   vae_patch_parallel — FP8 + VAE Patch Parallel    (需并行基线提供 >1 rank)
#
# 注：多方并行组合（如 fp8+ulysses+ring）可能需要 4 GPU；
#     GPU 不足时框架自动标记 SKIP (GPU)。
#
# 用法:
#   bash compatibility/scripts/10_fp8.sh
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
echo "Row 10 | baseline: fp8"
echo "        addons: teacache cache_dit ulysses ring cfg_parallel tp hsdp"
echo "                layerwise_offload vae_patch_parallel"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature fp8 \
    --addons teacache cache_dit ulysses ring cfg_parallel tp hsdp \
             layerwise_offload vae_patch_parallel \
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
