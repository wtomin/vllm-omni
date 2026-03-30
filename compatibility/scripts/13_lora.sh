#!/usr/bin/env bash
# ── Row 13: LoRA Inference ───────────────────────────────────────────────────
# Compatibility matrix row 13: lora as baseline, test combinations with all
# other features. All combinations are currently ❓ (not yet verified).
#
# Requires:
#   --lora-path PATH   path to a valid LoRA adapter directory (required)
#
# Addon combinations to test (all ❓ — compatibility not yet verified):
#   teacache           — LoRA + TeaCache
#   cache_dit          — LoRA + Cache-DiT
#   ulysses            — LoRA + Ulysses-SP          (requires 2 GPUs)
#   ring               — LoRA + Ring-Attn            (requires 2 GPUs)
#   cfg_parallel       — LoRA + CFG-Parallel         (requires 2 GPUs)
#   tp                 — LoRA + Tensor Parallel       (requires 2 GPUs)
#   hsdp               — LoRA + HSDP                 (requires 2 GPUs)
#   layerwise_offload  — LoRA + Layerwise Offload     (single-GPU only)
#   cpu_offload        — LoRA + CPU Offload (Module-wise)
#   vae_patch_parallel — LoRA + VAE Patch Parallel    (requires parallel baseline)
#   fp8                — LoRA + FP8 Quantization
#
# Note: the framework auto-marks SKIP (GPU) when GPUs are insufficient.
#
# Usage:
#   LORA_PATH=/path/to/adapter bash compatibility/scripts/13_lora.sh
#   MODEL=Tongyi-MAI/Z-Image-Turbo LORA_PATH=/path/to/adapter bash compatibility/scripts/13_lora.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen-Image}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
STEPS="${STEPS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-./compat_results}"
CHARTS="${CHARTS:-1}"
LORA_PATH="${LORA_PATH:-}"

if [[ -z "${LORA_PATH}" ]]; then
    echo "[ERROR] LORA_PATH must be set to a valid LoRA adapter directory."
    echo "        Usage: LORA_PATH=/path/to/adapter bash $0"
    exit 1
fi

echo "======================================================================"
echo "Row 13 | baseline: lora | lora-path: ${LORA_PATH}"
echo "        addons: teacache cache_dit ulysses ring cfg_parallel tp hsdp"
echo "                layerwise_offload cpu_offload vae_patch_parallel fp8"
echo "======================================================================"

python "${SCRIPT_DIR}/../run_compat_test.py" \
    --baseline-feature lora \
    --lora-path "${LORA_PATH}" \
    --addons teacache cache_dit ulysses ring cfg_parallel tp hsdp \
             layerwise_offload cpu_offload vae_patch_parallel fp8 \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --steps "${STEPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "--- Analyzing results: lora ---"
python "${REPO_ROOT}/tests/e2e/offline_inference/analyze_compat_results.py" \
    --results-dir "${OUTPUT_DIR}/lora" \
    ${CHARTS:+--charts} \
    || echo "[WARN] analyze_compat_results returned non-zero (check results above)"
