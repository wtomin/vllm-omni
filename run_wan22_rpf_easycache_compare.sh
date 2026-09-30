#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Round-2 benchmark: single-GPU baseline vs Rotational PipeFusion + cumulative EasyCache.
#
# Requires branch: pipefusion_wan_cache_rpf  (pipefusion_wan_rotate + easycache replay)
# Prereqs on the target machine:
#   - .venv + env.sh working (source env.sh happens inside)
#   - HF cache contains Wan-AI/Wan2.2-TI2V-5B-Diffusers
#   - captions_20.txt at repo root (untracked file, copy it over)
#   - LAZY_CKPT (trained lazy_horizon_predictor_full.pt) reachable
#
# Usage:
#   bash run_wan22_rpf_easycache_compare.sh
#
# Key overrides (env):
#   SINGLE_GPU=0  MULTI_GPUS=0,1,2,3
#   PF_PIPELINE_SIZE=4  PF_WARMUP_STEPS=2
#   LAZY_CKPT=/path/to/lazy_horizon_predictor_full.pt
#   LAZY_THRESHOLD=0.06  LAZY_WARMUP_STEPS=7
#   CAPTIONS=/path/to/captions_20.txt
#   RESULT_DIR=...        (default: timestamped under benchmarks/diffusion/results/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/env.sh"

# ---- Config ----
SINGLE_GPU="${SINGLE_GPU:-0}"
MULTI_GPUS="${MULTI_GPUS:-0,1,2,3}"
PF_PIPELINE_SIZE="${PF_PIPELINE_SIZE:-4}"
PF_WARMUP_STEPS="${PF_WARMUP_STEPS:-2}"
LAZY_CKPT="${LAZY_CKPT:-/home/fq9hpsac/fq9hpsacuser04/didan-new/EasyCache/results/easycache_lazy_cumulative/lazy_horizon_predictor_full.pt}"
LAZY_THRESHOLD="${LAZY_THRESHOLD:-0.06}"
LAZY_WARMUP_STEPS="${LAZY_WARMUP_STEPS:-7}"
CAPTIONS="${CAPTIONS:-${REPO_ROOT}/captions_20.txt}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmarks/diffusion/results/wan22_rpf_easycache_$(date +%Y%m%d_%H%M%S)}"

[[ -f "${CAPTIONS}" ]] || { echo "ERROR: captions file not found: ${CAPTIONS}" >&2; exit 1; }
[[ -f "${LAZY_CKPT}" ]] || { echo "ERROR: lazy ckpt not found: ${LAZY_CKPT}" >&2; exit 1; }

BRANCH="$(git branch --show-current)"
echo "branch: ${BRANCH}  (expect pipefusion_wan_cache_rpf)"
echo "result dir: ${RESULT_DIR}"

EXTRA_BODY="{\"lazy_enabled\":true,\"lazy_ckpt\":\"${LAZY_CKPT}\",\"lazy_threshold\":${LAZY_THRESHOLD},\"lazy_warmup_steps\":${LAZY_WARMUP_STEPS},\"lazy_log_stats\":true}"
export EXTRA_ARGS="--extra-body=${EXTRA_BODY}"
echo "extra args: ${EXTRA_ARGS}"

echo
echo "=== Phase 1/2: single-GPU baseline (GPU ${SINGLE_GPU}) ==="
SINGLE_GPU="${SINGLE_GPU}" MULTI_GPUS="${MULTI_GPUS}" \
  RESULT_DIR="${RESULT_DIR}" MODES="single" \
  bash benchmarks/diffusion/run_wan22_parallel_compare.sh "${CAPTIONS}" single

echo
echo "=== Phase 2/2: rotational PipeFusion + EasyCache (GPUs ${MULTI_GPUS}) ==="
SINGLE_GPU="${SINGLE_GPU}" MULTI_GPUS="${MULTI_GPUS}" \
  RESULT_DIR="${RESULT_DIR}" \
  PF_PIPELINE_SIZE="${PF_PIPELINE_SIZE}" PF_WARMUP_STEPS="${PF_WARMUP_STEPS}" \
  MODES="rotational" \
  bash benchmarks/diffusion/run_wan22_parallel_compare.sh "${CAPTIONS}" rotational

echo
echo "=== Metrics: PSNR / SSIM / FVD vs single-GPU baseline ==="
python benchmarks/diffusion/wan22_parallel_metrics.py --results-dir "${RESULT_DIR}"

echo
echo "All done."
echo "Report:   ${RESULT_DIR}/report.md"
echo "Perf TSV: ${RESULT_DIR}/perf.tsv"
