#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Generate the same Wan2.2 TI2V-5B prompts with several parallel backends and
# record perf, then score every mode against the single-GPU baseline with
# PSNR / SSIM / FVD.
#
# Usage:
#   source env.sh
#   bash benchmarks/diffusion/run_wan22_parallel_compare.sh [captions.txt] [modes...]
#
#   captions.txt : one prompt per line (optional; PROMPTS_FILE env also works).
#   modes        : any of "single sp pipefusion rotational"
#                  (default: "single pipefusion rotational" =
#                   1-GPU baseline, 4-GPU PipeFusion, 4-GPU Rotational PipeFusion)
#
# Examples:
#   bash benchmarks/diffusion/run_wan22_parallel_compare.sh captions.txt
#   bash benchmarks/diffusion/run_wan22_parallel_compare.sh captions.txt "single pipefusion rotational"
#   MODES="single sp" bash benchmarks/diffusion/run_wan22_parallel_compare.sh
#   SKIP_METRICS=1 bash benchmarks/diffusion/run_wan22_parallel_compare.sh
#
# Key environment overrides:
#   RESULT_DIR=...          Where logs/videos/reports go (default: timestamped).
#   SEED=42                 Fixed seed used for every prompt in every mode.
#   SINGLE_GPU=5            Device used by the "single" baseline.
#   MULTI_GPUS=0,1,2,3      Devices used by sp / pipefusion / rotational.
#   SP_DEGREE=2             Ulysses degree for "sp".
#   PF_PIPELINE_SIZE=4      Pipeline (PipeFusion) size for pipefusion / rotational.
#   PF_WARMUP_STEPS=2       PipeFusion warmup steps (shared by both PF modes).
#   WIDTH HEIGHT FRAMES NUM_INFERENCE_STEPS   Shape / budget of each video.
#   PROMPTS_FILE=path       One prompt per line (same as the captions.txt arg).
#   EXTRA_ARGS="--enforce-eager"  Extra flags appended to every run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/env.sh"
fi
cd "${REPO_ROOT}"

MODEL="${MODEL:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"

WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-704}"
FRAMES="${FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
GUIDANCE_SCALE_HIGH="${GUIDANCE_SCALE_HIGH:-6.0}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
FLOW_SHIFT="${FLOW_SHIFT:-12.0}"
FPS="${FPS:-16}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-low quality, blurry}"
SEED="${SEED:-42}"

SINGLE_GPU="${SINGLE_GPU:-5}"
MULTI_GPUS="${MULTI_GPUS:-0,1,2,3}"
SP_DEGREE="${SP_DEGREE:-2}"
PF_PIPELINE_SIZE="${PF_PIPELINE_SIZE:-4}"
PF_WARMUP_STEPS="${PF_WARMUP_STEPS:-2}"
PF_SPLIT_DIM="${PF_SPLIT_DIM:-temporal}"

RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmarks/diffusion/results/wan22_parallel_accuracy_$(date +%Y%m%d_%H%M%S)}"
VIDEO_DIR="${VIDEO_DIR:-${RESULT_DIR}/videos}"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/logs}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULT_DIR}/perf.tsv}"
SKIP_METRICS="${SKIP_METRICS:-0}"

# First positional argument may be a captions file (one prompt per line).
if [[ $# -gt 0 && -f "$1" ]]; then
  PROMPTS_FILE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
  shift
fi

if [[ -n "${MODES:-}" ]]; then
  RUN_MODES="${MODES}"
else
  RUN_MODES="${1:-single pipefusion rotational}"
fi

EXTRA_ARGS=(${EXTRA_ARGS:-})

DEFAULT_PROMPTS=(
  "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion."
  "A red sports car driving along a rainy neon city street at night, cinematic reflections."
  "A golden retriever running through shallow ocean waves at sunset, joyful smooth motion."
)

PROMPTS=()
SEEDS=()
if [[ -n "${PROMPTS_FILE:-}" ]]; then
  idx=0
  while IFS= read -r prompt; do
    [[ -z "${prompt}" ]] && continue
    PROMPTS+=("${prompt}")
    SEEDS+=("${SEED}")
    idx=$((idx + 1))
  done < "${PROMPTS_FILE}"
else
  PROMPTS=("${DEFAULT_PROMPTS[@]}")
  SEEDS=("${SEED}" "${SEED}" "${SEED}")
fi

if [[ "${#PROMPTS[@]}" -eq 0 ]]; then
  echo "No prompts configured." >&2
  exit 1
fi

for mode in ${RUN_MODES}; do
  case "${mode}" in
    single | sp | pipefusion | rotational | rpf) ;;
    *)
      echo "Unknown mode '${mode}'. Use any of: single sp pipefusion rotational." >&2
      exit 1
      ;;
  esac
done

mkdir -p "${VIDEO_DIR}" "${LOG_DIR}"
if [[ ! -f "${SUMMARY_PATH}" ]]; then
  printf "mode\tdegree\tprompt_id\ttotal_time_s\tdiffuse_time_s\tvae_decode_s\tforward_time_s\tpeak_mem_gib\tprompt\tvideo_path\tlog_path\n" \
    > "${SUMMARY_PATH}"
fi

GIT_COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

{
  echo "git_commit_sha: ${GIT_COMMIT_SHA}"
  echo "model: ${MODEL}"
  echo "shape: ${WIDTH}x${HEIGHT}x${FRAMES}"
  echo "num_inference_steps: ${NUM_INFERENCE_STEPS}"
  echo "guidance_scale: ${GUIDANCE_SCALE}"
  echo "guidance_scale_high: ${GUIDANCE_SCALE_HIGH}"
  echo "boundary_ratio: ${BOUNDARY_RATIO}"
  echo "flow_shift: ${FLOW_SHIFT}"
  echo "modes: ${RUN_MODES}"
  echo "seed: ${SEED}"
  echo "single_gpu: ${SINGLE_GPU}"
  echo "multi_gpus: ${MULTI_GPUS}"
  echo "sp_degree: ${SP_DEGREE}"
  echo "pf_pipeline_size: ${PF_PIPELINE_SIZE}"
  echo "pf_warmup_steps: ${PF_WARMUP_STEPS}"
  echo "pf_split_dim: ${PF_SPLIT_DIM}"
  echo "prompts: ${#PROMPTS[@]}"
  echo "started_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo unknown)"
} > "${RESULT_DIR}/meta.txt"

echo "Git commit SHA: ${GIT_COMMIT_SHA}"
echo "Model: ${MODEL}"
echo "Result directory: ${RESULT_DIR}"
echo "Modes: ${RUN_MODES}"
echo "Shape: ${WIDTH}x${HEIGHT}x${FRAMES} @ ${NUM_INFERENCE_STEPS} steps"
echo "Prompts: ${#PROMPTS[@]} (fixed seed ${SEED})"
echo "single -> GPU ${SINGLE_GPU}; parallel -> GPUs ${MULTI_GPUS}"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"

# Warn early (do not fail) if a selected GPU already holds a lot of memory.
declare -A GPU_USED_MB=()
if command -v nvidia-smi >/dev/null 2>&1; then
  while IFS=, read -r idx used; do
    GPU_USED_MB["${idx// /}"]="${used// /}"
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
fi
for devices in "${SINGLE_GPU}" "${MULTI_GPUS}"; do
  IFS=',' read -r -a device_list <<<"${devices}"
  for idx in "${device_list[@]}"; do
    used="${GPU_USED_MB[$idx]:-0}"
    if [[ "${used}" -gt 8000 ]]; then
      echo "WARNING: GPU ${idx} already has ${used} MiB in use; consider MULTI_GPUS/SINGLE_GPU overrides." >&2
    fi
  done
done

# Fail fast when a mode requests more parallel degrees than MULTI_GPUS provides;
# otherwise the engine only reports this deep inside worker startup.
IFS=',' read -r -a _multi_devices <<<"${MULTI_GPUS}"
needs_sp=0
needs_pf=0
for mode in ${RUN_MODES}; do
  case "${mode}" in
    sp) needs_sp=1 ;;
    pipefusion | rotational | rpf) needs_pf=1 ;;
  esac
done
if (( needs_sp && SP_DEGREE > ${#_multi_devices[@]} )); then
  echo "ERROR: SP_DEGREE=${SP_DEGREE} needs more devices than MULTI_GPUS=${MULTI_GPUS} provides (${#_multi_devices[@]})." >&2
  exit 1
fi
if (( needs_pf && PF_PIPELINE_SIZE > ${#_multi_devices[@]} )); then
  echo "ERROR: PF_PIPELINE_SIZE=${PF_PIPELINE_SIZE} needs more devices than MULTI_GPUS=${MULTI_GPUS} provides (${#_multi_devices[@]})." >&2
  exit 1
fi

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_' |
    sed 's/^_//; s/_$//; s/__*/_/g' | cut -c1-60
}

run_case() {
  local mode="$1"
  local prompt_id="$2"
  local prompt="$3"
  local seed="$4"

  local degree=1
  local devices="${SINGLE_GPU}"
  local -a mode_args=()

  case "${mode}" in
    single)
      devices="${SINGLE_GPU}"
      mode_args=()
      ;;
    sp)
      degree="${SP_DEGREE}"
      devices="${MULTI_GPUS}"
      mode_args=(
        --ulysses-degree "${degree}"
        --vae-patch-parallel-size "${degree}"
      )
      ;;
    pipefusion)
      degree="${PF_PIPELINE_SIZE}"
      devices="${MULTI_GPUS}"
      mode_args=(
        --pipeline-parallel-size "${degree}"
        --vae-patch-parallel-size "${degree}"
        --enable-pipefusion
        --pipefusion-warmup-steps "${PF_WARMUP_STEPS}"
        --pipefusion-split-dim "${PF_SPLIT_DIM}"
      )
      ;;
    rotational | rpf)
      mode="rotational"
      degree="${PF_PIPELINE_SIZE}"
      devices="${MULTI_GPUS}"
      mode_args=(
        --pipeline-parallel-size "${degree}"
        --vae-patch-parallel-size "${degree}"
        --enable-pipefusion
        --pipefusion-warmup-steps "${PF_WARMUP_STEPS}"
        --pipefusion-split-dim "${PF_SPLIT_DIM}"
        --enable-rotational-pipefusion
      )
      ;;
  esac

  local prompt_id_tag
  prompt_id_tag="$(printf 'p%02d' "${prompt_id}")"
  local stem="${mode}_${prompt_id_tag}_$(slugify "${prompt}")"
  local video_path="${VIDEO_DIR}/${stem}.mp4"
  local log_path="${LOG_DIR}/${stem}.log"

  echo
  echo "============================================================"
  echo "mode=${mode} degree=${degree} prompt=${prompt_id_tag}"
  echo "CUDA_VISIBLE_DEVICES=${devices}"
  echo "video: ${video_path}"
  echo "log:   ${log_path}"
  echo "============================================================"

  {
    echo "mode: ${mode}"
    echo "degree: ${degree}"
    echo "prompt_id: ${prompt_id_tag}"
    echo "seed: ${seed}"
    echo "CUDA_VISIBLE_DEVICES: ${devices}"
    echo "parallel args: ${mode_args[*]:-<none>}"
    echo "extra args: ${EXTRA_ARGS[*]:-<none>}"
    echo "git: ${GIT_COMMIT_SHA}"
  } > "${log_path}"

  local status=0
  env CUDA_VISIBLE_DEVICES="${devices}" python \
    "${REPO_ROOT}/examples/offline_inference/text_to_video/text_to_video.py" \
    --model="${MODEL}" \
    --width="${WIDTH}" \
    --height="${HEIGHT}" \
    --num-frames "${FRAMES}" \
    --guidance-scale="${GUIDANCE_SCALE}" \
    --guidance-scale-high="${GUIDANCE_SCALE_HIGH}" \
    --boundary-ratio="${BOUNDARY_RATIO}" \
    --flow-shift="${FLOW_SHIFT}" \
    --fps "${FPS}" \
    --prompt="${prompt}" \
    --negative-prompt="${NEGATIVE_PROMPT}" \
    --output="${video_path}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --seed "${seed}" \
    --enable-diffusion-pipeline-profiler \
    --vae-use-tiling \
    "${mode_args[@]}" \
    "${EXTRA_ARGS[@]}" >>"${log_path}" 2>&1 || status=$?

  if [[ "${status}" -ne 0 || ! -f "${video_path}" ]]; then
    echo "Run failed (exit=${status}, video_present=$([[ -f "${video_path}" ]] && echo yes || echo no)): ${stem}"
    tail -n 30 "${log_path}" || true
    return 1
  fi

  python - "${log_path}" "${SUMMARY_PATH}" "${mode}" "${degree}" "${prompt_id_tag}" \
    "${prompt}" "${video_path}" <<'PY'
import re
import sys
from pathlib import Path

(
    log_path,
    summary_path,
    mode,
    degree,
    prompt_id,
    prompt,
    video_path,
) = sys.argv[1:]

text = Path(log_path).read_text(errors="replace")
# Engine initialization runs a warmup generation pass whose profiler lines also land in
# this log, and every PP/SP rank logs its own copy of each timer. Restrict parsing to the
# measured request window and take the slowest rank, so modes stay comparable.
window_start = text.find("Processed prompts")
if window_start != -1:
    text = text[window_start:]


def last(pattern: str) -> str:
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""


def slowest_rank(pattern: str) -> str:
    values = [float(v) for v in re.findall(pattern, text)]
    return f"{max(values):.6f}" if values else ""


total_time = last(r"Total generation time:\s*([0-9.]+)\s+seconds")
diffuse_time = slowest_rank(r"Wan22Pipeline\.diffuse took\s*([0-9.]+)s")
vae_decode_time = slowest_rank(r"Wan22Pipeline\.vae\.decode took\s*([0-9.]+)s")
forward_times = [float(v) for v in re.findall(r"Wan22Pipeline\.forward took\s*([0-9.]+)s", text)]
forward_time = f"{max(forward_times):.6f}" if forward_times else ""
peak_mem = last(r"Worker peak GPU memory \(reserved\):.*\(([0-9.]+)\s+GiB\)")

row = [
    mode,
    degree,
    prompt_id,
    total_time,
    diffuse_time,
    vae_decode_time,
    forward_time,
    peak_mem,
    prompt.replace("\t", " "),
    video_path,
    str(log_path),
]
with Path(summary_path).open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY

  echo "finished ${stem}"
}

FAILED=0
for mode in ${RUN_MODES}; do
  for prompt_index in "${!PROMPTS[@]}"; do
    prompt_id=$((prompt_index + 1))
    if ! run_case "${mode}" "${prompt_id}" "${PROMPTS[${prompt_index}]}" "${SEEDS[${prompt_index}]}"; then
      echo "mode=${mode} prompt=${prompt_id} failed; continuing." >&2
      FAILED=1
    fi
  done
done

echo
echo "Generation finished. Perf summary: ${SUMMARY_PATH}"
if [[ -f "${video_path:-}" ]]; then
  echo "Videos: ${VIDEO_DIR}"
fi

if [[ "${SKIP_METRICS}" != "1" ]]; then
  echo
  if [[ " ${RUN_MODES} " == *" single "* ]]; then
    echo "Computing PSNR / SSIM / FVD against the single-GPU baseline..."
    if ! python "${SCRIPT_DIR}/wan22_parallel_metrics.py" --results-dir "${RESULT_DIR}"; then
      echo "Metric computation failed; inspect ${RESULT_DIR}." >&2
      FAILED=1
    fi
  else
    echo "Skipping PSNR / SSIM / FVD: baseline mode 'single' was not part of '${RUN_MODES}'."
  fi
fi

if [[ "${FAILED}" -ne 0 ]]; then
  echo "Some runs failed; inspect ${LOG_DIR}." >&2
  exit 1
fi
