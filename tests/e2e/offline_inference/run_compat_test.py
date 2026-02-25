#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""run_compat_test.py — Feature compatibility test runner (Python)

Uses batch_text_to_image.py for efficient batch processing with a structured 
two-level test matrix:

  Baseline 1  : pure baseline — no acceleration features
  Baseline 2  : --baseline-feature enabled alone
  Addon 1..N  : Baseline-2 arguments + one addon feature each

Each run saves:
  <output_dir>/<baseline_feature>/baseline/batch_generation.log
  <output_dir>/<baseline_feature>/baseline/batch_generation.exitcode
  <output_dir>/<baseline_feature>/baseline/prompt_NN.png
  <output_dir>/<baseline_feature>/baseline/prompt_NN.exitcode
  <output_dir>/<baseline_feature>/<config_name>/batch_generation.log
  <output_dir>/<baseline_feature>/<config_name>/prompt_NN.{png,exitcode}

Note: Individual prompt_NN.log files are not created; all logs are in batch_generation.log

Usage:
  python run_compat_test.py \\
      --baseline-feature cfg_parallel \\
      --addons teacache cache_dit ulysses ring \\
      --model Qwen/Qwen-Image-2512 \\
      --output-dir ./compat_results \\
      --steps 30 \\
      --num-prompts 20 \\
      --prompt-file ./prompts.txt

  python analyze_compat_results.py \\
      --results-dir ./compat_results/cfg_parallel \\
      --charts

Extending the feature registry:
  Add a new entry to FEATURE_REGISTRY with:
    args            : list of CLI tokens passed to batch_text_to_image.py
    gpu_multiplier  : parallel dimension (1 = no extra GPUs, 2 = doubles GPU count, …)
    lossy           : True if the feature may degrade image quality (cache-based methods)
    label           : human-readable display name for reports/charts
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ── Feature registry ──────────────────────────────────────────────────────────
#
# gpu_multiplier: how many GPUs this feature alone needs.
#   Combined GPU count = product of all active features' multipliers.
# lossy: True if this feature trades quality for speed (cache methods).

FEATURE_REGISTRY: dict[str, dict] = {
    "cfg_parallel": {
        "args": ["--cfg-parallel-size", "2"],
        "gpu_multiplier": 2,
        "lossy": False,
        "label": "CFG-Parallel (×2)",
    },
    "teacache": {
        "args": ["--cache-backend", "tea_cache"],
        "gpu_multiplier": 1,
        "lossy": True,
        "label": "TeaCache",
    },
    "cache_dit": {
        "args": ["--cache-backend", "cache_dit"],
        "gpu_multiplier": 1,
        "lossy": True,
        "label": "Cache-DiT",
    },
    "ulysses": {
        "args": ["--ulysses-degree", "2"],
        "gpu_multiplier": 2,
        "lossy": False,
        "label": "Ulysses SP (×2)",
    },
    "ring": {
        "args": ["--ring-degree", "2"],
        "gpu_multiplier": 2,
        "lossy": False,
        "label": "Ring SP (×2)",
    },
    "tp": {
        "args": ["--tensor-parallel-size", "2"],
        "gpu_multiplier": 2,
        "lossy": False,
        "label": "Tensor Parallel (×2)",
    },
}

NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark, noise"

# Default prompt file bundled alongside this script
_DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "prompts.txt"


def load_prompts(prompt_file: str | Path) -> list[str]:
    """Load prompts from a text file (one prompt per line, blank lines ignored)."""
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [ln.strip() for ln in lines if ln.strip()]
    if not prompts:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompts


# ── Helper utilities ──────────────────────────────────────────────────────────


def _log(msg: str, level: str = "INFO") -> None:
    colors = {
        "INFO": "\033[36m",
        "OK": "\033[32m",
        "WARN": "\033[33m",
        "FAIL": "\033[31m",
    }
    reset = "\033[0m"
    c = colors.get(level, "")
    print(f"{c}[{level:4s}]{reset}  {msg}")


def get_gpu_count() -> int:
    """Return the number of CUDA GPUs visible to PyTorch."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


# ── Config builder ────────────────────────────────────────────────────────────


def build_configs(baseline_feature: str, addons: list[str]) -> list[dict]:
    """Return the ordered list of test configs for the given baseline + addons.

    Parameters
    ----------
    baseline_feature:
        Key from FEATURE_REGISTRY used as the primary feature under test.
    addons:
        Zero or more additional feature keys to stack on top of the baseline.

    Returns
    -------
    list[dict]  ordered: baseline1 → baseline2 → addon-1 … addon-N
    """
    unknown = [f for f in [baseline_feature, *addons] if f not in FEATURE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}\nAvailable: {sorted(FEATURE_REGISTRY)}")

    bf = FEATURE_REGISTRY[baseline_feature]
    configs: list[dict] = []

    # Baseline 1 — no features
    configs.append(
        {
            "name": "baseline",
            "role": "baseline1",
            "features": [],
            "args": [],
            "gpu_req": 1,
            "lossy": False,
        }
    )

    # Baseline 2 — baseline feature only
    configs.append(
        {
            "name": baseline_feature,
            "role": "baseline2",
            "features": [baseline_feature],
            "args": list(bf["args"]),
            "gpu_req": bf["gpu_multiplier"],
            "lossy": bf["lossy"],
        }
    )

    # Addons — baseline feature + one addon each
    for addon in addons:
        af = FEATURE_REGISTRY[addon]
        gpu_req = bf["gpu_multiplier"] * af["gpu_multiplier"]
        configs.append(
            {
                "name": f"{baseline_feature}+{addon}",
                "role": "addon",
                "features": [baseline_feature, addon],
                "args": list(bf["args"]) + list(af["args"]),
                "gpu_req": gpu_req,
                "lossy": bf["lossy"] or af["lossy"],
            }
        )

    return configs


# ── I/O helpers ───────────────────────────────────────────────────────────────


def write_manifest(
    row_dir: Path,
    baseline_feature: str,
    addons: list[str],
    configs: list[dict],
    args: argparse.Namespace,
    gpu_count: int,
) -> None:
    manifest = {
        "baseline_feature": baseline_feature,
        "addons": addons,
        "model": args.model,
        "output_dir": str(row_dir),
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_prompts": args.num_prompts,
        "gpu_count": gpu_count,
        "configs": [c["name"] for c in configs],
        "created_at": datetime.now().isoformat(),
    }
    manifest_path = row_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written → {manifest_path}")


def write_config_meta(
    cfg_dir: Path,
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    meta = {
        "name": cfg["name"],
        "role": cfg["role"],
        "features": cfg["features"],
        "extra_args": " ".join(cfg["args"]),
        "total_gpus": cfg["gpu_req"],
        "is_lossy": cfg["lossy"],
        "model": args.model,
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_prompts": args.num_prompts,
    }
    (cfg_dir / "config_info.json").write_text(json.dumps(meta, indent=2))


# ── Per-prompt runner ─────────────────────────────────────────────────────────


def run_batch_prompts(
    batch_script: Path,
    cfg: dict,
    prompt_file: Path,
    num_prompts: int,
    cfg_dir: Path,
    args: argparse.Namespace,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Run batch_text_to_image.py for all prompts; return (n_ok, n_fail)."""
    # Use cfg_dir as output directory for batch generation
    cmd = [
        sys.executable,
        str(batch_script),
        "--model",
        args.model,
        "--prompt-file",
        str(prompt_file),
        "--negative-prompt",
        f"'{NEGATIVE_PROMPT}'",
        "--output",
        str(cfg_dir),
        "--num-inference-steps",
        str(args.steps),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--cfg-scale",
        str(args.cfg_scale),
        "--seed",
        str(args.seed),
        *cfg["args"],
    ]

    print(f"  Running batch generation for {num_prompts} prompts...", flush=True)

    if dry_run:
        print(f"    DRY-RUN: {' '.join(cmd)}")
        return num_prompts, 0

    # Run batch generation
    t0 = time.monotonic()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed_ms = (time.monotonic() - t0) * 1_000
    rc = result.returncode

    # Save batch log
    batch_log_path = cfg_dir / "batch_generation.log"
    batch_log_path.write_bytes(result.stdout)
    
    # Save batch exit code
    batch_rc_path = cfg_dir / "batch_generation.exitcode"
    batch_rc_path.write_text(str(rc))

    if rc != 0:
        _log(f"BATCH FAIL (rc={rc})", "FAIL")
        tail = result.stdout.decode(errors="replace").splitlines()[-10:]
        for ln in tail:
            print(f"      │ {ln}")
        return 0, num_prompts
    
    # Rename generated images from image_NNNN.png to prompt_NN.png
    # and create minimal metadata files for compatibility with analyze script
    n_ok = 0
    n_fail = 0
    for idx in range(num_prompts):
        src_img = cfg_dir / f"image_{idx:04d}.png"
        dst_img = cfg_dir / f"prompt_{idx:02d}.png"
        rc_path = cfg_dir / f"prompt_{idx:02d}.exitcode"
        
        if src_img.exists():
            src_img.rename(dst_img)
            rc_path.write_text("0")
            n_ok += 1
        else:
            rc_path.write_text("1")
            n_fail += 1

    _log(f"Batch completed: {n_ok} OK / {n_fail} FAIL in {elapsed_ms:.0f}ms", "OK" if n_fail == 0 else "WARN")
    
    return n_ok, n_fail


# ── Per-config runner ─────────────────────────────────────────────────────────


def run_config(
    batch_script: Path,
    cfg: dict,
    prompt_file: Path,
    num_prompts: int,
    cfg_dir: Path,
    args: argparse.Namespace,
    dry_run: bool,
) -> tuple[int, int]:
    """Run all prompts for one config using batch generation; return (n_ok, n_fail)."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    write_config_meta(cfg_dir, cfg, args)

    sep = "─" * 72
    print(f"\n{sep}")
    _log(f"Config : {cfg['name']}  [{cfg['role']}]")
    _log(f"Args   : {' '.join(cfg['args']) or '(none)'}")
    _log(f"GPUs   : {cfg['gpu_req']}  |  Lossy: {cfg['lossy']}")
    print(sep)

    # Use batch processing instead of per-prompt processing
    n_ok, n_fail = run_batch_prompts(batch_script, cfg, prompt_file, num_prompts, cfg_dir, args, dry_run)

    _log(f"Config '{cfg['name']}': {n_ok} OK / {n_fail} FAIL / {num_prompts} total")
    return n_ok, n_fail


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Feature compatibility test runner — Python replacement for run_compat_test.sh.\n\n"
            "Runs text_to_image.py with a structured two-level matrix:\n"
            "  Baseline 1 = no features,  Baseline 2 = --baseline-feature alone,\n"
            "  Addons = Baseline-2 + each feature in --addons."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available features for --baseline-feature / --addons:\n"
            + "\n".join(f"  {name:<14} {info['label']}" for name, info in FEATURE_REGISTRY.items())
            + "\n\nExample:\n"
            "  python run_compat_test.py \\\n"
            "      --baseline-feature cfg_parallel \\\n"
            "      --addons teacache cache_dit ulysses ring \\\n"
            "      --model Qwen/Qwen-Image-2512 \\\n"
            "      --output-dir ./compat_results \\\n"
            "      --steps 30 --num-prompts 20\n\n"
            "  python analyze_compat_results.py \\\n"
            "      --results-dir ./compat_results/cfg_parallel --charts"
        ),
    )
    p.add_argument(
        "--baseline-feature",
        default="cfg_parallel",
        choices=sorted(FEATURE_REGISTRY),
        metavar="FEATURE",
        help=(
            "Primary feature under test.  "
            "Baseline-1 = no features; Baseline-2 = this feature alone.  "
            f"(default: cfg_parallel; choices: {sorted(FEATURE_REGISTRY)})"
        ),
    )
    p.add_argument(
        "--addons",
        nargs="*",
        default=[],
        choices=sorted(FEATURE_REGISTRY),
        metavar="FEATURE",
        help=(
            "Zero or more features stacked on top of --baseline-feature "
            "(e.g. teacache cache_dit ulysses ring). "
            "Each creates one additional config: <baseline>+<addon>."
        ),
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen-Image-2512",
        help="HuggingFace model ID or local path.",
    )
    p.add_argument(
        "--output-dir",
        default="compat_results",
        help="Root directory for images and logs.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Diffusion denoising steps.",
    )
    p.add_argument("--height", type=int, default=1024, help="Image height in pixels.")
    p.add_argument("--width", type=int, default=1024, help="Image width in pixels.")
    p.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="true_cfg_scale forwarded to text_to_image.py.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    p.add_argument(
        "--prompt-file",
        default=str(_DEFAULT_PROMPT_FILE),
        help=("Path to a text file with one prompt per line. (default: prompts.txt next to this script)"),
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=20,
        help="Number of prompts to use (taken from the top of --prompt-file).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()

    # Locate batch_text_to_image.py relative to this script's repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]  # tests/e2e/offline_inference → repo root
    batch_script = repo_root / "examples" / "offline_inference" / "text_to_image" / "batch_text_to_image.py"
    if not batch_script.exists():
        print(f"[ERROR] batch_text_to_image.py not found at: {batch_script}", file=sys.stderr)
        return 1

    # Validate and build config matrix
    try:
        configs = build_configs(args.baseline_feature, args.addons or [])
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        all_prompts = load_prompts(args.prompt_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    num_prompts = max(1, min(args.num_prompts, len(all_prompts)))
    prompts = all_prompts[:num_prompts]
    gpu_count = get_gpu_count()
    row_dir = Path(args.output_dir) / args.baseline_feature
    row_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary prompt file with only the prompts we need
    # This file will be shared across all configs to avoid redundant file creation
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        for prompt in prompts:
            tf.write(prompt + "\n")
        temp_prompt_file = Path(tf.name)

    try:
        # Header
        print()
        print("=" * 72)
        print("  Feature Compatibility Test (Batch Mode)")
        print(f"  Baseline feature : {args.baseline_feature}")
        print(f"  Addons           : {args.addons or '(none)'}")
        print(f"  Model            : {args.model}")
        print(f"  GPUs available   : {gpu_count}")
        print(f"  Configs          : {len(configs)}")
        print(f"  Prompts          : {num_prompts}")
        print(f"  Steps            : {args.steps}  |  {args.width}×{args.height}")
        print(f"  Output dir       : {row_dir}")
        print("=" * 72)

        write_manifest(row_dir, args.baseline_feature, args.addons or [], configs, args, gpu_count)

        total_ok = total_fail = total_skip = 0

        for cfg in configs:
            if cfg["gpu_req"] > gpu_count:
                _log(
                    f"SKIP '{cfg['name']}' — requires {cfg['gpu_req']} GPUs, only {gpu_count} available",
                    "WARN",
                )
                total_skip += 1
                continue

            cfg_dir = row_dir / cfg["name"]
            n_ok, n_fail = run_config(batch_script, cfg, temp_prompt_file, num_prompts, cfg_dir, args, args.dry_run)
            total_ok += n_ok
            total_fail += n_fail
    finally:
        # Clean up temporary prompt file
        temp_prompt_file.unlink(missing_ok=True)

    # Final summary
    print()
    print("=" * 72)
    print("  RUN COMPLETE")
    print(f"  Baseline feature : {args.baseline_feature}")
    print(f"  OK               : {total_ok}")
    print(f"  FAIL             : {total_fail}")
    print(f"  SKIP             : {total_skip}  (configs, not prompts)")
    print(f"  Output           : {row_dir}")
    print("=" * 72)
    print()
    print("Next step — run the analysis script:")
    print(f"  python analyze_compat_results.py --results-dir {row_dir} --charts")
    print()

    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
