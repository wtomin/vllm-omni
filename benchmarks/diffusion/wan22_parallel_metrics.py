#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Score parallel Wan2.2 runs against a single-GPU baseline.

Reads the ``perf.tsv`` produced by ``run_wan22_parallel_compare.sh`` and reports:

* performance per mode (total / diffusion / VAE time, peak memory, speedup),
* frame-level accuracy against the baseline video of the same prompt (PSNR, SSIM),
* FVD between each mode and the baseline, using clip-level I3D features so that
  a handful of videos still yields a usable sample count.

Outputs (in the results directory): ``report.md``, ``metrics_per_video.csv``,
``metrics_fvd.csv`` and ``metrics.json``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAME_METRICS_PATH = REPO_ROOT / "examples" / "offline_inference" / "image_to_video" / "video_frame_metrics.py"


def load_frame_metrics():
    spec = importlib.util.spec_from_file_location("video_frame_metrics", FRAME_METRICS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frame metrics helper from {FRAME_METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["video_frame_metrics"] = module
    spec.loader.exec_module(module)
    return module


frame_metrics = load_frame_metrics()


def read_meta(results_dir: Path | None) -> dict[str, str]:
    if results_dir is None:
        return {}
    meta_path = results_dir / "meta.txt"
    if not meta_path.exists():
        return {}
    meta: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing perf.tsv, videos/ and logs/ (from run_wan22_parallel_compare.sh).",
    )
    parser.add_argument("--summary", type=Path, default=None, help="Explicit path to a perf.tsv.")
    parser.add_argument("--baseline", default="single", help="Baseline mode name. Default: single.")
    parser.add_argument("--skip-modes", default="", help="Comma separated modes to leave out of the report.")
    parser.add_argument("--clip-frames", type=int, default=16, help="Frames per clip fed to I3D for FVD.")
    parser.add_argument("--clip-stride", type=int, default=8, help="Stride between FVD clips.")
    parser.add_argument("--fvd-batch-size", type=int, default=4, help="Clips per I3D forward pass.")
    parser.add_argument("--device", default="cuda", help="Device for I3D feature extraction.")
    parser.add_argument(
        "--i3d-path",
        type=Path,
        default=None,
        help="Local torchscript I3D checkpoint. Downloads flateon/FVD-I3D-torchscript when omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to --results-dir.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on compared frames per video.")
    return parser.parse_args()


def read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("video_path")]
    for row in rows:
        for key in ("total_time_s", "diffuse_time_s", "vae_decode_s", "forward_time_s", "peak_mem_gib"):
            value = (row.get(key) or "").strip()
            row[key] = float(value) if value else None
        row["degree"] = int(row.get("degree") or 1)
    return rows


def load_frames(path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def video_pair_metrics(reference_path: Path, compare_path: Path, max_frames: int | None) -> dict[str, Any]:
    reference = load_frames(reference_path, max_frames)
    compare = load_frames(compare_path, max_frames)

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    abs_diff: list[float] = []
    for ref_frame, cmp_frame in zip(reference, compare):
        if ref_frame.shape != cmp_frame.shape:
            import cv2

            cmp_frame = cv2.resize(cmp_frame, (ref_frame.shape[1], ref_frame.shape[0]), interpolation=cv2.INTER_AREA)
        psnr_values.append(frame_metrics.compute_psnr(ref_frame, cmp_frame))
        ssim_values.append(frame_metrics.compute_ssim(ref_frame, cmp_frame))
        abs_diff.append(float(np.mean(np.abs(ref_frame.astype(np.float32) - cmp_frame.astype(np.float32)))))

    if not psnr_values:
        raise RuntimeError(f"No overlapping frames between {reference_path} and {compare_path}")

    finite_psnr = [value for value in psnr_values if math.isfinite(value)]
    return {
        "frames": len(psnr_values),
        "psnr_mean": float(np.mean(finite_psnr)) if finite_psnr else math.inf,
        "psnr_min": float(np.min(finite_psnr)) if finite_psnr else math.inf,
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_min": float(np.min(ssim_values)),
        "mae_mean": float(np.mean(abs_diff)),
        "identical_frames": int(sum(1 for value in psnr_values if math.isinf(value))),
    }


def load_i3d(path: Path | None, device: str):
    import torch
    from huggingface_hub import hf_hub_download

    if path is None:
        path = Path(hf_hub_download(repo_id="flateon/FVD-I3D-torchscript", filename="i3d_torchscript.pt"))
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    return model.to(device).eval(), torch.device(device)


def extract_features(
    video_path: Path,
    model,
    device,
    *,
    clip_frames: int,
    clip_stride: int,
    batch_size: int,
    max_clips: int | None = None,
) -> np.ndarray:
    """Return one I3D logit vector per temporal clip of the video."""
    import cv2
    import torch

    frames = load_frames(video_path)
    total = len(frames)
    starts = list(range(0, max(total - clip_frames, 0) + 1, clip_stride))
    if not starts:
        starts = [0]
    if max_clips is not None:
        starts = starts[:max_clips]

    clips = np.stack(
        [
            np.stack(
                [
                    cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    for frame in frames[start : start + clip_frames]
                ]
            )
            for start in starts
        ]
    )  # (N, T, H, W, C) uint8 BGR
    if clips.shape[1] < clip_frames:  # pad short clips by repeating the last frame
        pad = np.repeat(clips[:, -1:], clip_frames - clips.shape[1], axis=1)
        clips = np.concatenate([clips, pad], axis=1)

    tensor = torch.from_numpy(clips[..., ::-1].copy()).permute(0, 4, 1, 2, 3).contiguous().float()

    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start : start + batch_size].to(device, non_blocking=True)
            output = model(batch, rescale=True, resize=False, return_features=True)
            features.append(output.detach().float().cpu().numpy())
    return np.concatenate(features, axis=0)


def frechet_distance(first: np.ndarray, second: np.ndarray, eps: float = 1e-6) -> float:
    from scipy import linalg

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("FVD expects (N, D) feature arrays.")

    mu_first = first.mean(axis=0)
    mu_second = second.mean(axis=0)
    cov_first = np.cov(first, rowvar=False)
    cov_second = np.cov(second, rowvar=False)
    cov_first = np.atleast_2d(cov_first) + np.eye(cov_first.shape[0]) * eps
    cov_second = np.atleast_2d(cov_second) + np.eye(cov_second.shape[0]) * eps

    diff = mu_first - mu_second
    product = cov_first @ cov_second
    covmean = None
    for attempt in range(5):
        try:
            result = linalg.sqrtm(product)
            candidate = result[0] if isinstance(result, tuple) else result
        except ValueError:
            candidate = None
        if candidate is not None and np.all(np.isfinite(candidate)):
            covmean = candidate
            break
        product = product + np.eye(product.shape[0]) * (eps * (10.0**attempt))
    if covmean is None:
        raise RuntimeError("sqrtm failed to converge while computing FVD.")
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_first) + np.trace(cov_second) - 2.0 * np.trace(covmean))


def format_number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return "nan"
        return round(value, 6)
    return value


def main() -> None:
    args = parse_args()

    if args.summary is None and args.results_dir is None:
        raise SystemExit("Pass either --results-dir or --summary.")
    summary_path = args.summary or (args.results_dir / "perf.tsv")  # type: ignore[operator]
    if not summary_path.exists():
        raise SystemExit(f"Summary not found: {summary_path}")
    output_dir = args.output_dir or args.results_dir or summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_modes = {mode.strip() for mode in args.skip_modes.split(",") if mode.strip()}
    rows = [row for row in read_summary(summary_path) if row["mode"] not in skip_modes]

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)

    if args.baseline not in by_mode:
        raise SystemExit(f"Baseline mode '{args.baseline}' missing from {summary_path}. Modes: {sorted(by_mode)}")

    baseline_videos = {row["prompt_id"]: Path(row["video_path"]) for row in by_mode[args.baseline]}
    modes = [mode for mode in sorted(by_mode) if mode != args.baseline]

    print(f"Baseline: {args.baseline} ({len(baseline_videos)} videos)")
    print(f"Comparing modes: {', '.join(modes) or '<none>'}")

    # ---- I3D features for every video (used by FVD) ----
    features: dict[tuple[str, str], np.ndarray] = {}
    fvd_rows: list[dict[str, Any]] = []
    if modes:
        print("Extracting I3D features for FVD...")
        model, device = load_i3d(args.i3d_path, args.device)
        for row in rows:
            key = (row["mode"], row["prompt_id"])
            if key in features:
                continue
            features[key] = extract_features(
                Path(row["video_path"]),
                model,
                device,
                clip_frames=args.clip_frames,
                clip_stride=args.clip_stride,
                batch_size=args.fvd_batch_size,
            )
            print(f"  {row['mode']} {row['prompt_id']}: {features[key].shape[0]} clips")

        baseline_features = np.concatenate([features[(args.baseline, pid)] for pid in sorted(baseline_videos)])
        for mode in modes:
            mode_features = np.concatenate(
                [features[(mode, row["prompt_id"])] for row in by_mode[mode] if (mode, row["prompt_id"]) in features]
            )
            fvd_rows.append(
                {
                    "mode": mode,
                    "baseline": args.baseline,
                    "fvd": frechet_distance(baseline_features, mode_features),
                    "baseline_samples": int(baseline_features.shape[0]),
                    "mode_samples": int(mode_features.shape[0]),
                    "feature_dim": int(baseline_features.shape[1]),
                    "clip_frames": args.clip_frames,
                    "clip_stride": args.clip_stride,
                }
            )

    # ---- frame-level accuracy against the baseline ----
    per_video_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["mode"] == args.baseline:
            continue
        prompt_id = row["prompt_id"]
        reference_path = baseline_videos.get(prompt_id)
        if reference_path is None:
            print(f"warning: no baseline video for {prompt_id}; skipping {row['mode']}")
            continue
        compare_path = Path(row["video_path"])
        if not compare_path.exists() or not reference_path.exists():
            print(f"warning: missing video for {row['mode']} {prompt_id}; skipping")
            continue
        metrics = video_pair_metrics(reference_path, compare_path, args.max_frames)
        per_video_rows.append(
            {
                "mode": row["mode"],
                "degree": row["degree"],
                "prompt_id": prompt_id,
                "prompt": row.get("prompt", ""),
                "reference": str(reference_path),
                "compare": str(compare_path),
                "total_time_s": row.get("total_time_s"),
                "diffuse_time_s": row.get("diffuse_time_s"),
                "vae_decode_s": row.get("vae_decode_s"),
                "peak_mem_gib": row.get("peak_mem_gib"),
                **metrics,
            }
        )

    # ---- aggregate performance ----
    perf_rows: list[dict[str, Any]] = []
    baseline_times = [
        row["total_time_s"] for row in by_mode[args.baseline] if row.get("total_time_s") is not None
    ]
    baseline_mean = float(np.mean(baseline_times)) if baseline_times else None
    for mode, mode_rows in sorted(by_mode.items()):
        times = [row["total_time_s"] for row in mode_rows if row.get("total_time_s") is not None]
        diffuses = [row["diffuse_time_s"] for row in mode_rows if row.get("diffuse_time_s") is not None]
        vae_decodes = [row["vae_decode_s"] for row in mode_rows if row.get("vae_decode_s") is not None]
        peak_mems = [row["peak_mem_gib"] for row in mode_rows if row.get("peak_mem_gib") is not None]
        mean_time = float(np.mean(times)) if times else None
        degree = mode_rows[0]["degree"]
        speedup = (baseline_mean / mean_time) if baseline_mean and mean_time else None
        efficiency = (speedup / degree) if speedup is not None and degree else None
        perf_rows.append(
            {
                "mode": mode,
                "degree": degree,
                "runs": len(mode_rows),
                "total_time_mean_s": mean_time,
                "total_time_min_s": float(np.min(times)) if times else None,
                "total_time_max_s": float(np.max(times)) if times else None,
                "diffuse_time_mean_s": float(np.mean(diffuses)) if diffuses else None,
                "vae_decode_mean_s": float(np.mean(vae_decodes)) if vae_decodes else None,
                "peak_mem_mean_gib": float(np.mean(peak_mems)) if peak_mems else None,
                "speedup_vs_single": speedup,
                "parallel_efficiency": efficiency,
            }
        )

    # ---- aggregate accuracy per mode ----
    accuracy_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_video_rows:
        grouped[row["mode"]].append(row)
    for mode, mode_rows in sorted(grouped.items()):
        accuracy_rows.append(
            {
                "mode": mode,
                "degree": mode_rows[0]["degree"],
                "videos": len(mode_rows),
                "frames": int(sum(row["frames"] for row in mode_rows)),
                "psnr_mean_db": float(np.mean([row["psnr_mean"] for row in mode_rows])),
                "psnr_min_db": float(np.min([row["psnr_min"] for row in mode_rows])),
                "ssim_mean": float(np.mean([row["ssim_mean"] for row in mode_rows])),
                "ssim_min": float(np.min([row["ssim_min"] for row in mode_rows])),
                "mae_mean": float(np.mean([row["mae_mean"] for row in mode_rows])),
                "identical_frames": int(sum(row["identical_frames"] for row in mode_rows)),
            }
        )

    fvd_by_mode = {row["mode"]: row for row in fvd_rows}

    # ---- persist ----
    write_csv(
        output_dir / "metrics_per_video.csv",
        per_video_rows,
        [
            "mode",
            "degree",
            "prompt_id",
            "total_time_s",
            "diffuse_time_s",
            "vae_decode_s",
            "peak_mem_gib",
            "frames",
            "psnr_mean",
            "psnr_min",
            "ssim_mean",
            "ssim_min",
            "mae_mean",
            "identical_frames",
            "prompt",
            "reference",
            "compare",
        ],
    )
    if fvd_rows:
        write_csv(
            output_dir / "metrics_fvd.csv",
            fvd_rows,
            ["mode", "baseline", "fvd", "baseline_samples", "mode_samples", "feature_dim", "clip_frames", "clip_stride"],
        )
    write_csv(
        output_dir / "perf_summary.csv",
        perf_rows,
        [
            "mode",
            "degree",
            "runs",
            "total_time_mean_s",
            "total_time_min_s",
            "total_time_max_s",
            "diffuse_time_mean_s",
            "vae_decode_mean_s",
            "peak_mem_mean_gib",
            "speedup_vs_single",
            "parallel_efficiency",
        ],
    )
    if accuracy_rows:
        write_csv(
            output_dir / "metrics_summary.csv",
            accuracy_rows,
            [
                "mode",
                "degree",
                "videos",
                "frames",
                "psnr_mean_db",
                "psnr_min_db",
                "ssim_mean",
                "ssim_min",
                "mae_mean",
                "identical_frames",
            ],
        )

    meta = read_meta(args.results_dir)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(summary_path),
        "baseline": args.baseline,
        "meta": meta,
        "perf": perf_rows,
        "accuracy": accuracy_rows,
        "fvd": fvd_rows,
        "per_video": [{**row, **{k: format_value(v) for k, v in row.items()}} for row in per_video_rows],
        "config": {
            "clip_frames": args.clip_frames,
            "clip_stride": args.clip_stride,
            "device": str(args.device),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- report ----
    lines: list[str] = []
    lines.append("# Wan2.2 parallel accuracy & performance report")
    lines.append("")
    lines.append(f"- summary: `{summary_path}`")
    lines.append(f"- baseline mode: `{args.baseline}`")
    lines.append(f"- generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    for key in ("git_commit_sha", "model", "shape", "num_inference_steps", "gpu", "modes"):
        if key in meta:
            lines.append(f"- {key}: `{meta[key]}`")
    lines.append("")
    lines.append("## Performance")
    lines.append("")
    lines.append(
        "| mode | gpus | mean total (s) | mean diffuse (s) | mean VAE decode (s) | peak mem (GiB) | speedup | efficiency |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in perf_rows:
        lines.append(
            f"| {row['mode']} | {row['degree']} | {format_number(row['total_time_mean_s'])} "
            f"| {format_number(row['diffuse_time_mean_s'])} | {format_number(row['vae_decode_mean_s'])} "
            f"| {format_number(row['peak_mem_mean_gib'], 2)} | {format_number(row['speedup_vs_single'], 3)} "
            f"| {format_number(row['parallel_efficiency'], 3)} |"
        )
    lines.append("")
    lines.append(
        "_`total` is end-to-end wall clock (`Total generation time`) and is what speedup is based on. "
        "`diffuse` / `VAE decode` are pipeline-profiler stage timers taken inside the measured request "
        "window (warmup pass excluded) as the slowest pipeline rank, so they stay comparable across "
        "modes; because stages overlap, they are not expected to sum to wall clock._"
    )
    lines.append("")
    lines.append(f"## Accuracy vs `{args.baseline}` (frame level)")
    lines.append("")
    lines.append("| mode | videos | frames | mean PSNR (dB) | min PSNR (dB) | mean SSIM | min SSIM | mean MAE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in accuracy_rows:
        lines.append(
            f"| {row['mode']} | {row['videos']} | {row['frames']} | {format_number(row['psnr_mean_db'], 3)} "
            f"| {format_number(row['psnr_min_db'], 3)} | {format_number(row['ssim_mean'], 5)} "
            f"| {format_number(row['ssim_min'], 5)} | {format_number(row['mae_mean'], 4)} |"
        )
    lines.append("")
    lines.append(
        f"## FVD vs `{args.baseline}` (I3D logits, {args.clip_frames}-frame clips, stride {args.clip_stride})"
    )
    lines.append("")
    if fvd_rows:
        lines.append("| mode | FVD | baseline samples | mode samples | feature dim |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in fvd_rows:
            lines.append(
                f"| {row['mode']} | {format_number(row['fvd'], 4)} | {row['baseline_samples']} "
                f"| {row['mode_samples']} | {row['feature_dim']} |"
            )
        lines.append("")
        lines.append(
            "_FVD is a distribution metric: the value compares clip feature statistics of the two sets, "
            "so lower is better but absolute numbers depend on the sample count._"
        )
    else:
        lines.append("_No non-baseline modes available; FVD skipped._")
    lines.append("")
    lines.append("## Per-video detail")
    lines.append("")
    lines.append("| mode | prompt | PSNR (dB) | SSIM | MAE | total (s) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in per_video_rows:
        lines.append(
            f"| {row['mode']} | {row['prompt_id']} | {format_number(row['psnr_mean'], 3)} "
            f"| {format_number(row['ssim_mean'], 5)} | {format_number(row['mae_mean'], 4)} "
            f"| {format_number(row.get('total_time_s'), 2)} |"
        )
    lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
