#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Score parallel Wan2.2 runs against a single-GPU baseline.

Reads the ``perf.tsv`` produced by ``run_wan22_parallel_compare.sh`` and reports:

* performance per mode (total / diffusion / VAE time, peak memory, speedup),
* per-video accuracy against the baseline video of the same prompt:
  PSNR from the pooled per-pixel MSE of the whole video (data range 1.0),
  SSIM from torchmetrics per frame averaged inside the video, then averaged
  across videos,
* FVD between each mode and the baseline with exactly one I3D clip per video
  (``fvd_num_frames`` frames sampled every ``fvd_frame_stride`` frames), so
  N videos give N samples per distribution.

Outputs (in the results directory): ``report.md``, ``metrics_per_video.csv``,
``metrics_fvd.csv`` and ``metrics.json``.

PSNR / SSIM / FVD follow ``evaluate_quality.py`` (MetricComputer, FVDComputer,
``_read_fvd_clip``, ``frechet_distance``) so numbers stay comparable with the
EasyCache / MagCache / D2Cache evaluation tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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
    parser.add_argument(
        "--fvd-num-frames",
        type=int,
        default=11,
        help="Frames per FVD clip; one clip per video. 11 @ stride 8 exactly spans an 81-frame Wan2.2 video (default: 11).",
    )
    parser.add_argument(
        "--fvd-frame-stride",
        type=int,
        default=8,
        help="Frame stride inside the FVD clip (default: 8).",
    )
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
    """Per-video PSNR / SSIM / MAE, aligned with ``evaluate_quality.evaluate_pair``.

    PSNR is ``-10 * log10(mse)`` with ``mse`` pooled over every pixel of the
    video on the [0, 1] scale (data range 1.0). SSIM is torchmetrics'
    ``structural_similarity_index_measure(data_range=1.0)`` per frame, averaged
    inside the video. MAE stays in 0-255 units and is only reported as extra
    context.
    """
    import cv2
    import torch

    try:
        from torchmetrics.functional.image import structural_similarity_index_measure
    except ImportError as error:
        raise RuntimeError("install torchmetrics to compute SSIM") from error

    reference = load_frames(reference_path, max_frames)
    compare = load_frames(compare_path, max_frames)
    if not reference or not compare:
        raise RuntimeError(f"No overlapping frames between {reference_path} and {compare_path}")

    squared_error = 0.0
    value_count = 0
    ssim_sum = 0.0
    abs_diff: list[float] = []
    identical_frames = 0
    frame_count = 0

    for ref_frame, cmp_frame in zip(reference, compare):
        if ref_frame.shape != cmp_frame.shape:
            cmp_frame = cv2.resize(
                cmp_frame, (ref_frame.shape[1], ref_frame.shape[0]), interpolation=cv2.INTER_AREA
            )

        ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        cmp_rgb = cv2.cvtColor(cmp_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        difference = ref_rgb - cmp_rgb
        frame_squared_error = float(np.square(difference).sum())
        squared_error += frame_squared_error
        value_count += difference.size
        if frame_squared_error == 0.0:
            identical_frames += 1

        abs_diff.append(float(np.mean(np.abs(ref_frame.astype(np.float32) - cmp_frame.astype(np.float32)))))
        ssim_sum += float(
            structural_similarity_index_measure(
                torch.from_numpy(cmp_rgb).permute(2, 0, 1)[None],
                torch.from_numpy(ref_rgb).permute(2, 0, 1)[None],
                data_range=1.0,
            ).item()
        )
        frame_count += 1

    if frame_count == 0:
        raise RuntimeError(f"No overlapping frames between {reference_path} and {compare_path}")

    mse = squared_error / value_count
    return {
        "frames": frame_count,
        "psnr": math.inf if mse == 0 else -10.0 * math.log10(mse),
        "ssim": ssim_sum / frame_count,
        "mae": float(np.mean(abs_diff)),
        "identical_frames": identical_frames,
    }


def load_i3d(path: Path | None, device: str):
    import torch
    from huggingface_hub import hf_hub_download

    if path is None:
        path = Path(hf_hub_download(repo_id="flateon/FVD-I3D-torchscript", filename="i3d_torchscript.pt"))
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    return model.to(device).eval(), torch.device(device)


def read_fvd_clip(path: Path, num_frames: int, frame_stride: int) -> np.ndarray:
    """Decode exactly ``num_frames`` RGB frames at indices 0, stride, 2*stride, ...

    Mirrors ``evaluate_quality._read_fvd_clip``: the video must contain at least
    ``(num_frames - 1) * frame_stride + 1`` decodable frames.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open FVD video {path}")
    selected = []
    wanted = {index * frame_stride for index in range(num_frames)}
    last_index = (num_frames - 1) * frame_stride
    try:
        for index in range(last_index + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                selected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(selected) != num_frames:
        required = last_index + 1
        raise ValueError(f"FVD video {path} has fewer than {required} decodable frames")
    return np.stack(selected)


def extract_features(
    video_path: Path,
    model,
    device,
    *,
    num_frames: int,
    frame_stride: int,
) -> np.ndarray:
    """Return exactly one I3D feature vector of shape (1, D) for the video."""
    import torch

    clip = read_fvd_clip(video_path, num_frames=num_frames, frame_stride=frame_stride)
    video = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    with torch.no_grad():
        output = model(video.to(device), rescale=True, resize=True, return_features=True)
    return output.detach().float().cpu().numpy().reshape(1, -1)


def frechet_distance(origin_features: np.ndarray, target_features: np.ndarray) -> float:
    """Fréchet distance between two sets of per-video I3D features.

    Same formulation as ``evaluate_quality.frechet_distance``: covariance
    factors scaled by ``1 / sqrt(n - 1)``, the cross term taken as the sum of
    singular values of ``origin_factor.T @ target_factor``, clamped at zero.
    Requires at least two videos on each side.
    """
    origin_features = np.asarray(origin_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    if origin_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("FVD features must be rank-2 arrays")
    if origin_features.shape[1] != target_features.shape[1]:
        raise ValueError("FVD feature dimensions do not match")
    if min(len(origin_features), len(target_features)) < 2:
        raise ValueError("FVD requires at least two videos in each distribution")

    origin_mean = origin_features.mean(axis=0)
    target_mean = target_features.mean(axis=0)
    origin_centered = origin_features - origin_mean
    target_centered = target_features - target_mean
    origin_factor = origin_centered.T / math.sqrt(len(origin_features) - 1)
    target_factor = target_centered.T / math.sqrt(len(target_features) - 1)
    covariance_overlap = np.linalg.svd(origin_factor.T @ target_factor, compute_uv=False).sum()
    mean_distance = np.square(origin_mean - target_mean).sum()
    distance = (
        mean_distance
        + np.square(origin_factor).sum()
        + np.square(target_factor).sum()
        - 2 * covariance_overlap
    )
    return max(float(distance), 0.0)


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
            try:
                features[key] = extract_features(
                    Path(row["video_path"]),
                    model,
                    device,
                    num_frames=args.fvd_num_frames,
                    frame_stride=args.fvd_frame_stride,
                )
            except (ValueError, RuntimeError, OSError) as error:
                print(f"warning: skipping {row['mode']} {row['prompt_id']} for FVD: {error}")
                continue
            print(f"  {row['mode']} {row['prompt_id']}: {features[key].shape[0]} feature(s)")

        def stack_features(mode: str) -> np.ndarray:
            arrays = [
                features[(mode, row["prompt_id"])]
                for row in by_mode[mode]
                if (mode, row["prompt_id"]) in features
            ]
            return np.concatenate(arrays) if arrays else np.empty((0, 0))

        baseline_features = stack_features(args.baseline)
        for mode in modes:
            mode_features = stack_features(mode)
            if min(baseline_features.shape[0], mode_features.shape[0]) < 2:
                print(
                    f"warning: FVD needs at least 2 videos per distribution "
                    f"(baseline={baseline_features.shape[0]}, {mode}={mode_features.shape[0]}); skipping {mode}"
                )
                continue
            fvd_rows.append(
                {
                    "mode": mode,
                    "baseline": args.baseline,
                    "fvd": frechet_distance(baseline_features, mode_features),
                    "baseline_samples": int(baseline_features.shape[0]),
                    "mode_samples": int(mode_features.shape[0]),
                    "feature_dim": int(baseline_features.shape[1]),
                    "fvd_num_frames": args.fvd_num_frames,
                    "fvd_frame_stride": args.fvd_frame_stride,
                }
            )

    # ---- per-video accuracy against the baseline ----
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
                "psnr_mean_db": float(np.mean([row["psnr"] for row in mode_rows])),
                "psnr_min_db": float(np.min([row["psnr"] for row in mode_rows])),
                "ssim_mean": float(np.mean([row["ssim"] for row in mode_rows])),
                "ssim_min": float(np.min([row["ssim"] for row in mode_rows])),
                "mae_mean": float(np.mean([row["mae"] for row in mode_rows])),
                "identical_frames": int(sum(row["identical_frames"] for row in mode_rows)),
            }
        )

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
            "psnr",
            "ssim",
            "mae",
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
            ["mode", "baseline", "fvd", "baseline_samples", "mode_samples", "feature_dim", "fvd_num_frames", "fvd_frame_stride"],
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
            "fvd_num_frames": args.fvd_num_frames,
            "fvd_frame_stride": args.fvd_frame_stride,
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
    lines.append(f"## Accuracy vs `{args.baseline}` (per video)")
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
        f"## FVD vs `{args.baseline}` (one I3D clip per video: {args.fvd_num_frames} frames, "
        f"stride {args.fvd_frame_stride})"
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
            f"_FVD is a distribution metric: one feature vector per video, so lower is better and the "
            f"absolute value depends on the sample count (here one sample per video) and on the clip "
            f"spec ({args.fvd_num_frames} frames @ stride {args.fvd_frame_stride}, "
            f"needing {(args.fvd_num_frames - 1) * args.fvd_frame_stride + 1} decodable frames; "
            f"the default 11 @ 8 spans an 81-frame video exactly)._"
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
            f"| {row['mode']} | {row['prompt_id']} | {format_number(row['psnr'], 3)} "
            f"| {format_number(row['ssim'], 5)} | {format_number(row['mae'], 4)} "
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
