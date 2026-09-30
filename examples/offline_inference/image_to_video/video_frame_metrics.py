#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute frame-level PSNR and SSIM for same-name videos."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare same-name video files in two output directories and print "
            "frame-level PSNR/SSIM."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("outputs/before"),
        help="Directory containing reference videos. Default: outputs/before",
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=Path("outputs/after"),
        help="Directory containing videos to compare. Default: outputs/after",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search videos recursively under each directory.",
    )
    parser.add_argument(
        "--resize-compare",
        action="store_true",
        help="Resize compare frames to the reference resolution when sizes differ.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only compare the first N frames from each matched video.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print per-video and overall averages, not every frame.",
    )
    parser.add_argument(
        "--print-sha",
        action="store_true",
        help="Print SHA256 for each matched video before computing frame metrics.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Only print SHA256 equality for matched videos; skip frame metrics.",
    )
    return parser.parse_args()


def iter_video_files(root: Path, *, recursive: bool) -> Iterator[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def collect_by_name(root: Path, *, recursive: bool) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for path in iter_video_files(root, recursive=recursive):
        if path.name in videos:
            raise ValueError(
                f"Duplicate video name {path.name!r} under {root}. "
                "Use non-recursive mode or make file names unique."
            )
        videos[path.name] = path
    return videos


def read_frames(path: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_sha_summary(name: str, reference_path: Path, compare_path: Path) -> None:
    reference_sha = compute_sha256(reference_path)
    compare_sha = compute_sha256(compare_path)
    same_file = reference_sha == compare_sha
    print(f"\nvideo: {name}")
    print(f"reference_sha256: {reference_sha}")
    print(f"compare_sha256:   {compare_sha}")
    print(f"sha256_equal: {same_file}")


def compute_psnr(reference: np.ndarray, compare: np.ndarray) -> float:
    diff = reference.astype(np.float64) - compare.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(reference: np.ndarray, compare: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    compare = compare.astype(np.float64)

    height, width = reference.shape[:2]
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1

    if window_size < 3:
        return compute_global_ssim(reference, compare)

    kernel = (window_size, window_size)
    sigma = 1.5
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_ref = cv2.GaussianBlur(reference, kernel, sigma)
    mu_cmp = cv2.GaussianBlur(compare, kernel, sigma)
    mu_ref_sq = mu_ref * mu_ref
    mu_cmp_sq = mu_cmp * mu_cmp
    mu_ref_cmp = mu_ref * mu_cmp

    sigma_ref_sq = cv2.GaussianBlur(reference * reference, kernel, sigma) - mu_ref_sq
    sigma_cmp_sq = cv2.GaussianBlur(compare * compare, kernel, sigma) - mu_cmp_sq
    sigma_ref_cmp = cv2.GaussianBlur(reference * compare, kernel, sigma) - mu_ref_cmp

    numerator = (2.0 * mu_ref_cmp + c1) * (2.0 * sigma_ref_cmp + c2)
    denominator = (mu_ref_sq + mu_cmp_sq + c1) * (sigma_ref_sq + sigma_cmp_sq + c2)
    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def compute_global_ssim(reference: np.ndarray, compare: np.ndarray) -> float:
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_ref = float(np.mean(reference))
    mu_cmp = float(np.mean(compare))
    var_ref = float(np.var(reference))
    var_cmp = float(np.var(compare))
    cov = float(np.mean((reference - mu_ref) * (compare - mu_cmp)))
    numerator = (2.0 * mu_ref * mu_cmp + c1) * (2.0 * cov + c2)
    denominator = (mu_ref**2 + mu_cmp**2 + c1) * (var_ref + var_cmp + c2)
    return numerator / denominator


def format_metric(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def compare_video_pair(
    name: str,
    reference_path: Path,
    compare_path: Path,
    *,
    resize_compare: bool,
    max_frames: int | None,
    summary_only: bool,
) -> tuple[list[float], list[float]]:
    print(f"\nvideo: {name}")
    print(f"reference: {reference_path}")
    print(f"compare:   {compare_path}")
    if not summary_only:
        print("frame,psnr,ssim")

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    ref_frames = read_frames(reference_path)
    cmp_frames = read_frames(compare_path)

    for frame_idx, (ref_frame, cmp_frame) in enumerate(zip(ref_frames, cmp_frames)):
        if max_frames is not None and frame_idx >= max_frames:
            break

        if ref_frame.shape != cmp_frame.shape:
            if not resize_compare:
                raise ValueError(
                    f"Frame shape mismatch in {name} at frame {frame_idx}: "
                    f"reference={ref_frame.shape}, compare={cmp_frame.shape}. "
                    "Pass --resize-compare to resize compare frames."
                )
            cmp_frame = cv2.resize(
                cmp_frame,
                (ref_frame.shape[1], ref_frame.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        psnr = compute_psnr(ref_frame, cmp_frame)
        ssim = compute_ssim(ref_frame, cmp_frame)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        if not summary_only:
            print(f"{frame_idx},{format_metric(psnr)},{format_metric(ssim)}")

    if max_frames is not None and len(psnr_values) >= max_frames:
        ref_remaining = None
        cmp_remaining = None
    else:
        ref_remaining = next(ref_frames, None)
        cmp_remaining = next(cmp_frames, None)
    if ref_remaining is not None or cmp_remaining is not None:
        print(
            "warning: video lengths differ; metrics were computed for the "
            "overlapping frame range only."
        )

    if not psnr_values:
        print("warning: no frames compared")
        return psnr_values, ssim_values

    print_summary("summary", psnr_values, ssim_values)
    return psnr_values, ssim_values


def print_summary(label: str, psnr_values: list[float], ssim_values: list[float]) -> None:
    psnr_mean = float(np.mean(psnr_values))
    ssim_mean = float(np.mean(ssim_values))
    print(
        f"{label},"
        f"frames={len(psnr_values)},"
        f"mean_psnr={format_metric(psnr_mean)},"
        f"mean_ssim={format_metric(ssim_mean)}"
    )


def main() -> None:
    args = parse_args()

    reference_dir = args.reference_dir.resolve()
    compare_dir = args.compare_dir.resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
    if not compare_dir.is_dir():
        raise FileNotFoundError(f"Compare directory not found: {compare_dir}")

    reference_videos = collect_by_name(reference_dir, recursive=args.recursive)
    compare_videos = collect_by_name(compare_dir, recursive=args.recursive)
    matched_names = sorted(reference_videos.keys() & compare_videos.keys())
    only_reference = sorted(reference_videos.keys() - compare_videos.keys())
    only_compare = sorted(compare_videos.keys() - reference_videos.keys())

    if only_reference:
        print(f"warning: only in reference dir: {', '.join(only_reference)}")
    if only_compare:
        print(f"warning: only in compare dir: {', '.join(only_compare)}")
    if not matched_names:
        raise RuntimeError("No same-name videos found to compare.")

    all_psnr_values: list[float] = []
    all_ssim_values: list[float] = []
    for name in matched_names:
        if args.print_sha or args.sha_only:
            print_sha_summary(name, reference_videos[name], compare_videos[name])
        if args.sha_only:
            continue

        psnr_values, ssim_values = compare_video_pair(
            name,
            reference_videos[name],
            compare_videos[name],
            resize_compare=args.resize_compare,
            max_frames=args.max_frames,
            summary_only=args.summary_only,
        )
        all_psnr_values.extend(psnr_values)
        all_ssim_values.extend(ssim_values)

    if all_psnr_values:
        print()
        print_summary("overall_summary", all_psnr_values, all_ssim_values)


if __name__ == "__main__":
    main()
