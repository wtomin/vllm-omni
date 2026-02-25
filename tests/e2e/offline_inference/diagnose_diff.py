#!/usr/bin/env python3
"""诊断图像差异 - 找出具体哪些图片差异大"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def diff_metrics(img_ref: Image.Image, img_test: Image.Image) -> tuple[float, float]:
    """返回 (mean_abs_diff, max_abs_diff)，范围 [0, 1]"""
    a = np.asarray(img_ref, dtype=np.float32) / 255.0
    b_img = img_test
    if img_ref.size != img_test.size:
        b_img = img_test.resize(img_ref.size, Image.BILINEAR)
    b = np.asarray(b_img, dtype=np.float32) / 255.0
    diff = np.abs(a - b)
    return float(diff.mean()), float(diff.max())


def analyze_differences(results_dir: Path, config_name: str):
    """分析具体哪些图片差异大"""
    
    results_dir = Path(results_dir)
    baseline_dir = results_dir / "baseline"
    config_dir = results_dir / config_name
    
    if not baseline_dir.exists():
        print(f"错误: baseline 目录不存在: {baseline_dir}")
        return 1
    
    if not config_dir.exists():
        print(f"错误: 配置目录不存在: {config_dir}")
        return 1
    
    print(f"\n{'='*80}")
    print(f"诊断图像差异: {config_name}")
    print(f"{'='*80}\n")
    
    # 收集所有图片的差异
    idx = 0
    diffs = []
    
    while True:
        baseline_img_path = baseline_dir / f"prompt_{idx:02d}.png"
        config_img_path = config_dir / f"prompt_{idx:02d}.png"
        
        if not baseline_img_path.exists():
            break
        
        baseline_img = load_image(baseline_img_path)
        config_img = load_image(config_img_path)
        
        if baseline_img is None or config_img is None:
            print(f"⚠️  Prompt {idx:02d}: 图片加载失败")
            idx += 1
            continue
        
        mean_diff, max_diff = diff_metrics(baseline_img, config_img)
        diffs.append({
            "idx": idx,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "baseline_path": str(baseline_img_path),
            "config_path": str(config_img_path),
        })
        
        idx += 1
    
    if not diffs:
        print("没有找到可对比的图片")
        return 1
    
    # 排序并显示
    print(f"找到 {len(diffs)} 张图片\n")
    
    # 按 max_diff 排序
    sorted_by_max = sorted(diffs, key=lambda x: x["max_diff"], reverse=True)
    
    print("按最大像素差异排序（前 10 张）:")
    print("-" * 80)
    print(f"{'#':<4} {'Prompt':<10} {'MeanDiff':<12} {'MaxDiff':<12} {'状态':<8}")
    print("-" * 80)
    
    for rank, d in enumerate(sorted_by_max[:10], 1):
        status = "❌ 大" if d["max_diff"] > 0.2 else "✅ 小"
        print(f"{rank:<4} prompt_{d['idx']:02d}   {d['mean_diff']:<12.6f} {d['max_diff']:<12.6f} {status}")
    
    print("\n" + "=" * 80)
    print("统计信息:")
    print("=" * 80)
    
    mean_diffs = [d["mean_diff"] for d in diffs]
    max_diffs = [d["max_diff"] for d in diffs]
    
    print(f"平均差异 (MeanDiff):")
    print(f"  最小值: {min(mean_diffs):.6f}")
    print(f"  最大值: {max(mean_diffs):.6f}")
    print(f"  平均值: {np.mean(mean_diffs):.6f}")
    print(f"  中位数: {np.median(mean_diffs):.6f}")
    
    print(f"\n最大差异 (MaxDiff):")
    print(f"  最小值: {min(max_diffs):.6f}")
    print(f"  最大值: {max(max_diffs):.6f}")
    print(f"  平均值: {np.mean(max_diffs):.6f}")
    print(f"  中位数: {np.median(max_diffs):.6f}")
    
    # 统计超过阈值的图片
    high_max_diff = [d for d in diffs if d["max_diff"] > 0.2]
    high_mean_diff = [d for d in diffs if d["mean_diff"] > 0.02]
    
    print(f"\n超过阈值的图片:")
    print(f"  MaxDiff > 0.2:  {len(high_max_diff)} / {len(diffs)} ({len(high_max_diff)/len(diffs)*100:.1f}%)")
    print(f"  MeanDiff > 0.02: {len(high_mean_diff)} / {len(diffs)} ({len(high_mean_diff)/len(diffs)*100:.1f}%)")
    
    if high_max_diff:
        print(f"\nMaxDiff 超过阈值的图片索引: {[d['idx'] for d in high_max_diff]}")
    
    # 保存详细报告
    report_path = results_dir / f"diff_diagnosis_{config_name}.json"
    with open(report_path, "w") as f:
        json.dump({
            "config_name": config_name,
            "total_images": len(diffs),
            "statistics": {
                "mean_diff": {
                    "min": float(min(mean_diffs)),
                    "max": float(max(mean_diffs)),
                    "avg": float(np.mean(mean_diffs)),
                    "median": float(np.median(mean_diffs)),
                },
                "max_diff": {
                    "min": float(min(max_diffs)),
                    "max": float(max(max_diffs)),
                    "avg": float(np.mean(max_diffs)),
                    "median": float(np.median(max_diffs)),
                },
            },
            "high_diff_images": [
                {
                    "idx": d["idx"],
                    "mean_diff": d["mean_diff"],
                    "max_diff": d["max_diff"],
                }
                for d in sorted_by_max if d["max_diff"] > 0.2
            ],
            "all_diffs": diffs,
        }, f, indent=2)
    
    print(f"\n详细报告已保存到: {report_path}")
    print("=" * 80 + "\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="诊断具体哪些图片差异大",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 诊断 cfg_parallel 配置
  python diagnose_diff.py --results-dir ./compat_results/cfg_parallel --config cfg_parallel
  
  # 对比多个配置
  python diagnose_diff.py --results-dir ./compat_results/cfg_parallel --config cfg_parallel
  python diagnose_diff.py --results-dir ./compat_results/cfg_parallel --config cfg_parallel+teacache
""",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="结果目录路径（包含 baseline/ 和配置目录）",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="要诊断的配置名称（如 cfg_parallel）",
    )
    
    args = parser.parse_args()
    return analyze_differences(Path(args.results_dir), args.config)


if __name__ == "__main__":
    sys.exit(main())
