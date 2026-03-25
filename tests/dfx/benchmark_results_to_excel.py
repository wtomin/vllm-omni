#!/usr/bin/env python3
"""
读取 benchmark JSON 结果文件并生成 Excel 表格
"""

import json
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Please install it with: pip install pandas openpyxl")
    exit(1)


def load_json_file(file_path):
    """加载 JSON 文件并返回数据"""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def extract_backend_from_filename(filename):
    """从文件名中提取 backend 类型"""
    if "sglang_diffusion" in filename:
        return "sglang_diffusion"
    elif "vllm_omni" in filename:
        return "vllm_omni"
    else:
        return "unknown"


def extract_records(data, backend_type):
    """从 JSON 数据中提取记录"""
    records = []

    for item in data:
        # 提取 benchmark_params 中的 name
        benchmark_name = item.get("benchmark_params", {}).get("name", "")

        # 提取 test_name
        test_name = item.get("test_name", "")

        # 从 result 中提取性能指标
        result = item.get("result", {})

        record = {
            "backend": backend_type,
            "benchmark_params": benchmark_name,
            "test_name(server_params)": test_name,
            "throughput_qps": result.get("throughput_qps"),
            "latency_mean": result.get("latency_mean"),
            "latency_median": result.get("latency_median"),
            "latency_p99": result.get("latency_p99"),
            "latency_p95": result.get("latency_p95"),
            "latency_p50": result.get("latency_p50"),
            "peak_memory_mb_max": result.get("peak_memory_mb_max"),
            "peak_memory_mb_mean": result.get("peak_memory_mb_mean"),
            "peak_memory_mb_median": result.get("peak_memory_mb_median"),
        }
        records.append(record)

    return records


def main():
    # 结果目录路径
    results_dir = Path(r"C:\Users\didan\Downloads\results")

    # 查找 JSON 文件
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"Error: No JSON files found in {results_dir}")
        return

    print(f"Found {len(json_files)} JSON files:")
    for f in json_files:
        print(f"  - {f.name}")

    # 收集所有记录
    all_records = []

    for json_file in json_files:
        filename = json_file.name

        # 确定 backend 类型
        if "sglang_diffusion" in filename:
            backend_type = "sglang_diffusion"
        elif "vllm_omni" in filename:
            backend_type = "vllm_omni"
        else:
            print(f"Warning: Cannot determine backend type for {filename}, skipping...")
            continue

        # 加载并解析数据
        try:
            data = load_json_file(json_file)
            records = extract_records(data, backend_type)
            all_records.extend(records)
            print(f"  Extracted {len(records)} records from {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if not all_records:
        print("Error: No valid records extracted!")
        return

    # 创建 DataFrame
    df = pd.DataFrame(all_records)

    # 定义列顺序
    columns = [
        "backend",
        "benchmark_params",
        "test_name(server_params)",
        "throughput_qps",
        "latency_mean",
        "latency_median",
        "latency_p99",
        "latency_p95",
        "latency_p50",
        "peak_memory_mb_max",
        "peak_memory_mb_mean",
        "peak_memory_mb_median",
    ]

    df = df[columns]

    # 生成输出文件路径
    output_file = results_dir / "benchmark_results_summary.xlsx"

    # 保存为 Excel
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Benchmark Results", index=False)

        # 获取工作表以调整列宽
        worksheet = writer.sheets["Benchmark Results"]

        # 自动调整列宽
        for idx, col in enumerate(df.columns, 1):
            max_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[worksheet.cell(row=1, column=idx).column_letter].width = min(max_length, 50)

    print(f"\nSuccess! Excel file saved to: {output_file}")
    print(f"Total records: {len(all_records)}")

    # 打印预览
    print("\nData Preview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
