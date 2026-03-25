#!/usr/bin/env python3
"""
Read benchmark result JSON files and generate an Excel summary.

Example:
    $ python benchmark_results_to_excel.py
    Please enter the results directory path: \\path\to\results

    Scanning directory: \\path\to\results

    Found X JSON file(s):
      - benchmark_results_test_sglang_diffusion_xxx.json
      - benchmark_results_test_vllm_omni_xxx.json

    Please enter the output Excel file path (press Enter to use default):
    Default: \\path\to\results\benchmark_results_summary.xlsx
    Your choice: \\path\to\report.xlsx

    Success! Excel file saved to: \\path\to\report.xlsx
"""

import json
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Please install it with: pip install pandas openpyxl")
    exit(1)


def load_json_file(file_path):
    """Load a JSON file and return its data."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


BASE_COLUMN_ORDER = [
    "Model",
    "Framework",
    "Hardware",
    "Deployment",
    "Task",
    "Dataset",
    "resolution",
    "Parallelism",
    "max_concurrency",
    "Cache",
    "Quantization",
    "offload",
    "compile",
    "Attn_backend",
    "num_inference_steps",
    "completed",
    "failed",
    "throughput_qps",
    "latency_mean",
    "latency_median",
    "latency_p99",
    "latency_p95",
    "latency_p50",
    "peak_memory_mb_max",
    "peak_memory_mb_mean",
    "peak_memory_mb_median",
    "commit_sha",
    "build_id",
    "build_url",
    "source_file",
]


def _infer_framework_from_filename(filename: str) -> str:
    if "sglang" in filename:
        return "sglang"
    if "vllm_omni" in filename:
        return "vllm-omni"
    return ""


def _resolution_from_params(params: dict) -> str:
    resolutions = []
    seen = set()
    width = params.get("width")
    height = params.get("height")
    if width and height:
        res = f"{width}x{height}"
        seen.add(res)
        resolutions.append(res)
    random_cfg = params.get("random-request-config")
    if isinstance(random_cfg, list):
        for item in random_cfg:
            if not isinstance(item, dict):
                continue
            rw = item.get("width")
            rh = item.get("height")
            if rw and rh:
                res = f"{rw}x{rh}"
                if res not in seen:
                    seen.add(res)
                    resolutions.append(res)
    return "|".join(resolutions)


def _record_to_row(item: dict, filename: str) -> dict:
    params = item.get("benchmark_params", {}) if isinstance(item.get("benchmark_params"), dict) else {}
    result = item.get("result", {}) if isinstance(item.get("result"), dict) else {}

    framework = item.get("Framework") or item.get("backend") or _infer_framework_from_filename(filename)
    completed = item.get("completed")
    if completed is None:
        completed = result.get("completed_requests", result.get("completed"))
    failed = item.get("failed")
    if failed is None:
        failed = result.get("failed_requests", result.get("failed"))

    row = {
        "Model": item.get("Model", ""),
        "Framework": framework,
        "Hardware": item.get("Hardware", ""),
        "Deployment": item.get("Deployment", ""),
        "Task": item.get("Task", params.get("task", "")),
        "Dataset": item.get("Dataset", params.get("dataset", "")),
        "resolution": item.get("resolution", _resolution_from_params(params)),
        "Parallelism": item.get("Parallelism", ""),
        "max_concurrency": item.get("max_concurrency", params.get("max-concurrency", "")),
        "Cache": item.get("Cache", ""),
        "Quantization": item.get("Quantization", ""),
        "offload": item.get("offload", ""),
        "compile": item.get("compile", ""),
        "Attn_backend": item.get("Attn_backend", ""),
        "num_inference_steps": item.get("num_inference_steps", params.get("num-inference-steps", "")),
        "completed": completed,
        "failed": failed,
        "throughput_qps": item.get("throughput_qps", result.get("throughput_qps")),
        "latency_mean": item.get("latency_mean", result.get("latency_mean")),
        "latency_median": item.get("latency_median", result.get("latency_median")),
        "latency_p99": item.get("latency_p99", result.get("latency_p99")),
        "latency_p95": item.get("latency_p95", result.get("latency_p95")),
        "latency_p50": item.get("latency_p50", result.get("latency_p50")),
        "peak_memory_mb_max": item.get("peak_memory_mb_max", result.get("peak_memory_mb_max")),
        "peak_memory_mb_mean": item.get("peak_memory_mb_mean", result.get("peak_memory_mb_mean")),
        "peak_memory_mb_median": item.get("peak_memory_mb_median", result.get("peak_memory_mb_median")),
        "commit_sha": item.get("commit_sha", ""),
        # Export-layer policy: force empty source/build provenance columns.
        "build_id": "",
        "build_url": "",
        "source_file": "",
    }

    # Stage durations are nested in `result` by default. Flatten them into
    # per-stage columns so they can be directly compared in Excel.
    stage_mean = item.get("stage_durations_mean", result.get("stage_durations_mean", {}))
    stage_p50 = item.get("stage_durations_p50", result.get("stage_durations_p50", {}))
    stage_p99 = item.get("stage_durations_p99", result.get("stage_durations_p99", {}))

    if not isinstance(stage_mean, dict):
        stage_mean = {}
    if not isinstance(stage_p50, dict):
        stage_p50 = {}
    if not isinstance(stage_p99, dict):
        stage_p99 = {}

    # Don't present raw JSON in excel report. Extract stage-wise
    stage_names = sorted(set(stage_mean.keys()) | set(stage_p50.keys()) | set(stage_p99.keys()))
    for stage in stage_names:
        row[f"{stage}_mean"] = stage_mean.get(stage)
        row[f"{stage}_p50"] = stage_p50.get(stage)
        row[f"{stage}_p99"] = stage_p99.get(stage)
    return row


def check_log_files(results_dir):
    """Check if there are log files in the results directory."""
    log_files = list(results_dir.glob("*.log"))
    log_files_in_subdir = list(results_dir.rglob("logs/*.log"))
    all_log_files = log_files + log_files_in_subdir

    if all_log_files:
        print(f"Found {len(all_log_files)} log file(s):")
        for log_file in all_log_files[:10]:  # Show first 10 log files
            print(f"  - {log_file.relative_to(results_dir)}")
        if len(all_log_files) > 10:
            print(f"  ... and {len(all_log_files) - 10} more")
    else:
        print("Warning: No log files found in the specified directory.")

    return all_log_files


def main():
    # Prompt user for the results directory path
    user_input = input("Please enter the results directory path: ").strip()

    if not user_input:
        print("Error: No path provided. Exiting.")
        return

    results_dir = Path(user_input)

    # Validate the directory exists
    if not results_dir.exists():
        print(f"Error: Directory '{results_dir}' does not exist.")
        return

    if not results_dir.is_dir():
        print(f"Error: '{results_dir}' is not a directory.")
        return

    print(f"\nScanning directory: {results_dir}")

    # Check for log files
    check_log_files(results_dir)
    print()

    # Find JSON files
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"Error: No JSON files found in {results_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s):")
    for f in json_files:
        print(f"  - {f.name}")

    # Collect all records
    all_records = []

    for json_file in json_files:
        filename = json_file.name

        # Load and parse data
        try:
            data = load_json_file(json_file)
            if not isinstance(data, list):
                print(f"Warning: {filename} is not a list of records, skipping...")
                continue
            records = [_record_to_row(item, filename) for item in data if isinstance(item, dict)]
            all_records.extend(records)
            print(f"  Extracted {len(records)} records from {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if not all_records:
        print("Error: No valid records extracted!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_records)

    for col in BASE_COLUMN_ORDER:
        if col not in df.columns:
            df[col] = ""

    # Preserve any additional metrics (e.g., stage durations) instead of
    # dropping them when reordering columns.
    extra_columns = [c for c in df.columns if c not in BASE_COLUMN_ORDER]
    df = df[BASE_COLUMN_ORDER + extra_columns]

    # Prompt user for output file path
    default_output = results_dir / "benchmark_results_summary.xlsx"
    print("\nPlease enter the output Excel file path (press Enter to use default):")
    print(f"Default: {default_output}")
    output_input = input("Your choice: ").strip()

    if output_input:
        output_file = Path(output_input)
        # Ensure parent directory exists
        output_parent = output_file.parent
        if output_parent and not output_parent.exists():
            try:
                output_parent.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {output_parent}")
            except Exception as e:
                print(f"Error creating directory {output_parent}: {e}")
                print(f"Falling back to default: {default_output}")
                output_file = default_output
    else:
        output_file = default_output

    # Save to Excel
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Benchmark Results", index=False)

        # Get worksheet to adjust column widths
        worksheet = writer.sheets["Benchmark Results"]

        # Auto-adjust column widths
        for idx, col in enumerate(df.columns, 1):
            cell_lengths = [len(str(v)) for v in df[col].tolist()]
            max_length = max(max(cell_lengths, default=0), len(col)) + 2
            worksheet.column_dimensions[worksheet.cell(row=1, column=idx).column_letter].width = min(max_length, 50)

    # Resolve absolute path for display
    output_file_display = output_file.resolve()
    print(f"\nSuccess! Excel file saved to: {output_file_display}")
    print(f"Total records: {len(all_records)}")

    # Print preview
    print("\nData Preview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
