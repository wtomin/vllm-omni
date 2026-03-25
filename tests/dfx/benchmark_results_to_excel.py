#!/usr/bin/env python3
r"""
Read benchmark JSON result files and generate an Excel spreadsheet.

Usage:
    python benchmark_results_to_excel.py

Description:
    This script reads benchmark result JSON files from a user-specified directory,
    extracts performance metrics, and generates a summary Excel spreadsheet.

Input:
    - The script will prompt you to enter the path to the results directory.
    - The directory should contain JSON files with benchmark results.
    - Supported backend types (determined from filenames): sglang_diffusion, vllm_omni

Output:
    - The script will prompt you to specify the output Excel file path.
    - If not specified, defaults to '<results_dir>/benchmark_results_summary.xlsx'.
    - The Excel file contains the following columns:
        * backend: The backend type (sglang_diffusion or vllm_omni)
        * benchmark_params: The name field from benchmark_params
        * test_name(server_params): The test name
        * throughput_qps: Queries per second
        * latency_mean: Mean latency
        * latency_median: Median latency
        * latency_p99: P99 latency
        * latency_p95: P95 latency
        * latency_p50: P50 latency
        * peak_memory_mb_max: Maximum peak memory usage (MB)
        * peak_memory_mb_mean: Mean peak memory usage (MB)
        * peak_memory_mb_median: Median peak memory usage (MB)

Example:
    $ python benchmark_results_to_excel.py
    Please enter the results directory path: \path\to\results

    Scanning directory: \path\to\results

    Found X JSON file(s):
      - benchmark_results_test_sglang_diffusion_xxx.json
      - benchmark_results_test_vllm_omni_xxx.json

    Please enter the output Excel file path (press Enter to use default):
    Default: \path\to\results\benchmark_results_summary.xlsx
    Your choice: \path\to\report.xlsx

    Success! Excel file saved to: \path\to\report.xlsx

Dependencies:
    - pandas
    - openpyxl

    Install dependencies with:
        pip install pandas openpyxl
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


def extract_records(data, backend_type):
    """Extract records from JSON data."""
    records = []

    for item in data:
        # Extract name from benchmark_params
        benchmark_name = item.get("benchmark_params", {}).get("name", "")

        # Extract test_name
        test_name = item.get("test_name", "")

        # Extract performance metrics from result
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

        # Determine backend type based on filename
        if "sglang_diffusion" in filename:
            backend_type = "sglang_diffusion"
        elif "vllm_omni" in filename:
            backend_type = "vllm_omni"
        else:
            print(f"Warning: Cannot determine backend type for {filename}, skipping...")
            continue

        # Load and parse data
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

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Define column order
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
            max_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
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
