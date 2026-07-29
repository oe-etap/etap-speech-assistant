#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to aggregate and average performance metrics from latency log CSV files.

Parses the last X CSV run logs from the 'outputs' directory, averages timing metrics
(such as STT, TTFA, LLM evaluation, TTS, E2E response ready), resource usage, and extra
performance statistics, and generates a formatted text table suitable for Windows Notepad
as well as a pure tab-separated (TSV) file.
"""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Any, Tuple


def find_latest_csv_logs(outputs_dir: Path, count: int) -> List[Path]:
    """
    Find the latest X CSV log files within the outputs directory.

    Args:
        outputs_dir: Path to the outputs directory.
        count: Number of latest CSV logs to retrieve.

    Returns:
        List of Path objects for the selected CSV files sorted newest first.
    """
    if not outputs_dir.exists():
        print(f"Error: Directory '{outputs_dir}' does not exist.", file=sys.stderr)
        return []

    # Search for all latency_log_*.csv files recursively
    csv_files = list(outputs_dir.glob("**/latency_log_*.csv"))
    
    # Fallback to any .csv files if specific pattern yields nothing
    if not csv_files:
        csv_files = [p for p in outputs_dir.glob("**/*.csv") if not p.name.endswith("_summary.csv")]

    if not csv_files:
        print(f"Warning: No CSV log files found in '{outputs_dir}'.", file=sys.stderr)
        return []

    # Sort files by modification time descending (latest first)
    csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return csv_files[:count]


def parse_csv_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a single latency log CSV file and return a list of record dicts.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of parsed row dictionaries.
    """
    records = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception as e:
        print(f"Warning: Could not read '{file_path}': {e}", file=sys.stderr)
    return records


def aggregate_metrics(csv_files: List[Path]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], List[Dict[str, Any]]]:
    """
    Aggregate metrics across multiple CSV log files.

    Returns:
        Tuple containing:
        - stage_latencies: dict mapping stage name to list of duration_ms values
        - resource_metrics: dict mapping metric name to list of float values
        - extra_metrics: dict mapping metric name to list of float values
        - per_run_summaries: list of summary dicts for each individual run
    """
    stage_latencies: Dict[str, List[float]] = {}
    resource_metrics: Dict[str, List[float]] = {
        "cpu_percent": [],
        "ram_percent": [],
        "rss_mb": [],
        "gpu_util_percent": [],
        "gpu_mem_used_mb": [],
    }
    extra_metrics: Dict[str, List[float]] = {
        "stt_rtf": [],
        "tokens_per_sec": [],
    }

    per_run_summaries = []

    for file_path in csv_files:
        rows = parse_csv_file(file_path)
        if not rows:
            continue

        run_stage_latencies: Dict[str, List[float]] = {}

        for row in rows:
            stage = row.get("stage", "").strip()
            duration_str = row.get("duration_ms", "").strip()
            
            if stage and duration_str:
                try:
                    duration = float(duration_str)
                    stage_latencies.setdefault(stage, []).append(duration)
                    run_stage_latencies.setdefault(stage, []).append(duration)
                except ValueError:
                    pass

            # Parse resource metrics
            for res_key in resource_metrics.keys():
                val_str = row.get(res_key, "").strip()
                if val_str:
                    try:
                        resource_metrics[res_key].append(float(val_str))
                    except ValueError:
                        pass

            # Parse extra_json metrics
            extra_raw = row.get("extra_json", "").strip()
            if extra_raw:
                try:
                    extra_data = json.loads(extra_raw)
                    if isinstance(extra_data, dict):
                        if "stt_rtf" in extra_data and isinstance(extra_data["stt_rtf"], (int, float)):
                            extra_metrics["stt_rtf"].append(float(extra_data["stt_rtf"]))
                        if "tokens_per_sec" in extra_data and isinstance(extra_data["tokens_per_sec"], (int, float)):
                            extra_metrics["tokens_per_sec"].append(float(extra_data["tokens_per_sec"]))
                except json.JSONDecodeError:
                    pass

        # Calculate run averages
        run_avg = {"folder": file_path.parent.name, "file": file_path.name}
        for stg, val_list in run_stage_latencies.items():
            if val_list:
                run_avg[stg] = statistics.mean(val_list)
        per_run_summaries.append(run_avg)

    return stage_latencies, resource_metrics, extra_metrics, per_run_summaries


def format_summary_table(
    csv_files: List[Path],
    requested_count: int,
    stage_latencies: Dict[str, List[float]],
    resource_metrics: Dict[str, List[float]],
    extra_metrics: Dict[str, List[float]],
    per_run_summaries: List[Dict[str, Any]]
) -> str:
    """
    Format aggregated metrics into an aesthetic, Notepad-friendly tab-separated table text string.
    """
    lines = []
    divider_thick = "=" * 90
    divider_thin  = "-" * 90

    lines.append(divider_thick)
    lines.append("                              PERFORMANCE LOG AVERAGE SUMMARY")
    lines.append(divider_thick)
    lines.append(f"Generated On         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Requested Log Count  : {requested_count}")
    lines.append(f"Analyzed Log Count   : {len(csv_files)}")
    lines.append("")
    lines.append("Included Log Files:")
    for idx, path in enumerate(csv_files, 1):
        lines.append(f"  {idx}. {path}")
    lines.append(divider_thick)

    # 1. Stage Latencies Table
    lines.append("1. STAGE LATENCY METRICS (values in ms)")
    lines.append(divider_thick)
    # Header line with tab delimiter and padded column titles
    lines.append(f"{'Stage Name':<25}\t{'Average (ms)':>14}\t{'Min (ms)':>12}\t{'Max (ms)':>12}\t{'Samples':>8}")
    lines.append(divider_thin)

    # Preferred order of stages for clear reporting
    known_stages = ["stt", "llm_ttft", "llm_ttfc", "tts_first_chunk",
                    "ttfa_from_speech_end", "ttfa", "llm_eval", "tts_total",
                    "e2e_response_ready"]
    all_stages = known_stages + [s for s in stage_latencies.keys() if s not in known_stages]

    for stage in all_stages:
        values = stage_latencies.get(stage, [])
        if values:
            avg_val = statistics.mean(values)
            min_val = min(values)
            max_val = max(values)
            count_val = len(values)
            lines.append(
                f"{stage:<25}\t{avg_val:>14.2f}\t{min_val:>12.2f}\t{max_val:>12.2f}\t{count_val:>8}"
            )
        else:
            lines.append(f"{stage:<25}\t{'N/A':>14}\t{'N/A':>12}\t{'N/A':>12}\t{0:>8}")

    lines.append(divider_thick)
    lines.append("")

    # 2. Per-Run Breakdown Table
    if per_run_summaries:
        lines.append("2. PER-RUN BREAKDOWN (Averages per run in ms)")
        lines.append(divider_thick)
        run_headers = f"{'Run Directory':<25}\t{'stt (ms)':>12}\t{'llm_ttfc':>12}\t{'ttfa (ms)':>12}\t{'e2e_ready':>12}"
        lines.append(run_headers)
        lines.append(divider_thin)

        for run in per_run_summaries:
            folder_name = run["folder"]
            stt_str = f"{run.get('stt', 0.0):>12.2f}" if "stt" in run else f"{'N/A':>12}"
            ttfc_str = f"{run.get('llm_ttfc', 0.0):>12.2f}" if "llm_ttfc" in run else f"{'N/A':>12}"
            ttfa_str = f"{run.get('ttfa', 0.0):>12.2f}" if "ttfa" in run else f"{'N/A':>12}"
            e2e_str = f"{run.get('e2e_response_ready', 0.0):>12.2f}" if "e2e_response_ready" in run else f"{'N/A':>12}"
            lines.append(f"{folder_name:<25}\t{stt_str}\t{ttfc_str}\t{ttfa_str}\t{e2e_str}")

        lines.append(divider_thin)
        # Add Overall Average Row
        overall_stt = f"{statistics.mean(stage_latencies['stt']):>12.2f}" if "stt" in stage_latencies and stage_latencies["stt"] else f"{'N/A':>12}"
        overall_ttfc = f"{statistics.mean(stage_latencies['llm_ttfc']):>12.2f}" if "llm_ttfc" in stage_latencies and stage_latencies["llm_ttfc"] else f"{'N/A':>12}"
        overall_ttfa = f"{statistics.mean(stage_latencies['ttfa']):>12.2f}" if "ttfa" in stage_latencies and stage_latencies["ttfa"] else f"{'N/A':>12}"
        overall_e2e = f"{statistics.mean(stage_latencies['e2e_response_ready']):>12.2f}" if "e2e_response_ready" in stage_latencies and stage_latencies["e2e_response_ready"] else f"{'N/A':>12}"
        
        lines.append(f"{'OVERALL AVERAGE':<25}\t{overall_stt}\t{overall_ttfc}\t{overall_ttfa}\t{overall_e2e}")
        lines.append(divider_thick)
        lines.append("")

    # 3. System Resource Metrics Table
    has_resources = any(len(v) > 0 for v in resource_metrics.values())
    if has_resources:
        lines.append("3. SYSTEM RESOURCE USAGE")
        lines.append(divider_thick)
        lines.append(f"{'Resource Metric':<25}\t{'Average':>14}\t{'Min':>12}\t{'Max':>12}\t{'Samples':>8}")
        lines.append(divider_thin)

        resource_labels = {
            "cpu_percent": "CPU Usage (%)",
            "ram_percent": "RAM Usage (%)",
            "rss_mb": "RAM RSS (MB)",
            "gpu_util_percent": "GPU Util (%)",
            "gpu_mem_used_mb": "GPU Mem Used (MB)",
        }

        for key, label in resource_labels.items():
            vals = resource_metrics.get(key, [])
            if vals:
                avg_val = statistics.mean(vals)
                min_val = min(vals)
                max_val = max(vals)
                count_val = len(vals)
                lines.append(
                    f"{label:<25}\t{avg_val:>14.2f}\t{min_val:>12.2f}\t{max_val:>12.2f}\t{count_val:>8}"
                )
        lines.append(divider_thick)
        lines.append("")

    # 4. Extra Performance Metrics Table
    has_extra = any(len(v) > 0 for v in extra_metrics.values())
    if has_extra:
        lines.append("4. EXTRA METRICS (STT RTF & LLM Tokens/sec)")
        lines.append(divider_thick)
        lines.append(f"{'Extra Metric':<25}\t{'Average':>14}\t{'Min':>12}\t{'Max':>12}\t{'Samples':>8}")
        lines.append(divider_thin)

        extra_labels = {
            "stt_rtf": "STT Real Time Factor",
            "tokens_per_sec": "LLM Tokens/sec",
        }

        for key, label in extra_labels.items():
            vals = extra_metrics.get(key, [])
            if vals:
                avg_val = statistics.mean(vals)
                min_val = min(vals)
                max_val = max(vals)
                count_val = len(vals)
                lines.append(
                    f"{label:<25}\t{avg_val:>14.2f}\t{min_val:>12.2f}\t{max_val:>12.2f}\t{count_val:>8}"
                )
        lines.append(divider_thick)
        lines.append("")

    return "\n".join(lines)


def format_pure_tsv_table(stage_latencies: Dict[str, List[float]]) -> str:
    """
    Format stage latency averages into a simple, pure tab-separated table string.
    """
    lines = ["Stage\tAverage_ms\tMin_ms\tMax_ms\tSamples"]
    known_stages = ["stt", "llm_ttft", "llm_ttfc", "tts_first_chunk",
                    "ttfa_from_speech_end", "ttfa", "llm_eval", "tts_total",
                    "e2e_response_ready"]
    all_stages = known_stages + [s for s in stage_latencies.keys() if s not in known_stages]

    for stage in all_stages:
        values = stage_latencies.get(stage, [])
        if values:
            avg_val = statistics.mean(values)
            min_val = min(values)
            max_val = max(values)
            count_val = len(values)
            lines.append(f"{stage}\t{avg_val:.2f}\t{min_val:.2f}\t{max_val:.2f}\t{count_val}")
        else:
            lines.append(f"{stage}\tN/A\tN/A\tN/A\t0")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and average performance metrics from the last X run CSV logs in outputs folder.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--log-count",
        type=int,
        default=4,
        help="Number of latest CSV run log files to average (default: 4). Example: --log-count 4",
    )
    parser.add_argument(
        "-d",
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Path to the outputs directory containing run folders (default: 'outputs').",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=os.path.join("outputs", "log_averages_summary.txt"),
        help="Output text file path for saving formatted summary (default: 'outputs/log_averages_summary.txt').",
    )
    parser.add_argument(
        "-t",
        "--tsv-file",
        type=str,
        default=os.path.join("outputs", "log_averages.tsv"),
        help="Output TSV file path for saving tab-separated raw table (default: 'outputs/log_averages.tsv').",
    )

    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    requested_count = args.log_count

    if requested_count <= 0:
        print("Error: --log-count must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    csv_files = find_latest_csv_logs(outputs_dir, requested_count)

    if not csv_files:
        print(f"No log files found in '{outputs_dir}'. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing the last {len(csv_files)} CSV log file(s) from '{outputs_dir}'...")

    stage_latencies, resource_metrics, extra_metrics, per_run_summaries = aggregate_metrics(csv_files)

    table_text = format_summary_table(
        csv_files=csv_files,
        requested_count=requested_count,
        stage_latencies=stage_latencies,
        resource_metrics=resource_metrics,
        extra_metrics=extra_metrics,
        per_run_summaries=per_run_summaries,
    )

    tsv_text = format_pure_tsv_table(stage_latencies)

    # Print result to console
    print("\n" + table_text + "\n")

    # Save formatted summary text file
    output_path = Path(args.output_file)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(table_text)
        print(f"Summary text table saved to : {output_path.resolve()}")
    except Exception as e:
        print(f"Error writing output file '{output_path}': {e}", file=sys.stderr)

    # Save pure TSV table file
    tsv_path = Path(args.tsv_file)
    try:
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write(tsv_text)
        print(f"Tab-separated TSV table saved to : {tsv_path.resolve()}")
    except Exception as e:
        print(f"Error writing TSV file '{tsv_path}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
