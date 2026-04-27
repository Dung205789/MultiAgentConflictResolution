"""
Enhanced reporting and comparison for multi-agent memory evaluation.

Provides utilities to:
- Format evaluation results clearly
- Compare multiple evaluation runs
- Generate summary statistics
- Track improvements over time
"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime


def format_metrics_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Format metrics as a readable table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Mode':<15} {'Acc':<10} {'Prec':<10} {'Recall':<10} {'F1':<10} {'Action':<10}")
    lines.append("-" * 80)

    for mode in ["conflict_aware", "lww", "naive"]:
        if mode not in results:
            continue
        r = results[mode]
        acc = r.get("scenario_accuracy", 0.0)
        prec = r.get("conflict_precision", 0.0)
        rec = r.get("conflict_recall", 0.0)
        f1 = r.get("conflict_f1", 0.0)
        act = r.get("action_accuracy", 0.0)
        lines.append(f"{mode:<15} {acc:<10.3f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f} {act:<10.3f}")

    lines.append("=" * 80)
    return "\n".join(lines)


def format_deltas_table(deltas: Dict[str, Dict[str, float]]) -> str:
    """Format delta comparisons between modes."""
    lines = []
    lines.append("=" * 100)
    lines.append(f"{'Metric':<25} {'CA - LWW':<25} {'CA - Naive':<25} {'LWW - Naive':<25}")
    lines.append("-" * 100)

    for metric, values in deltas.items():
        ca_lww = values.get("conflict_aware_minus_lww", 0.0)
        ca_naive = values.get("conflict_aware_minus_naive", 0.0)
        lww_naive = values.get("lww_minus_naive", 0.0)
        lines.append(f"{metric:<25} {ca_lww:>+8.3f}     {ca_naive:>+8.3f}     {lww_naive:>+8.3f}")

    lines.append("=" * 100)
    return "\n".join(lines)


def format_per_type_breakdown(per_type: Dict[str, Dict[str, float]]) -> str:
    """Format per-scenario-type accuracy breakdown."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Scenario Type':<30} {'CA Acc':<10} {'LWW Acc':<10} {'Naive Acc':<10}")
    lines.append("-" * 80)

    for stype, metrics in sorted(per_type.items()):
        ca = metrics.get("conflict_aware_action_accuracy", 0.0)
        lww = metrics.get("lww_action_accuracy", 0.0)
        naive = metrics.get("naive_action_accuracy", 0.0)
        lines.append(f"{stype:<30} {ca:<10.3f} {lww:<10.3f} {naive:<10.3f}")

    lines.append("=" * 80)
    return "\n".join(lines)


def generate_summary_report(report: Dict[str, Any]) -> str:
    """Generate a human-readable summary report from evaluation results."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("EVALUATION SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append(f"Benchmark: {report.get('benchmark', 'Unknown')}")
    lines.append(f"Scenarios: {report.get('num_scenarios', 0)}")
    lines.append(f"Timestamp: {report.get('timestamp', 'N/A')}")
    lines.append("")

    if "results" in report:
        lines.append("--- Performance Metrics ---")
        lines.append(format_metrics_table(report["results"]))

    if "deltas" in report:
        lines.append("\n--- Improvement Deltas (Conflict-Aware vs Baselines) ---")
        lines.append(format_deltas_table(report["deltas"]))

    if "per_scenario_type" in report:
        lines.append("\n--- Action Accuracy by Scenario Type ---")
        lines.append(format_per_type_breakdown(report["per_scenario_type"]))

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def save_comparison_report(
    baseline_path: str,
    improved_path: str,
    output_path: str
) -> str:
    """
    Load two evaluation reports and generate a comparison.

    Args:
        baseline_path: Path to baseline (before) evaluation report
        improved_path: Path to improved (after) evaluation report
        output_path: Where to save the comparison report

    Returns:
        Path to saved report
    """
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    with open(improved_path, 'r') as f:
        improved = json.load(f)

    lines = []
    lines.append("=" * 100)
    lines.append("BEFORE vs AFTER COMPARISON REPORT")
    lines.append("=" * 100)
    lines.append(f"Baseline: {baseline_path}")
    lines.append(f"Improved: {improved_path}")
    lines.append("")

    # Compare metrics for each mode
    baseline_results = baseline.get("results", {})
    improved_results = improved.get("results", {})

    lines.append("--- Metric Changes by Mode ---")
    lines.append(f"{'Mode':<15} {'Metric':<25} {'Before':<12} {'After':<12} {'Delta':<12}")
    lines.append("-" * 100)

    for mode in ["conflict_aware", "lww", "naive"]:
        if mode not in baseline_results or mode not in improved_results:
            continue

        b = baseline_results[mode]
        i = improved_results[mode]

        metrics_to_compare = [
            ("scenario_accuracy", "Scenario Accuracy"),
            ("conflict_f1", "Conflict F1"),
            ("action_accuracy", "Action Accuracy"),
            ("conflict_precision", "Conflict Precision"),
            ("conflict_recall", "Conflict Recall"),
        ]

        for key, label in metrics_to_compare:
            before = b.get(key, 0.0)
            after = i.get(key, 0.0)
            delta = after - before
            lines.append(f"{mode:<15} {label:<25} {before:<12.3f} {after:<12.3f} {delta:>+12.3f}")

    lines.append("=" * 100)

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    return output_path


def export_json_report(report: Dict[str, Any], path: str) -> None:
    """Export full report as JSON for downstream processing."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Example: generate summary from latest report
    import sys
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
        with open(report_path, 'r') as f:
            report = json.load(f)
        print(generate_summary_report(report))
    else:
        print("Usage: python reporter.py <evaluation_report.json>")
