#!/usr/bin/env python
"""
Unified benchmark runner for all supported benchmarks.

This script provides a single interface to run evaluations on:
- Custom synthetic benchmark (enhanced_multi_agent_benchmark.jsonl)
- MemoryAgentBench (MAB) from Hugging Face
- LongMemEval from Hugging Face
- LoCoMo from Hugging Face

Usage examples:
  python run_benchmarks.py custom
  python run_benchmarks.py mab --subset conflict
  python run_benchmarks.py longmemeval --subset s
  python run_benchmarks.py lococo --subset test
  python run_benchmarks.py all
"""
import sys
import os
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.run_evaluation import run_evaluation_with_scenarios


def load_custom_benchmark(path: str) -> list:
    """Load custom JSONL benchmark."""
    import json
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_custom_benchmark(benchmark_path: str = None, output_dir: str = "reports"):
    """Run evaluation on custom benchmark."""
    if not benchmark_path:
        benchmark_path = "data/enhanced_multi_agent_benchmark.jsonl"

    print(f"\n{'='*60}")
    print(f"Running Custom Benchmark")
    print(f"Source: {benchmark_path}")
    print(f"{'='*60}\n")

    scenarios = load_custom_benchmark(benchmark_path)
    if not scenarios:
        print(f"Error: No scenarios loaded from {benchmark_path}")
        return None

    output_path = os.path.join(output_dir, "custom_benchmark_report.json")
    report = run_evaluation_with_scenarios(
        scenarios=scenarios,
        benchmark_name=f"Custom-{os.path.basename(benchmark_path)}",
        output_path=output_path
    )
    if report:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.splitext(os.path.basename(benchmark_path))[0]
        snapshot_path = os.path.join(output_dir, f"custom_benchmark_report_{dataset_name}_{timestamp}.json")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Snapshot saved to: {snapshot_path}")
    return report


def run_memoryagentbench(subset: str = "all", num_samples: int = None, output_dir: str = "reports"):
    """Run evaluation on MemoryAgentBench."""
    print(f"\n{'='*60}")
    print(f"Running MemoryAgentBench")
    print(f"Subset: {subset}")
    if num_samples:
        print(f"Max samples: {num_samples}")
    print(f"{'='*60}\n")

    from src.benchmarks.memoryagentbench_loader import load_memoryagentbench
    scenarios = load_memoryagentbench(subset=subset, num_samples=num_samples)

    if not scenarios:
        print(f"Error: No scenarios loaded from MemoryAgentBench (subset={subset})")
        return None

    output_path = os.path.join(output_dir, f"mab_{subset}_report.json")
    report = run_evaluation_with_scenarios(
        scenarios=scenarios,
        benchmark_name=f"MemoryAgentBench-{subset}",
        output_path=output_path
    )
    return report


def run_longmemeval(subset: str = "all", num_samples: int = None, output_dir: str = "reports"):
    """Run evaluation on LongMemEval."""
    print(f"\n{'='*60}")
    print(f"Running LongMemEval")
    print(f"Subset: {subset}")
    if num_samples:
        print(f"Max samples: {num_samples}")
    print(f"{'='*60}\n")

    from src.benchmarks.longmemeval_loader import load_longmemeval
    scenarios = load_longmemeval(subset=subset, num_samples=num_samples)

    if not scenarios:
        print(f"Error: No scenarios loaded from LongMemEval (subset={subset})")
        return None

    output_path = os.path.join(output_dir, f"longmemeval_{subset}_report.json")
    report = run_evaluation_with_scenarios(
        scenarios=scenarios,
        benchmark_name=f"LongMemEval-{subset}",
        output_path=output_path
    )
    return report


def run_lococo(subset: str = "test", num_samples: int = None, output_dir: str = "reports"):
    """Run evaluation on LoCoMo."""
    print(f"\n{'='*60}")
    print(f"Running LoCoMo")
    print(f"Split: {subset}")
    if num_samples:
        print(f"Max samples: {num_samples}")
    print(f"{'='*60}\n")

    from src.benchmarks.lococo_loader import load_lococo
    scenarios = load_lococo(subset=subset, num_samples=num_samples)

    if not scenarios:
        print(f"Error: No scenarios loaded from LoCoMo (subset={subset})")
        return None

    output_path = os.path.join(output_dir, f"lococo_{subset}_report.json")
    report = run_evaluation_with_scenarios(
        scenarios=scenarios,
        benchmark_name=f"LoCoMo-{subset}",
        output_path=output_path
    )
    return report


def run_all_benchmarks(num_samples: int = None, output_dir: str = "reports"):
    """Run all available benchmarks and generate a comprehensive report."""
    print(f"\n{'='*60}")
    print(f"Running ALL Benchmarks")
    if num_samples:
        print(f"Max samples per benchmark: {num_samples}")
    print(f"{'='*60}\n")

    all_reports = {}
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # 1. Custom benchmark
    print("\n>>> 1/4: Custom benchmark")
    try:
        all_reports["custom"] = run_custom_benchmark(output_dir=output_dir)
    except Exception as e:
        print(f"Failed: {e}")
        all_reports["custom"] = None

    # 2. MemoryAgentBench
    print("\n>>> 2/4: MemoryAgentBench")
    try:
        all_reports["mab"] = run_memoryagentbench(subset="all", num_samples=num_samples, output_dir=output_dir)
    except Exception as e:
        print(f"Failed: {e}")
        all_reports["mab"] = None

    # 3. LongMemEval
    print("\n>>> 3/4: LongMemEval")
    try:
        all_reports["longmemeval"] = run_longmemeval(subset="all", num_samples=num_samples, output_dir=output_dir)
    except Exception as e:
        print(f"Failed: {e}")
        all_reports["longmemeval"] = None

    # 4. LoCoMo
    print("\n>>> 4/4: LoCoMo")
    try:
        all_reports["lococo"] = run_lococo(subset="test", num_samples=num_samples, output_dir=output_dir)
    except Exception as e:
        print(f"Failed: {e}")
        all_reports["lococo"] = None

    # Generate combined summary
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "benchmarks_run": list(all_reports.keys()),
        "results": {},
        "failures": []
    }

    for name, report in all_reports.items():
        if report:
            summary["results"][name] = {
                "num_scenarios": report.get("num_scenarios", 0),
                "conflict_aware_accuracy": report["results"]["conflict_aware"]["scenario_accuracy"],
                "lww_accuracy": report["results"]["lww"]["scenario_accuracy"],
                "naive_accuracy": report["results"]["naive"]["scenario_accuracy"],
            }
        else:
            summary["failures"].append(name)

    combined_output = os.path.join(output_dir, f"all_benchmarks_summary_{timestamp}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"All benchmarks complete!")
    print(f"Combined summary saved to: {combined_output}")
    print(f"{'='*60}\n")

    # Print summary table
    print("\nSummary:")
    print(f"{'Benchmark':<20} {'Scenarios':<10} {'CA Acc':<10} {'LWW Acc':<10} {'Naive Acc':<10}")
    print("-" * 70)
    for name in sorted(all_reports.keys()):
        if name in summary["results"]:
            r = summary["results"][name]
            print(f"{name:<20} {r['num_scenarios']:<10} "
                  f"{r['conflict_aware_accuracy']:<10.3f} "
                  f"{r['lww_accuracy']:<10.3f} "
                  f"{r['naive_accuracy']:<10.3f}")
        else:
            print(f"{name:<20} FAILED")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for multiple memory evaluation datasets"
    )
    parser.add_argument(
        "benchmark",
        choices=["custom", "mab", "longmemeval", "lococo", "all"],
        help="Which benchmark to run"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Subset to use (e.g., 'conflict', 's', 'm', 'test')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save reports"
    )
    parser.add_argument(
        "--custom-path",
        type=str,
        default=None,
        help="Path to custom benchmark JSONL (for custom benchmark only)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load from streaming datasets (for HF benchmarks)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run requested benchmark
    if args.benchmark == "custom":
        report = run_custom_benchmark(benchmark_path=args.custom_path, output_dir=args.output_dir)
    elif args.benchmark == "mab":
        report = run_memoryagentbench(subset=args.subset, num_samples=args.num_samples, output_dir=args.output_dir)
    elif args.benchmark == "longmemeval":
        report = run_longmemeval(subset=args.subset, num_samples=args.num_samples, output_dir=args.output_dir)
    elif args.benchmark == "lococo":
        report = run_lococo(subset=args.subset, num_samples=args.num_samples, output_dir=args.output_dir)
    elif args.benchmark == "all":
        summary = run_all_benchmarks(num_samples=args.num_samples, output_dir=args.output_dir)
        report = summary
    else:
        parser.error(f"Unknown benchmark: {args.benchmark}")

    return report


if __name__ == "__main__":
    main()
