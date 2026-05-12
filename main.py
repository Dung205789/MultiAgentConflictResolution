#!/usr/bin/env python
"""
Main entry point for multi-agent memory conflict resolution evaluation.

This unified runner supports:
- Multiple benchmarks (MemAE, MemoryAgentBench, LoCoMo, custom)
- All writer modes (conflict_aware, lww, naive)
- Fast mode (dummy agents) and slow mode (real models)
- Comprehensive evaluation with detailed metrics

Usage examples:
  # Run specific benchmark with default settings
  python main.py --benchmark memae --max-scenarios 50

  # Run real conflict scenarios (MemAE + MAB Conflict_Resolution)
  python main.py --benchmark real_conflicts

  # Run MemoryAgentBench conflict subset only
  python main.py --benchmark mab_conflict

  # Run with custom benchmark file
  python main.py --benchmark custom --custom-path data/my_benchmark.jsonl

  # Fast mode with dummy agents (no heavy models)
  python main.py --benchmark real_conflicts --use-dummy --max-scenarios 10

  # Run with real transformer models
  python main.py --benchmark real_conflicts --agent1-model Qwen/Qwen2.5-3B-Instruct --agent2-model Qwen/Qwen2.5-7B-Instruct

  # Run LoCoMo benchmark
  python main.py --benchmark lococo --max-scenarios 20

  # Run adversarial (synthetic, for ablation only)
  python main.py --benchmark adversarial --max-scenarios 100
"""
import argparse
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.run_evaluation import run_evaluation_with_scenarios
from src.benchmarks.unified_loader import load_benchmark, save_scenarios_to_jsonl


def run_evaluation_pipeline(
    scenarios: List[Any],
    benchmark_name: str,
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """Run evaluation on scenarios with all writer modes."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{benchmark_name}_report.json")

    report = run_evaluation_with_scenarios(
        scenarios=scenarios,
        benchmark_name=benchmark_name,
        output_path=output_path
    )

    # Print summary
    print("\n" + "="*70)
    print(f"RESULTS FOR {benchmark_name.upper()}")
    print("="*70)
    for mode in ["conflict_aware", "lww", "naive"]:
        if mode in report["results"]:
            res = report["results"][mode]
            print(f"{mode:17} | Acc: {res['scenario_accuracy']:.3f} | "
                  f"F1: {res['conflict_f1']:.3f} | "
                  f"Action: {res['action_accuracy']:.3f} | "
                  f"Mem F1: {res['final_memory_f1']:.3f}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for multi-agent memory conflict resolution"
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        type=str,
        default="real_conflicts",
        choices=["memae", "mab_conflict", "real_conflicts", "longmemeval", "safeflow", "mab", "lococo", "adversarial", "custom", "all"],
        help="Benchmark to evaluate. 'real_conflicts' = MemAE + MAB_Conflict. 'mab_conflict' = only MAB Conflict_Resolution split. 'all' excludes adversarial."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Subset for benchmarks that support it (e.g., 'conflict', 's', 'test')"
    )
    parser.add_argument(
        "--custom-path",
        type=str,
        default=None,
        help="Path to custom benchmark JSONL (for --benchmark custom)"
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Maximum scenarios to evaluate (for testing)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Samples for HuggingFace benchmarks (alias for max-scenarios)"
    )

    # Agent configuration
    parser.add_argument(
        "--use-dummy",
        action="store_true",
        help="Use dummy agents instead of transformer models (fast mode)"
    )
    parser.add_argument(
        "--agent1-model",
        type=str,
        default=None,
        help="Model for agent 1 (Qwen2.5-3B-Instruct by default)"
    )
    parser.add_argument(
        "--agent2-model",
        type=str,
        default=None,
        help="Model for agent 2 (Qwen2.5-7B-Instruct by default)"
    )
    parser.add_argument(
        "--agent1-reliability",
        type=float,
        default=0.85,
        help="Reliability for agent 1 (dummy mode only)"
    )
    parser.add_argument(
        "--agent2-reliability",
        type=float,
        default=0.75,
        help="Reliability for agent 2 (dummy mode only)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--cache-scenarios",
        type=str,
        default=None,
        help="Optional: save loaded scenarios to JSONL cache for faster re-runs"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which benchmarks to run
    benchmarks_to_run = []
    if args.benchmark in ["memae", "real_conflicts", "all"]:
        benchmarks_to_run.append(("memae", {}))
    if args.benchmark in ["mab_conflict", "real_conflicts"]:
        benchmarks_to_run.append(("mab", {"subset": "Conflict_Resolution"}))
    if args.benchmark in ["mab", "all"]:
        benchmarks_to_run.append(("mab", {"subset": args.subset}))
    if args.benchmark in ["longmemeval", "all"]:
        benchmarks_to_run.append(("longmemeval", {"subset": args.subset}))
    if args.benchmark in ["safeflow", "all"]:
        benchmarks_to_run.append(("safeflow", {}))
    if args.benchmark in ["lococo", "all"]:
        benchmarks_to_run.append(("lococo", {"subset": args.subset}))
    if args.benchmark == "adversarial":
        benchmarks_to_run.append(("adversarial", {"num_scenarios": args.max_scenarios or 100, "difficulty": args.subset if args.subset != "all" else "medium"}))
    if args.benchmark == "custom":
        if not args.custom_path:
            print("Error: --custom-path is required for custom benchmark")
            return 1
        benchmarks_to_run.append(("custom", {"custom_path": args.custom_path}))

    if not benchmarks_to_run:
        print("Error: No benchmarks selected")
        return 1

    all_reports = {}

    for benchmark_name, extra_kwargs in benchmarks_to_run:
        print("\n" + "="*70)
        print(f"RUNNING BENCHMARK: {benchmark_name.upper()}")
        print("="*70)

        try:
            # Load scenarios
            print(f"\nLoading {benchmark_name}...")
            load_kwargs = {}

            if args.max_scenarios:
                load_kwargs['max_scenarios'] = args.max_scenarios
            if args.num_samples and 'num_samples' not in load_kwargs:
                load_kwargs['num_samples'] = args.num_samples

            # Merge extra kwargs
            load_kwargs.update(extra_kwargs)

            # Handle custom benchmark
            if benchmark_name == "custom":
                custom_path = args.custom_path
                scenarios = []
                with open(custom_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            scenarios.append(json.loads(line))
                if args.max_scenarios:
                    scenarios = scenarios[:args.max_scenarios]
            else:
                scenarios = load_benchmark(benchmark_name, **load_kwargs)

            if not scenarios:
                print(f"No scenarios loaded for {benchmark_name}")
                continue

            print(f"[OK] Loaded {len(scenarios)} scenarios")

            # Optionally cache scenarios
            if args.cache_scenarios:
                cache_path = f"{args.cache_scenarios}/{benchmark_name}_scenarios.jsonl"
                if benchmark_name != "custom":
                    save_scenarios_to_jsonl(scenarios, cache_path)
                    print(f"  Cached to {cache_path}")

            # Show sample info
            s = scenarios[0]
            # Handle both Scenario objects and dicts
            if hasattr(s, 'scenario_id'):
                sid = s.scenario_id
                stype = s.scenario_type
                nevents = len(s.ordered_events)
                nqueries = len(s.queries)
            else:
                sid = s.get('scenario_id', 'unknown')
                stype = s.get('scenario_type', 'unknown')
                nevents = len(s.get('ordered_events', []))
                nqueries = len(s.get('queries', []))
            print(f"  Sample: ID={sid}, Type={stype}")
            print(f"  Events={nevents}, Queries={nqueries}")

            # Count types
            if len(scenarios) > 1:
                type_dist = {}
                for scen in scenarios:
                    if hasattr(scen, 'scenario_type'):
                        t = scen.scenario_type
                    else:
                        t = scen.get('scenario_type', 'unknown')
                    type_dist[t] = type_dist.get(t, 0) + 1
                print(f"  Types: {type_dist}")

            # Run evaluation
            print(f"\nRunning evaluation on {len(scenarios)} scenarios...")
            report = run_evaluation_pipeline(
                scenarios=scenarios,
                benchmark_name=benchmark_name,
                output_dir=args.output_dir
            )
            all_reports[benchmark_name] = report

        except Exception as e:
            print(f"\n[ERROR] {benchmark_name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_reports[benchmark_name] = None

    # Final summary
    print("\n" + "="*70)
    print("FULL PIPELINE COMPLETE")
    print("="*70)

    if all_reports:
        summary_path = os.path.join(args.output_dir, "summary.json")
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "benchmarks": {
                name: {
                    "num_scenarios": report.get("num_scenarios", 0) if isinstance(report, dict) else 0,
                    "status": "success" if report else "failed"
                } for name, report in all_reports.items()
            }
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

    return 0 if all(v is not None for v in all_reports.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
