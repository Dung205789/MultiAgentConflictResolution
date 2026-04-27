#!/usr/bin/env python
"""
Comprehensive evaluation runner for multi-agent memory conflict resolution.
Usage:
  python run_all.py [--benchmark path] [--use-mab] [--mab-subset subset]
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Path to custom benchmark JSONL (default: data/enhanced_multi_agent_benchmark.jsonl)")
    parser.add_argument("--use-mab", action="store_true",
                        help="Use MemoryAgentBench from Hugging Face")
    parser.add_argument("--mab-subset", type=str, default="all",
                        help="MemoryAgentBench subset (conflict, temporal, update, etc.)")
    parser.add_argument("--use-longmemeval", action="store_true",
                        help="Use LongMemEval from Hugging Face")
    parser.add_argument("--longmemeval-subset", type=str, default="all",
                        help="LongMemEval subset (s, m, oracle, all)")
    parser.add_argument("--use-lococo", action="store_true",
                        help="Use LoCoMo from Hugging Face")
    parser.add_argument("--lococo-subset", type=str, default="test",
                        help="LoCoMo split (test, train, validation, all)")
    parser.add_argument("--output", type=str, default="reports/evaluation_report.json",
                        help="Output path for report")
    parser.add_argument("--regenerate-benchmark", action="store_true",
                        help="Regenerate custom benchmark before evaluation")
    args = parser.parse_args()

    # Step 1: Regenerate custom benchmark if requested
    if args.regenerate_benchmark and not any([args.use_mab, args.use_longmemeval, args.use_lococo]):
        print("=== Regenerating benchmark ===")
        from src.benchmarks import generate_benchmark_scenarios, save_benchmark
        scenarios = generate_benchmark_scenarios()
        save_benchmark(scenarios, "data/enhanced_multi_agent_benchmark.jsonl")
        print(f"Generated {len(scenarios)} scenarios")

    # Step 2: Run evaluation
    print("\n=== Running evaluation ===")
    from src.evaluation.run_evaluation import run_evaluation

    # Determine which benchmark to use (priority: MAB > LongMemEval > LoCoMo > custom)
    if args.use_mab:
        benchmark_name = f"MemoryAgentBench-{args.mab_subset}"
        from src.benchmarks.memoryagentbench_loader import load_memoryagentbench
        scenarios = load_memoryagentbench(subset=args.mab_subset)
    elif args.use_longmemeval:
        benchmark_name = f"LongMemEval-{args.longmemeval_subset}"
        from src.benchmarks.longmemeval_loader import load_longmemeval
        scenarios = load_longmemeval(subset=args.longmemeval_subset)
    elif args.use_lococo:
        benchmark_name = f"LoCoMo-{args.lococo_subset}"
        from src.benchmarks.lococo_loader import load_lococo
        scenarios = load_lococo(subset=args.lococo_subset)
    else:
        benchmark_path = args.benchmark or "data/enhanced_multi_agent_benchmark.jsonl"
        benchmark_name = os.path.basename(benchmark_path)
        from src.evaluation.run_evaluation import load_custom_benchmark
        scenarios = load_custom_benchmark(benchmark_path)

    if not scenarios:
        print("No scenarios loaded!")
        return

    print(f"Loaded {len(scenarios)} scenarios from {benchmark_name}")

    # Run evaluation
    from src.evaluation.run_evaluation import run_evaluation_with_scenarios
    report = run_evaluation_with_scenarios(
        scenarios=scenarios,
        benchmark_name=benchmark_name,
        output_path=args.output
    )

    if report:
        print("\n=== Results ===")
        for mode in ["conflict_aware", "lww", "naive"]:
            res = report["results"][mode]
            print(f"{mode:15} | Scen Acc: {res['scenario_accuracy']:.3f} | "
                  f"Conflict F1: {res['conflict_f1']:.3f} | "
                  f"Action Acc: {res['action_accuracy']:.3f}")

        # Show per-type breakdown if available
        if report.get("per_scenario_type"):
            print("\n=== Per-Scenario-Type Action Accuracy ===")
            for st, metrics in report["per_scenario_type"].items():
                print(f"{st}:")
                for key, val in metrics.items():
                    print(f"  {key}: {val:.3f}")

        print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
