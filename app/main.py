#!/usr/bin/env python
"""
Main entry point for multi-agent memory conflict resolution evaluation.

This unified runner supports:
- Multiple benchmarks (MemAE, MemoryAgentBench, LoCoMo, custom)
- All writer modes (conflict_aware, lww, naive)
- Adapter-structured fast mode and optional model-based re-extraction
- Comprehensive evaluation with detailed metrics

Usage examples:
  # Run specific benchmark with default settings
  python app/main.py --benchmark memae --max-scenarios 50

  # Run real conflict scenarios (MemAE + MAB Conflict_Resolution)
  python app/main.py --benchmark real_conflicts

  # Run MemoryAgentBench conflict subset only
  python app/main.py --benchmark mab_conflict

  # Run with custom benchmark file
  python app/main.py --benchmark custom --custom-path data/my_benchmark.jsonl

  # Fast benchmark mode using adapter-structured proposals
  python app/main.py --benchmark real_conflicts --use-dummy --max-scenarios 10

  # Re-extract benchmark facts with local transformer models
  python app/main.py --benchmark real_conflicts --agent1-model Qwen/Qwen2.5-3B-Instruct --agent2-model Qwen/Qwen2.5-7B-Instruct

  # Run LoCoMo benchmark
  python app/main.py --benchmark locomo --max-scenarios 20
"""
import argparse
import json
import sys
import os
from datetime import datetime, UTC
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.run_evaluation import run_evaluation_with_scenarios
from src.benchmarks.unified_loader import load_benchmark, save_scenarios_to_jsonl


def resolve_execution_device(preference: str) -> str:
    """Resolve requested runtime device into `cpu` or `cuda`."""
    if preference in {"cpu", "cuda"}:
        return preference

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_execution_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate CLI agent flags into a concrete execution configuration."""
    if args.use_dummy and (args.agent1_model or args.agent2_model):
        raise ValueError("`--use-dummy` cannot be combined with `--agent1-model` or `--agent2-model`.")

    use_model_reextract = bool(args.agent1_model or args.agent2_model)
    proposal_source = "agent_extract" if use_model_reextract else "structured"
    strict_agent_execution = use_model_reextract
    resolved_device = resolve_execution_device(args.device)

    agent_configs = {
        "__slot_0__": {
            "role": "primary",
            "reliability": args.agent1_reliability,
            "runtime_mode": "research_strict" if use_model_reextract else "debug_fallback",
        },
        "__slot_1__": {
            "role": "secondary",
            "reliability": args.agent2_reliability,
            "runtime_mode": "research_strict" if use_model_reextract else "debug_fallback",
        },
    }

    if use_model_reextract:
        agent_configs["__slot_0__"].update({
            "model_type": "transformer",
            "model_name": args.agent1_model or "Qwen/Qwen2.5-1.5B-Instruct",
            "device": resolved_device,
        })
        agent_configs["__slot_1__"].update({
            "model_type": "transformer",
            "model_name": args.agent2_model or args.agent1_model or "Qwen/Qwen2.5-1.5B-Instruct",
            "device": resolved_device,
        })
    else:
        agent_configs["__slot_0__"]["model_type"] = "structured"
        agent_configs["__slot_1__"]["model_type"] = "structured"

    return {
        "proposal_source": proposal_source,
        "strict_agent_execution": strict_agent_execution,
        "agent_configs": agent_configs,
        "device": resolved_device,
        "mode_label": "dummy_structured" if args.use_dummy or not use_model_reextract else "transformer_reextract",
    }


def run_evaluation_pipeline(
    scenarios: List[Any],
    benchmark_name: str,
    output_dir: str = "reports",
    execution_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Run evaluation on scenarios with all writer modes."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{benchmark_name}_report.json")

    report = run_evaluation_with_scenarios(
        scenarios=scenarios,
        benchmark_name=benchmark_name,
        output_path=output_path,
        execution_config=execution_config,
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


def load_real_conflicts_bundle(max_scenarios: int = None) -> List[Any]:
    """
    Load the accepted conflict bundle used by `--benchmark real_conflicts`.
    The bundle combines MemAE and MemoryAgentBench Conflict_Resolution.
    """
    if max_scenarios is None:
        memae_limit = None
        mab_limit = None
    else:
        memae_limit = max_scenarios // 2
        mab_limit = max_scenarios - memae_limit

    scenarios: List[Any] = []
    if memae_limit is None or memae_limit > 0:
        scenarios.extend(load_benchmark("memae", max_scenarios=memae_limit))
    if mab_limit is None or mab_limit > 0:
        scenarios.extend(load_benchmark("mab", subset="Conflict_Resolution", max_scenarios=mab_limit))
    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for multi-agent memory conflict resolution"
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        type=str,
        default="real_conflicts",
        choices=["memae", "mab_conflict", "real_conflicts", "longmemeval", "safeflow", "mab", "locomo", "lococo", "custom", "all"],
        help="Benchmark to evaluate. `real_conflicts` = MemAE + MAB Conflict_Resolution. All exposed choices are accepted external benchmarks or user-provided custom data."
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
        help="Fast accepted-benchmark mode: use adapter-structured proposals and apply reliability priors without model re-extraction"
    )
    parser.add_argument(
        "--agent1-model",
        type=str,
        default=None,
        help="Optional local transformer model for agent 1. When set, the runner re-extracts proposals from raw benchmark text."
    )
    parser.add_argument(
        "--agent2-model",
        type=str,
        default=None,
        help="Optional local transformer model for agent 2. When set, the runner re-extracts proposals from raw benchmark text."
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for transformer agents. `auto` uses CUDA when available."
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
    if args.benchmark == "lococo":
        print("Warning: `lococo` is deprecated. Using `locomo` instead.")
        args.benchmark = "locomo"

    try:
        execution_config = build_execution_config(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which benchmarks to run
    benchmarks_to_run = []
    if args.benchmark == "memae":
        benchmarks_to_run.append({"report_name": "memae", "load_name": "memae", "extra_kwargs": {}})
    if args.benchmark == "mab_conflict":
        benchmarks_to_run.append({"report_name": "mab_conflict", "load_name": "mab", "extra_kwargs": {"subset": "Conflict_Resolution"}})
    if args.benchmark == "real_conflicts":
        benchmarks_to_run.append({"report_name": "real_conflicts", "load_name": "real_conflicts", "extra_kwargs": {}})
    if args.benchmark in ["mab", "all"]:
        benchmarks_to_run.append({"report_name": "mab", "load_name": "mab", "extra_kwargs": {"subset": args.subset}})
    if args.benchmark in ["longmemeval", "all"]:
        benchmarks_to_run.append({"report_name": "longmemeval", "load_name": "longmemeval", "extra_kwargs": {"subset": args.subset}})
    if args.benchmark in ["safeflow", "all"]:
        benchmarks_to_run.append({"report_name": "safeflow", "load_name": "safeflow", "extra_kwargs": {}})
    if args.benchmark in ["locomo", "all"]:
        benchmarks_to_run.append({"report_name": "locomo", "load_name": "locomo", "extra_kwargs": {"subset": args.subset}})
    if args.benchmark == "all":
        benchmarks_to_run.insert(0, {"report_name": "memae", "load_name": "memae", "extra_kwargs": {}})
    if args.benchmark == "custom":
        if not args.custom_path:
            print("Error: --custom-path is required for custom benchmark")
            return 1
        benchmarks_to_run.append({"report_name": "custom", "load_name": "custom", "extra_kwargs": {"custom_path": args.custom_path}})

    if not benchmarks_to_run:
        print("Error: No benchmarks selected")
        return 1

    all_reports = {}

    for benchmark_spec in benchmarks_to_run:
        benchmark_name = benchmark_spec["report_name"]
        load_name = benchmark_spec["load_name"]
        extra_kwargs = benchmark_spec["extra_kwargs"]
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
            if load_name == "custom":
                custom_path = args.custom_path
                scenarios = []
                with open(custom_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            scenarios.append(json.loads(line))
                if args.max_scenarios:
                    scenarios = scenarios[:args.max_scenarios]
            elif load_name == "real_conflicts":
                scenarios = load_real_conflicts_bundle(max_scenarios=args.max_scenarios or args.num_samples)
            else:
                scenarios = load_benchmark(load_name, **load_kwargs)

            if not scenarios:
                print(f"No scenarios loaded for {benchmark_name}")
                continue

            print(f"[OK] Loaded {len(scenarios)} scenarios")

            # Optionally cache scenarios
            if args.cache_scenarios:
                cache_path = f"{args.cache_scenarios}/{benchmark_name}_scenarios.jsonl"
                if load_name != "custom":
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
                output_dir=args.output_dir,
                execution_config=execution_config,
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
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "execution": execution_config,
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
