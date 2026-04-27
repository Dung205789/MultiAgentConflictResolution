#!/usr/bin/env python
"""
Comparative benchmark: Memory Layer Impact on Multi-Agent Communication

This script runs two experiments:
1. WITHOUT memory layer: Agents use naive writer (no conflict resolution)
2. WITH memory layer: Agents use conflict-aware writer (full memory infrastructure)

Metrics are collected and compared to demonstrate the value of the memory layer.
"""
import sys
import os
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.memory.shared_memory_store import SharedMemoryStore
from src.local_models.runner import create_agent, MultiAgentLocalRunner
from src.evaluation.run_evaluation import _compute_mode_metrics as _compute_mode_metrics_eval

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def load_scenarios(benchmark_path: str, max_scenarios: int = None) -> List[Dict[str, Any]]:
    """Load benchmark scenarios from JSONL file."""
    scenarios = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
                if max_scenarios and len(scenarios) >= max_scenarios:
                    break
    print(f"Loaded {len(scenarios)} scenarios from {benchmark_path}")
    return scenarios


def run_experiment(
    scenarios: List[Dict[str, Any]],
    agent_configs: List[Dict[str, Any]],
    writer_type: str = "conflict_aware",
    enable_retrieval_eval: bool = True,
    force_model_extraction: bool = False,
) -> Dict[str, Any]:
    """
    Run scenarios with specified writer type.

    Returns:
        Results dict with metrics and logs per scenario
    """
    print(f"\n{'='*70}")
    print(f"Running experiment with writer: {writer_type}")
    print(f"{'='*70}\n")

    all_results = []
    total_start = time.time()

    # Create fresh store for this experiment
    store = SharedMemoryStore(
        persistence_path="tmp_comparison_store.jsonl",
        enable_persistence=False,
    )
    store.records = []

    runner = MultiAgentLocalRunner(
        agent_configs=agent_configs,
        memory_store=store,
        writer_type=writer_type,
        force_model_extraction=force_model_extraction,
        show_event_progress=force_model_extraction,
    )

    # Setup progress bar
    progress_bar = tqdm(total=len(scenarios), desc=f"{writer_type}", unit="scenario") if TQDM_AVAILABLE else None

    for i, scenario in enumerate(scenarios, 1):
        scenario_id = scenario.get("scenario_id", f"scenario_{i}")

        start = time.time()
        try:
            result = runner.run_scenario(scenario, enable_retrieval_eval=enable_retrieval_eval)
            elapsed = time.time() - start
            result["metrics"]["execution_time"] = elapsed
            all_results.append(result)
            conflicts = result.get("metrics", {}).get("num_conflicts", 0)
            state_match = result.get("metrics", {}).get("state_match", False)
            if progress_bar:
                progress_bar.set_postfix({"ok": state_match, "conf": conflicts})
                progress_bar.update(1)
            else:
                print(f"  [{i}/{len(scenarios)}] Running {scenario_id}... [OK] (state_match={state_match}, conflicts={conflicts})")
        except Exception as e:
            if progress_bar:
                progress_bar.set_postfix({"error": str(e)[:20]})
                progress_bar.update(1)
            else:
                print(f"  [{i}/{len(scenarios)}] Running {scenario_id}... [ERR] Error: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "scenario_id": scenario_id,
                "error": str(e),
                "metrics": {"state_match": False, "num_writes": 0, "num_conflicts": 0, "execution_time": time.time() - start},
                "arbitration_decisions": [],
                "final_visible_state": []
            })

    if progress_bar:
        progress_bar.close()

    total_time = time.time() - total_start

    # Aggregate basic summary
    num_scenarios = len(all_results)
    num_success = sum(1 for r in all_results if "error" not in r)
    total_writes = sum(r.get("metrics", {}).get("num_writes", 0) for r in all_results)
    total_conflicts = sum(r.get("metrics", {}).get("num_conflicts", 0) for r in all_results)
    state_matches = sum(1 for r in all_results if r.get("metrics", {}).get("state_match", False))
    avg_execution_time = sum(r.get("metrics", {}).get("execution_time", 0) for r in all_results) / num_scenarios if num_scenarios else 0

    summary = {
        "writer_type": writer_type,
        "num_scenarios": num_scenarios,
        "num_success": num_success,
        "total_execution_time": total_time,
        "avg_execution_time": avg_execution_time,
        "total_writes": total_writes,
        "total_conflicts": total_conflicts,
        "conflict_rate": total_conflicts / total_writes if total_writes else 0,
        "overall_state_accuracy": state_matches / num_scenarios if num_scenarios else 0,
        "detailed_results": all_results  # Keep all for further metric computation
    }

    print(f"\n  Summary:")
    print(f"    Success: {num_success}/{num_scenarios}")
    print(f"    State accuracy: {summary['overall_state_accuracy']:.3f}")
    print(f"    Total writes: {total_writes}, conflicts: {total_conflicts}")
    print(f"    Avg time/scenario: {avg_execution_time:.3f}s")

    return summary


def compute_mode_metrics(mode_results: List[Dict[str, Any]], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute metrics using the canonical evaluator implementation.
    """
    metrics = _compute_mode_metrics_eval(mode_results, scenarios)
    # Preserve old key name for compatibility with existing reports/scripts.
    metrics["retrieval_recall_avg"] = metrics.get("retrieval_recall_at_5", 0.0)
    return metrics


def compare_memory_impact(
    scenarios: List[Dict[str, Any]],
    agent_configs: List[Dict[str, Any]],
    output_dir: str = "reports",
    enable_retrieval_eval: bool = True,
    force_model_extraction: bool = False,
) -> Dict[str, Any]:
    """
    Run comparison: WITHOUT memory layer vs WITH memory layer.

    Returns:
        Comparative report dictionary
    """
    print("\n" + "="*70)
    print("MEMORY LAYER IMPACT COMPARISON")
    print("="*70)
    print(f"\nAgents configuration:")
    for i, cfg in enumerate(agent_configs, 1):
        print(f"  Agent {i}: {cfg.get('model_type', 'unknown')}")
        if 'model_name' in cfg:
            print(f"    Model: {cfg['model_name']}")
        if 'reliability' in cfg:
            print(f"    Reliability: {cfg['reliability']}")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Retrieval evaluation: {'enabled' if enable_retrieval_eval else 'disabled'}")

    # Experiment 1: WITHOUT memory layer (naive writer)
    baseline_summary = run_experiment(
        scenarios,
        agent_configs,
        writer_type="naive",
        enable_retrieval_eval=enable_retrieval_eval,
        force_model_extraction=force_model_extraction,
    )

    # Experiment 2: WITH memory layer (conflict-aware writer)
    memory_summary = run_experiment(
        scenarios,
        agent_configs,
        writer_type="conflict_aware",
        enable_retrieval_eval=enable_retrieval_eval,
        force_model_extraction=force_model_extraction,
    )

    # Compute detailed metrics for each mode using the raw results
    print("\n" + "="*70)
    print("Computing detailed metrics...")
    print("="*70)

    baseline_metrics = compute_mode_metrics(baseline_summary["detailed_results"], scenarios)
    memory_metrics = compute_mode_metrics(memory_summary["detailed_results"], scenarios)

    # Build final report with standard format
    report = {
        "benchmark": "Multi-Agent Memory Benchmark",
        "num_scenarios": len(scenarios),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_configs": agent_configs,
        "results": {
            "naive": baseline_metrics,
            "conflict_aware": memory_metrics
        },
        "raw_summaries": {
            "naive": baseline_summary,
            "conflict_aware": memory_summary
        }
    }

    # Compute deltas
    deltas = {}
    for metric in ["scenario_accuracy", "conflict_f1", "action_accuracy", "retrieval_recall_avg", "stale_handling_accuracy"]:
        naive_val = baseline_metrics.get(metric, 0.0)
        ca_val = memory_metrics.get(metric, 0.0)
        deltas[metric] = {
            "conflict_aware_minus_naive": ca_val - naive_val,
            "relative_improvement": (ca_val - naive_val) / naive_val * 100 if naive_val > 0 else float('inf')
        }

    report["deltas"] = deltas

    # Per-scenario-type breakdown (combine from both modes)
    per_type = {}
    for mode_name, metrics in [("naive", baseline_metrics), ("conflict_aware", memory_metrics)]:
        for ct, acc in metrics.get("per_conflict_type_action_accuracy", {}).items():
            if ct not in per_type:
                per_type[ct] = {}
            per_type[ct][f"{mode_name}_action_accuracy"] = acc
    report["per_scenario_type"] = per_type

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"memory_comparison_{timestamp}.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Generate and print summary
    summary_lines = []
    summary_lines.append("\n" + "="*70)
    summary_lines.append("COMPARISON SUMMARY")
    summary_lines.append("="*70)
    summary_lines.append(f"\nBenchmark: {report['benchmark']}")
    summary_lines.append(f"Scenarios: {report['num_scenarios']}")
    summary_lines.append(f"Timestamp: {report['timestamp']}")
    summary_lines.append("\n--- Metrics ---")
    summary_lines.append(f"{'Metric':<30} {'Naive':<12} {'Memory-Aware':<15} {'Delta':<12}")
    summary_lines.append("-"*70)

    metrics_to_show = [
        ("Scenario Accuracy", "scenario_accuracy"),
        ("Conflict F1", "conflict_f1"),
        ("Action Accuracy", "action_accuracy"),
        ("Retrieval Recall", "retrieval_recall_avg"),
        ("Stale Handling Accuracy", "stale_handling_accuracy"),
        ("Avg Branch Count", "avg_branch_count"),
        ("Conflict Rate", "conflict_rate"),
    ]

    for label, key in metrics_to_show:
        naive_val = baseline_metrics.get(key, 0.0)
        ca_val = memory_metrics.get(key, 0.0)
        delta = ca_val - naive_val
        summary_lines.append(f"{label:<30} {naive_val:<12.4f} {ca_val:<15.4f} {delta:>+12.4f}")

    summary_lines.append("\n--- Per-Conflict-Type Action Accuracy ---")
    for ct, accs in per_type.items():
        naive = accs.get("naive_action_accuracy", 0.0)
        ca = accs.get("conflict_aware_action_accuracy", 0.0)
        summary_lines.append(f"{ct:<30} naive: {naive:.4f}  memory: {ca:.4f}  Delta: {ca-naive:>+.4f}")

    summary_lines.append("\n" + "="*70)
    summary = "\n".join(summary_lines)
    print(summary)
    print(f"\nFull JSON report saved to: {report_path}")

    # Also save summary as text
    summary_path = os.path.join(output_dir, f"memory_comparison_summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved to: {summary_path}")
    print("="*70)

    return report


def _resolve_default_model(
    cli_value: str,
    env_key: str,
    local_model_dir_name: str,
    hf_fallback: str
) -> str:
    """Resolve model path/name with precedence: CLI > ENV > local cache > HF fallback."""
    if cli_value:
        return cli_value

    env_value = os.environ.get(env_key)
    if env_value:
        return env_value

    project_root = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(project_root, "models_cache", local_model_dir_name)
    if os.path.isdir(local_path):
        return local_path

    return hf_fallback


def main():
    parser = argparse.ArgumentParser(description="Run memory layer impact comparison")
    parser.add_argument("--benchmark", type=str, default="data/enhanced_multi_agent_benchmark.jsonl",
                        help="Path to benchmark JSONL file")
    parser.add_argument("--num-scenarios", type=int, default=None,
                        help="Maximum number of scenarios to run (default: all)")
    parser.add_argument("--output-dir", type=str, default="reports",
                        help="Output directory for reports")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Hugging Face token for gated models (can also set HF_TOKEN env var)")
    parser.add_argument("--agent1-model", type=str, default=None,
                        help="Model for agent 1 (overrides AGENT1_MODEL env var)")
    parser.add_argument("--agent2-model", type=str, default=None,
                        help="Model for agent 2 (overrides AGENT2_MODEL env var)")
    parser.add_argument("--agent1-reliability", type=float, default=0.85,
                        help="Reliability for agent 1 (for dummy mode)")
    parser.add_argument("--agent2-reliability", type=float, default=0.75,
                        help="Reliability for agent 2 (for dummy mode)")
    parser.add_argument("--use-dummy", action="store_true",
                        help="Use dummy agents instead of transformer models (for testing)")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="Disable retrieval evaluation (faster)")
    parser.add_argument(
        "--force-model-extraction",
        action="store_true",
        help="Force model extraction on every write event even when proposal already exists",
    )

    args = parser.parse_args()

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Resolve agent models with local-cache-first behavior.
    agent1_model = _resolve_default_model(
        cli_value=args.agent1_model,
        env_key="AGENT1_MODEL",
        local_model_dir_name="Qwen2.5-3B-Instruct",
        hf_fallback="Qwen/Qwen2.5-3B-Instruct",
    )
    agent2_model = _resolve_default_model(
        cli_value=args.agent2_model,
        env_key="AGENT2_MODEL",
        local_model_dir_name="Qwen2.5-7B-Instruct",
        hf_fallback="Qwen/Qwen2.5-7B-Instruct",
    )

    # Load scenarios
    if not os.path.exists(args.benchmark):
        print(f"Error: Benchmark file not found: {args.benchmark}")
        print("Available benchmarks:")
        for f in os.listdir("data") if os.path.exists("data") else []:
            print(f"  - data/{f}")
        return 1

    scenarios = load_scenarios(args.benchmark, max_scenarios=args.num_scenarios)
    if not scenarios:
        print("Error: No scenarios loaded")
        return 1

    # Configure agents
    if args.use_dummy:
        agent_configs = [
            {"agent_id": "agent_a", "model_type": "dummy", "reliability": args.agent1_reliability},
            {"agent_id": "agent_b", "model_type": "dummy", "reliability": args.agent2_reliability},
        ]
    else:
        agent_configs = [
            {
                "agent_id": "agent_a",
                "model_type": "transformer",
                "model_name": agent1_model,
                "device": "cpu"  # Change to "cuda" if GPU available
            },
            {
                "agent_id": "agent_b",
                "model_type": "transformer",
                "model_name": agent2_model,
                "device": "cpu"
            }
        ]

    # Run comparison
    try:
        report = compare_memory_impact(
            scenarios,
            agent_configs,
            output_dir=args.output_dir,
            enable_retrieval_eval=not args.no_retrieval,
            force_model_extraction=args.force_model_extraction,
        )
        return 0
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
