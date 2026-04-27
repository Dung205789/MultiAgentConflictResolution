#!/usr/bin/env python
"""
Local evaluation script using dummy agents (no heavy models required).

This script demonstrates before/after comparison of the memory layer
using lightweight agents that work on CPU-only environments.
"""
import sys
import json
import time
from typing import Dict, Any

sys.path.insert(0, '.')

from src.memory.shared_memory_store import SharedMemoryStore
from src.local_models.runner import MultiAgentLocalRunner, create_agent
from src.benchmarks import generate_benchmark_scenarios


def run_local_comparison(n_scenarios: int = 5, seed: int = 42):
    """
    Run before/after comparison using local dummy agents.

    Args:
        n_scenarios: Number of scenarios to test
        seed: Random seed for deterministic behavior (if applicable)

    Returns:
        Comparison results dict
    """
    print("=" * 80)
    print("LOCAL AGENT EVALUATION: BEFORE vs AFTER MEMORY LAYER")
    print("=" * 80)
    print(f"Testing with {n_scenarios} scenarios")
    print(f"Models: Dummy agents with varying reliability scores")
    print("")

    # Load benchmark scenarios
    print("Loading benchmark scenarios...")
    all_scenarios = generate_benchmark_scenarios()
    scenarios = all_scenarios[:n_scenarios]
    print(f"Selected {len(scenarios)} scenarios")
    for s in scenarios:
        print(f"  - {s['scenario_id']} ({s['scenario_type']})")

    # Define agent configurations
    # Use two agents with different reliability to create more interesting conflicts
    agent_configs = [
        {"agent_id": "agent_a", "model_type": "dummy", "reliability": 0.7},
        {"agent_id": "agent_b", "model_type": "dummy", "reliability": 0.85},
    ]

    results = {}

    # Test each writer mode
    for writer_type in ["conflict_aware", "lww", "naive"]:
        print(f"\n{'='*80}")
        print(f"Running {writer_type.upper()} mode")
        print("=" * 80)

        mode_results = []

        for i, scenario in enumerate(scenarios):
            # Create fresh store for each scenario
            store = SharedMemoryStore(persistence_path=f"tmp_local_{writer_type}_{i}.jsonl")
            store.records = []
            with open(store.persistence_path, "w") as f:
                f.write("")

            runner = MultiAgentLocalRunner(
                agent_configs=agent_configs,
                memory_store=store,
                writer_type=writer_type
            )

            result = runner.run_scenario(scenario)
            mode_results.append(result)

            # Print per-scenario summary
            metrics = result["metrics"]
            print(f"Scenario {i+1}: {scenario['scenario_id']}")
            print(f"  State match: {metrics.get('state_match', False)}")
            print(f"  Conflicts: {metrics.get('num_conflicts', 0)}")
            print(f"  Writes: {metrics.get('num_writes', 0)}")

        # Aggregate metrics
        total_scenarios = len(mode_results)
        correct = sum(1 for r in mode_results if r["metrics"]["state_match"])
        accuracy = correct / total_scenarios if total_scenarios else 0.0
        total_conflicts = sum(r["metrics"]["num_conflicts"] for r in mode_results)
        total_writes = sum(r["metrics"]["num_writes"] for r in mode_results)
        conflict_rate = total_conflicts / total_writes if total_writes else 0.0

        results[writer_type] = {
            "scenario_accuracy": accuracy,
            "total_conflicts": total_conflicts,
            "total_writes": total_writes,
            "conflict_rate": conflict_rate,
            "detailed_results": mode_results
        }

        print(f"\nAggregate for {writer_type}:")
        print(f"  Scenario Accuracy: {accuracy:.3f}")
        print(f"  Conflict Rate: {conflict_rate:.3f}")

    # Compute deltas
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    ca_acc = results["conflict_aware"]["scenario_accuracy"]
    lww_acc = results["lww"]["scenario_accuracy"]
    naive_acc = results["naive"]["scenario_accuracy"]

    print(f"Conflict-Aware accuracy: {ca_acc:.3f}")
    print(f"LWW accuracy:           {lww_acc:.3f}  (delta vs CA: {ca_acc - lww_acc:>+.3f})")
    print(f"Naive accuracy:         {naive_acc:.3f}  (delta vs CA: {ca_acc - naive_acc:>+.3f})")

    # Save results
    output = {
        "benchmark": "local_dummy_agents",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_scenarios": n_scenarios,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "detailed_results"} for k, v in results.items()},
    }

    output_path = "reports/local_evaluation_report.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed report saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_local_comparison(n_scenarios=3)
