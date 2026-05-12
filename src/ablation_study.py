"""
Ablation Study: Test different weight and threshold configurations.

Experiments:
1. Weight sensitivity: Test different weight distributions
2. Threshold sensitivity: Test different threshold values
"""

import json
import os
import sys
import yaml
from typing import Dict, Any, List
from copy import deepcopy

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.benchmarks.unified_loader import load_benchmark
from src.evaluation.run_evaluation import _compute_mode_metrics
from src.pipeline.multi_agent_pipeline import MultiAgentPipeline

# Weight experiment configurations
WEIGHT_EXPERIMENTS = [
    {
        "name": "default",
        "weights": {"confidence": 0.4, "provenance": 0.3, "recency": 0.2, "authority": 0.1},
        "description": "Default weights from plan"
    },
    {
        "name": "confidence_heavy",
        "weights": {"confidence": 0.7, "provenance": 0.2, "recency": 0.1, "authority": 0.0},
        "description": "Confidence-heavy (70%)"
    },
    {
        "name": "recency_heavy",
        "weights": {"confidence": 0.2, "provenance": 0.1, "recency": 0.6, "authority": 0.1},
        "description": "Recency-heavy (60%)"
    },
    {
        "name": "provenance_heavy",
        "weights": {"confidence": 0.2, "provenance": 0.6, "recency": 0.1, "authority": 0.1},
        "description": "Provenance-heavy (60%)"
    },
    {
        "name": "authority_heavy",
        "weights": {"confidence": 0.2, "provenance": 0.1, "recency": 0.1, "authority": 0.6},
        "description": "Authority-heavy (60%)"
    },
]

# Threshold experiment configurations
THRESHOLD_EXPERIMENTS = [
    {
        "name": "default",
        "thresholds": {
            "overwrite_margin": 0.12,
            "keep_multiple_versions_margin": 0.20,
            "defer_below_score": 0.40,
        },
        "description": "Default thresholds"
    },
    {
        "name": "tight_margins",
        "thresholds": {
            "overwrite_margin": 0.05,
            "keep_multiple_versions_margin": 0.10,
            "defer_below_score": 0.40,
        },
        "description": "Tight margins (more overwrites)"
    },
    {
        "name": "loose_margins",
        "thresholds": {
            "overwrite_margin": 0.20,
            "keep_multiple_versions_margin": 0.30,
            "defer_below_score": 0.40,
        },
        "description": "Loose margins (more keep_multiple)"
    },
    {
        "name": "aggressive_overwrite",
        "thresholds": {
            "overwrite_margin": 0.08,
            "keep_multiple_versions_margin": 0.15,
            "defer_below_score": 0.30,
        },
        "description": "Aggressive overwrite (lower thresholds)"
    },
]


def load_config(config_path: str = "configs/arbitration.yaml") -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """Save config to YAML file."""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_weight_ablation(
    benchmark: str = "memae",
    max_scenarios: int = None,
    output_path: str = "reports/ablation/weight_sensitivity.json"
):
    """
    Run weight sensitivity analysis.

    Tests how different weight distributions affect accuracy.
    """
    print("=" * 60)
    print("WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Load benchmark
    print(f"\nLoading benchmark: {benchmark}")
    scenarios = load_benchmark(benchmark, max_scenarios=max_scenarios)
    print(f"Loaded {len(scenarios)} scenarios")

    results = {}

    for exp in WEIGHT_EXPERIMENTS:
        print(f"\n--- Running: {exp['name']} ---")
        print(f"Description: {exp['description']}")
        print(f"Weights: {exp['weights']}")

        # Create pipeline with modified config
        pipeline = MultiAgentPipeline(
            mode="conflict_aware",
            persistence_path=f"tmp_ablation_{exp['name']}.jsonl",
            enable_persistence=False
        )

        # Modify weights in the conflict writer
        if hasattr(pipeline.conflict_writer, 'weights'):
            pipeline.conflict_writer.weights = exp['weights']

        # Run scenarios
        mode_results = []
        for scenario in scenarios:
            res = pipeline.run_scenario(scenario, enable_retrieval_eval=False)
            mode_results.append(res)

        # Compute metrics
        metrics = _compute_mode_metrics(mode_results, scenarios)
        results[exp['name']] = {
            "description": exp['description'],
            "weights": exp['weights'],
            "metrics": metrics
        }

        # Print summary
        print(f"  Scenario accuracy: {metrics.get('scenario_accuracy', 0):.3f}")
        print(f"  Action accuracy: {metrics.get('action_accuracy', 0):.3f}")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("WEIGHT SENSITIVITY RESULTS")
    print("=" * 80)
    print(f"{'Experiment':<25} {'Accuracy':>10} {'Action Acc':>10} {'F1':>10}")
    print("-" * 80)
    for name, data in results.items():
        m = data['metrics']
        print(f"{name:<25} {m.get('scenario_accuracy', 0):>10.3f} {m.get('action_accuracy', 0):>10.3f} {m.get('conflict_f1', 0):>10.3f}")

    return results


def run_threshold_ablation(
    benchmark: str = "memae",
    max_scenarios: int = None,
    output_path: str = "reports/ablation/threshold_sensitivity.json"
):
    """
    Run threshold sensitivity analysis.

    Tests how different threshold values affect accuracy and behavior.
    """
    print("=" * 60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Load benchmark
    print(f"\nLoading benchmark: {benchmark}")
    scenarios = load_benchmark(benchmark, max_scenarios=max_scenarios)
    print(f"Loaded {len(scenarios)} scenarios")

    results = {}

    for exp in THRESHOLD_EXPERIMENTS:
        print(f"\n--- Running: {exp['name']} ---")
        print(f"Description: {exp['description']}")
        print(f"Thresholds: {exp['thresholds']}")

        # Create pipeline
        pipeline = MultiAgentPipeline(
            mode="conflict_aware",
            persistence_path=f"tmp_ablation_{exp['name']}.jsonl",
            enable_persistence=False
        )

        # Modify thresholds in the conflict writer
        if hasattr(pipeline.conflict_writer, 'thresholds'):
            pipeline.conflict_writer.thresholds.update(exp['thresholds'])

        # Run scenarios
        mode_results = []
        for scenario in scenarios:
            res = pipeline.run_scenario(scenario, enable_retrieval_eval=False)
            mode_results.append(res)

        # Compute metrics
        metrics = _compute_mode_metrics(mode_results, scenarios)
        results[exp['name']] = {
            "description": exp['description'],
            "thresholds": exp['thresholds'],
            "metrics": metrics
        }

        # Print summary
        print(f"  Scenario accuracy: {metrics.get('scenario_accuracy', 0):.3f}")
        print(f"  Action accuracy: {metrics.get('action_accuracy', 0):.3f}")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY RESULTS")
    print("=" * 80)
    print(f"{'Experiment':<25} {'Accuracy':>10} {'Action Acc':>10} {'F1':>10}")
    print("-" * 80)
    for name, data in results.items():
        m = data['metrics']
        print(f"{name:<25} {m.get('scenario_accuracy', 0):>10.3f} {m.get('action_accuracy', 0):>10.3f} {m.get('conflict_f1', 0):>10.3f}")

    return results


def run_full_ablation(
    benchmark: str = "memae",
    max_scenarios: int = None,
    output_dir: str = "reports/ablation"
):
    """Run both weight and threshold ablation studies."""
    print("=" * 60)
    print("FULL ABLATION STUDY")
    print("=" * 60)

    # Weight ablation
    weight_results = run_weight_ablation(
        benchmark=benchmark,
        max_scenarios=max_scenarios,
        output_path=os.path.join(output_dir, "weight_sensitivity.json")
    )

    # Threshold ablation
    threshold_results = run_threshold_ablation(
        benchmark=benchmark,
        max_scenarios=max_scenarios,
        output_path=os.path.join(output_dir, "threshold_sensitivity.json")
    )

    # Combined summary
    summary = {
        "benchmark": benchmark,
        "max_scenarios": max_scenarios,
        "weight_experiments": list(weight_results.keys()),
        "threshold_experiments": list(threshold_results.keys()),
    }

    summary_path = os.path.join(output_dir, "ablation_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nFull ablation study complete!")
    print(f"Summary saved to: {summary_path}")

    return {
        "weight_results": weight_results,
        "threshold_results": threshold_results,
        "summary": summary
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--benchmark", type=str, default="memae", help="Benchmark to use")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Max scenarios")
    parser.add_argument("--output-dir", type=str, default="reports/ablation", help="Output directory")
    parser.add_argument("--experiment", type=str, choices=["weight", "threshold", "all"], default="all", help="Which experiment to run")

    args = parser.parse_args()

    if args.experiment == "weight":
        run_weight_ablation(benchmark=args.benchmark, max_scenarios=args.max_scenarios, output_path=os.path.join(args.output_dir, "weight_sensitivity.json"))
    elif args.experiment == "threshold":
        run_threshold_ablation(benchmark=args.benchmark, max_scenarios=args.max_scenarios, output_path=os.path.join(args.output_dir, "threshold_sensitivity.json"))
    else:
        run_full_ablation(benchmark=args.benchmark, max_scenarios=args.max_scenarios, output_dir=args.output_dir)
