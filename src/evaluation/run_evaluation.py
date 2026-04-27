"""
Main evaluation script with multi-benchmark support and comprehensive metrics.
Runs conflict_aware, lww, and naive baselines on custom or MemoryAgentBench data.
"""
import json
import os
import sys
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.multi_agent_pipeline import MultiAgentPipeline

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


def load_custom_benchmark(path: str) -> List[Dict[str, Any]]:
    """Load custom JSONL benchmark."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_evaluation_with_scenarios(
    scenarios: List[Dict[str, Any]],
    benchmark_name: str = "custom",
    output_path: str = "reports/evaluation_report.json"
) -> Dict[str, Any]:
    """
    Run evaluation with pre-loaded scenarios.

    Args:
        scenarios: List of benchmark scenario dicts
        benchmark_name: Name for reporting
        output_path: Where to save detailed report

    Returns:
        Dictionary with metrics for each writer type and per-scenario analysis.
    """
    print(f"Running evaluation on {len(scenarios)} scenarios from {benchmark_name}")

    # Run each writer type
    results = {}
    raw_results = {}  # Store raw results for per-type breakdown
    for mode in ["conflict_aware", "lww", "naive"]:
        print(f"\n=== Running {mode} ===")
        from src.pipeline.multi_agent_pipeline import MultiAgentPipeline
        pipeline = MultiAgentPipeline(
            mode=mode,
            persistence_path=f"tmp_eval_{mode}.jsonl",
            enable_persistence=False,
        )
        mode_results = []
        scenario_iter = tqdm(scenarios, desc=f"{mode} progress", total=len(scenarios)) if HAVE_TQDM else scenarios
        for scenario in scenario_iter:
            res = pipeline.run_scenario(scenario, enable_retrieval_eval=bool(scenario.get("queries")))
            mode_results.append(res)

        # Store raw results for later breakdown
        raw_results[mode] = mode_results

        # Compute metrics
        results[mode] = _compute_mode_metrics(mode_results, scenarios)

        # Print summary
        print(f"  Scenario accuracy: {results[mode]['scenario_accuracy']:.3f}")
        print(f"  Conflict F1: {results[mode]['conflict_f1']:.3f}")
        print(f"  Action accuracy: {results[mode]['action_accuracy']:.3f}")

    # Compute deltas
    deltas = {}
    for metric in ["scenario_accuracy", "conflict_f1", "action_accuracy"]:
        ca = results["conflict_aware"][metric]
        lww = results["lww"][metric]
        naive = results["naive"][metric]
        deltas[metric] = {
            "conflict_aware_minus_lww": ca - lww,
            "conflict_aware_minus_naive": ca - naive,
            "lww_minus_naive": lww - naive,
        }

    # Build final report
    report = {
        "benchmark": benchmark_name,
        "num_scenarios": len(scenarios),
        "timestamp": _get_timestamp(),
        "results": results,
        "deltas": deltas,
        "per_scenario_type": _compute_per_type_breakdown(raw_results, scenarios),
    }

    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to {output_path}")
    return report


def run_evaluation(
    benchmark_path: str = None,
    use_memoryagentbench: bool = False,
    mab_subset: str = "all",
    output_path: str = "reports/evaluation_report.json"
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation.

    Args:
        benchmark_path: Path to custom benchmark JSONL
        use_memoryagentbench: If True, load from Hugging Face
        mab_subset: Filter for MemoryAgentBench subsets
        output_path: Where to save detailed report
    """
    # Load benchmark
    if use_memoryagentbench:
        print(f"Loading MemoryAgentBench (subset={mab_subset})...")
        from src.benchmarks.memoryagentbench_loader import load_memoryagentbench
        scenarios = load_memoryagentbench(subset=mab_subset)
        benchmark_name = f"MemoryAgentBench-{mab_subset}"
    else:
        if not benchmark_path:
            benchmark_path = "data/enhanced_multi_agent_benchmark.jsonl"
        print(f"Loading custom benchmark: {benchmark_path}")
        scenarios = load_custom_benchmark(benchmark_path)
        benchmark_name = os.path.basename(benchmark_path)

    if not scenarios:
        print("No scenarios loaded!")
        return {}

    print(f"Loaded {len(scenarios)} scenarios")

    # Run each writer type
    results = {}
    raw_results = {}  # Store raw results for per-type breakdown
    for mode in ["conflict_aware", "lww", "naive"]:
        print(f"\n=== Running {mode} ===")
        pipeline = MultiAgentPipeline(
            mode=mode,
            persistence_path=f"tmp_eval_{mode}.jsonl",
            enable_persistence=False,
        )
        mode_results = []
        scenario_iter = tqdm(scenarios, desc=f"{mode} progress", total=len(scenarios)) if HAVE_TQDM else scenarios
        for scenario in scenario_iter:
            res = pipeline.run_scenario(scenario, enable_retrieval_eval=bool(scenario.get("queries")))
            mode_results.append(res)

        # Store raw results for later breakdown
        raw_results[mode] = mode_results

        # Compute metrics
        results[mode] = _compute_mode_metrics(mode_results, scenarios)

        # Print summary
        print(f"  Scenario accuracy: {results[mode]['scenario_accuracy']:.3f}")
        print(f"  Conflict F1: {results[mode]['conflict_f1']:.3f}")
        print(f"  Action accuracy: {results[mode]['action_accuracy']:.3f}")

    # Compute deltas
    deltas = {}
    for metric in ["scenario_accuracy", "conflict_f1", "action_accuracy"]:
        ca = results["conflict_aware"][metric]
        lww = results["lww"][metric]
        naive = results["naive"][metric]
        deltas[metric] = {
            "conflict_aware_minus_lww": ca - lww,
            "conflict_aware_minus_naive": ca - naive,
            "lww_minus_naive": lww - naive,
        }

    # Build final report
    report = {
        "benchmark": benchmark_name,
        "num_scenarios": len(scenarios),
        "timestamp": _get_timestamp(),
        "results": results,
        "deltas": deltas,
        "per_scenario_type": _compute_per_type_breakdown(raw_results, scenarios),
    }

    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to {output_path}")
    return report


def _compute_mode_metrics(mode_results: List[Dict[str, Any]], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute comprehensive metrics for a mode."""
    if len(mode_results) != len(scenarios):
        raise ValueError(f"mode_results length ({len(mode_results)}) does not match scenarios length ({len(scenarios)})")

    def _normalize_conflict_type(conflict_type: str) -> str:
        """Map detector conflict types to gold taxonomy."""
        gold_types = {"compatible_extension", "mutually_exclusive", "none", "semantic_overlap", "stale_read_conflict"}
        if conflict_type in gold_types:
            return conflict_type
        mapping = {
            "exact_duplicate": "semantic_overlap",
            "semantic_duplicate": "semantic_overlap",
            "potential_contradiction": "mutually_exclusive",
            "concurrent_update_conflict": "mutually_exclusive",
        }
        return mapping.get(conflict_type, conflict_type)

    def _is_conflict_decision(decision: Dict[str, Any]) -> bool:
        """
        Detect whether a write decision corresponds to an actual conflict event.
        """
        result = decision.get("result", {})
        candidate_count = result.get("candidate_count")
        if candidate_count is not None:
            return candidate_count > 0
        if result.get("conflict_detected", False):
            return True
        return result.get("conflict_type", "none") != "none"

    total_scenarios = len(mode_results)
    scenario_correct = sum(1 for r in mode_results if r["metrics"]["state_match"])
    total_writes = sum(r["metrics"]["num_writes"] for r in mode_results)
    total_conflicts = sum(r["metrics"]["num_conflicts"] for r in mode_results)

    # Action accuracy (only for events with conflict, i.e., candidate_count > 0)
    action_correct = 0
    action_total = 0
    first_write_total = 0
    per_action = {"overwrite": 0, "merge": 0, "keep_multiple_versions": 0, "defer": 0, "reject": 0, "append": 0}
    per_action_gold = {"overwrite": 0, "merge": 0, "keep_multiple_versions": 0, "defer": 0, "reject": 0, "append": 0}

    # For per-scenario-level metrics
    conflict_detection_correct = 0
    conflict_type_correct = 0

    # Conflict type distribution across decisions
    conflict_type_distribution = {}

    for res, scen in zip(mode_results, scenarios):
        # Scenario-level conflict detection: did we detect any conflict?
        predicted_conflict_exists = res["metrics"]["num_conflicts"] > 0
        gold_conflict_exists = scen.get("gold_conflict_exists", False)
        if predicted_conflict_exists == gold_conflict_exists:
            conflict_detection_correct += 1

        # Scenario-level conflict type: derive predicted type from first conflict decision, else "none"
        predicted_type = "none"
        for dec in res.get("arbitration_decisions", []):
            result = dec.get("result", {})
            if result.get("conflict_detected", False):
                predicted_type = _normalize_conflict_type(result.get("conflict_type", "none"))
                break
        gold_type = scen.get("gold_conflict_type", "none")
        if predicted_type == gold_type:
            conflict_type_correct += 1

        # Count conflict type distribution on conflict decisions only
        for dec in res.get("arbitration_decisions", []):
            if _is_conflict_decision(dec):
                result = dec.get("result", {})
                ct = _normalize_conflict_type(result.get("conflict_type", "none"))
                conflict_type_distribution[ct] = conflict_type_distribution.get(ct, 0) + 1

        # Action accuracy per conflict decision
        for ev in res["arbitration_decisions"]:
            if not _is_conflict_decision(ev):
                first_write_total += 1
                continue

            pred = ev.get("resolution_action", "append")
            gold = scen.get("gold_resolution_action", "append")
            action_total += 1
            if pred == gold:
                action_correct += 1
            per_action[pred] = per_action.get(pred, 0) + 1
            per_action_gold[gold] = per_action_gold.get(gold, 0) + 1

    action_accuracy = action_correct / action_total if action_total else 0.0
    conflict_detection_accuracy = conflict_detection_correct / total_scenarios if total_scenarios else 0.0
    conflict_type_accuracy = conflict_type_correct / total_scenarios if total_scenarios else 0.0

    # Conflict detection metrics (precision/recall/F1) already computed
    tp = sum(1 for r, s in zip(mode_results, scenarios) if s.get("gold_conflict_exists") and r["metrics"]["num_conflicts"] > 0)
    fp = sum(1 for r, s in zip(mode_results, scenarios) if not s.get("gold_conflict_exists") and r["metrics"]["num_conflicts"] > 0)
    fn = sum(1 for r, s in zip(mode_results, scenarios) if s.get("gold_conflict_exists") and r["metrics"]["num_conflicts"] == 0)
    conflict_precision = tp / (tp + fp) if (tp + fp) else 0.0
    conflict_recall = tp / (tp + fn) if (tp + fn) else 0.0
    conflict_f1 = (
        2 * conflict_precision * conflict_recall / (conflict_precision + conflict_recall)
        if (conflict_precision + conflict_recall) else 0.0
    )

    # Retrieval metrics (recall@5)
    retrieval_recall_sum = 0.0
    retrieval_tasks = 0
    for r in mode_results:
        retrieval_results = r.get("retrieval_results")
        if retrieval_results:  # not None and not empty
            for m in retrieval_results:
                if "recall_at_k" in m:
                    retrieval_recall_sum += m["recall_at_k"]
                    retrieval_tasks += 1
    retrieval_recall_at_5 = retrieval_recall_sum / retrieval_tasks if retrieval_tasks else 0.0

    # Stale read handling accuracy (for scenarios with gold_conflict_type == "stale_read_conflict")
    stale_correct = 0
    stale_total = 0
    for res, scen in zip(mode_results, scenarios):
        if scen.get("gold_conflict_type") == "stale_read_conflict":
            stale_total += 1
            gold_action = scen.get("gold_resolution_action")
            # Find the first conflict decision (should be the stale read conflict)
            pred_action = None
            for dec in res.get("arbitration_decisions", []):
                result = dec.get("result", {})
                if result.get("conflict_detected", False):
                    pred_action = dec.get("resolution_action", "append")
                    break
            if pred_action is None:
                pred_action = "append"
            if pred_action == gold_action:
                stale_correct += 1
    stale_handling_accuracy = stale_correct / stale_total if stale_total else 0.0

    # Temporal update accuracy: for scenarios that are mutually exclusive and with ordered timestamps
    temporal_correct = 0
    temporal_total = 0
    for res, scen in zip(mode_results, scenarios):
        # Check if scenario qualifies as temporal update scenario:
        # gold_conflict_type is "mutually_exclusive" and events are ordered (second event later than first)
        if scen.get("gold_conflict_type") == "mutually_exclusive":
            events = scen.get("ordered_events", [])
            # Need at least two write proposals
            writes = [ev for ev in events if ev.get("event_type") == "write_proposal"]
            if len(writes) >= 2:
                # Check timestamps: assume they have timestamps or they are sequential
                # If timestamps are present, use them; else assume order by step
                ts1 = writes[0].get("timestamp", writes[0].get("step", 0))
                ts2 = writes[1].get("timestamp", writes[1].get("step", 0))
                if ts2 > ts1:
                    temporal_total += 1
                    # Check if the system's resolution action for that conflict is overwrite
                    # Find the decision corresponding to the second write (first conflict)
                    pred_action = None
                    # The second write decision should be the first conflict detected
                    for dec in res.get("arbitration_decisions", []):
                        result = dec.get("result", {})
                        if result.get("conflict_detected", False):
                            pred_action = dec.get("resolution_action", "append")
                            break
                    if pred_action == "overwrite":
                        temporal_correct += 1
    temporal_update_accuracy = temporal_correct / temporal_total if temporal_total else 0.0

    # Final memory F1: compare final_visible_state vs gold_visible_shared_state_after_commit
    def extract_facts(state):
        facts = set()
        for r in state:
            subj = r.get("subject", "")
            pred = r.get("predicate", "")
            obj = str(r.get("object_val", r.get("object", "")))
            facts.add((subj, pred, obj))
        return facts

    total_gold = 0
    total_pred = 0
    total_correct = 0
    for res, scen in zip(mode_results, scenarios):
        gold_state = scen.get("gold_visible_shared_state_after_commit", [])
        pred_state = res.get("final_visible_state", [])
        gold_facts = extract_facts(gold_state)
        pred_facts = extract_facts(pred_state)
        total_gold += len(gold_facts)
        total_pred += len(pred_facts)
        total_correct += len(gold_facts & pred_facts)
    memory_precision = total_correct / total_pred if total_pred else 0.0
    memory_recall = total_correct / total_gold if total_gold else 0.0
    memory_f1 = (
        2 * memory_precision * memory_recall / (memory_precision + memory_recall)
        if (memory_precision + memory_recall) else 0.0
    )

    # Per-conflict-type action accuracy (conflict decisions only)
    per_type_correct = {}
    per_type_total = {}
    for res, scen in zip(mode_results, scenarios):
        gold_type = scen.get("gold_conflict_type", "none")
        for ev in res.get("arbitration_decisions", []):
            if _is_conflict_decision(ev):
                pred = ev.get("resolution_action", "append")
                gold = scen.get("gold_resolution_action", "append")
                per_type_total[gold_type] = per_type_total.get(gold_type, 0) + 1
                if pred == gold:
                    per_type_correct[gold_type] = per_type_correct.get(gold_type, 0) + 1
    per_type_accuracy = {ct: per_type_correct.get(ct, 0) / per_type_total[ct] if per_type_total.get(ct, 0) else 0.0 for ct in per_type_total}

    # Branch explosion
    entity_counts = []
    for r in mode_results:
        final_visible = r.get("final_visible_state", [])
        entity_counts.append(len(final_visible))
    avg_branch_count = sum(entity_counts) / len(entity_counts) if entity_counts else 0.0

    # requires_judge count (not yet implemented; default 0)
    requires_judge_count = 0

    return {
        "scenario_accuracy": scenario_correct / total_scenarios if total_scenarios else 0.0,
        "conflict_detection_accuracy": conflict_detection_accuracy,
        "conflict_type_accuracy": conflict_type_accuracy,
        "conflict_precision": conflict_precision,
        "conflict_recall": conflict_recall,
        "conflict_f1": conflict_f1,
        "action_accuracy": action_accuracy,
        "first_writes_skipped": first_write_total,
        "total_writes": total_writes,
        "total_conflicts": total_conflicts,
        "conflict_rate": total_conflicts / total_writes if total_writes else 0.0,
        "action_distribution_pred": per_action,
        "action_distribution_gold": per_action_gold,
        "per_conflict_type_action_accuracy": per_type_accuracy,
        "retrieval_recall_at_5": retrieval_recall_at_5,
        "stale_handling_accuracy": stale_handling_accuracy,
        "stale_events": stale_total,
        "temporal_update_accuracy": temporal_update_accuracy,
        "final_memory_f1": memory_f1,
        "avg_branch_count": avg_branch_count,
        "conflict_type_distribution": conflict_type_distribution,
        "requires_judge_count": requires_judge_count,
    }


def _compute_per_type_breakdown(
    raw_results: Dict[str, List[Dict[str, Any]]],
    scenarios: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """Breakdown action accuracy by scenario type using raw results."""
    breakdown = {}
    # Map scenario index to its type
    idx_to_type = {i: s.get("scenario_type", "unknown") for i, s in enumerate(scenarios)}

    for mode, results_list in raw_results.items():
        # Accumulate correct/total per scenario type
        type_correct: Dict[str, int] = {}
        type_total: Dict[str, int] = {}

        for idx, res in enumerate(results_list):
            st = idx_to_type[idx]
            # For each arbitration decision in this scenario
            for ev in res.get("arbitration_decisions", []):
                result = ev.get("result", {})
                candidate_count = result.get("candidate_count")
                is_conflict_decision = (
                    (candidate_count is not None and candidate_count > 0)
                    or result.get("conflict_detected", False)
                    or result.get("conflict_type", "none") != "none"
                )
                if not is_conflict_decision:
                    continue
                pred = ev.get("resolution_action", "append")
                gold = scenarios[idx].get("gold_resolution_action", "append")
                type_total[st] = type_total.get(st, 0) + 1
                if pred == gold:
                    type_correct[st] = type_correct.get(st, 0) + 1

        # Compute per-type accuracy for this mode
        for st in type_total:
            acc = type_correct.get(st, 0) / type_total[st] if type_total[st] else 0.0
            if st not in breakdown:
                breakdown[st] = {}
            breakdown[st][f"{mode}_action_accuracy"] = acc

    return breakdown


def _get_timestamp():
    """Get current timestamp string."""
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"


if __name__ == "__main__":
    # Configure evaluation
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=None, help="Path to custom benchmark JSONL")
    parser.add_argument("--use-mab", action="store_true", help="Use MemoryAgentBench from Hugging Face")
    parser.add_argument("--mab-subset", type=str, default="all", help="MemoryAgentBench subset filter")
    parser.add_argument("--output", type=str, default="reports/evaluation_report.json", help="Output path")
    args = parser.parse_args()

    report = run_evaluation(
        benchmark_path=args.benchmark,
        use_memoryagentbench=args.use_mab,
        mab_subset=args.mab_subset,
        output_path=args.output
    )

    if report:
        print("\n=== Summary ===")
        for mode in ["conflict_aware", "lww", "naive"]:
            acc = report["results"][mode]["scenario_accuracy"]
            f1 = report["results"][mode]["conflict_f1"]
            act = report["results"][mode]["action_accuracy"]
            print(f"{mode:15} | Acc: {acc:.3f} | F1: {f1:.3f} | Action: {act:.3f}")
