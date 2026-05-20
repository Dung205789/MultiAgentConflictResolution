"""
Main evaluation script with multi-benchmark support and comprehensive metrics.
Runs conflict_aware, lww, and naive baselines on custom or MemoryAgentBench data.
"""
import json
import os
import sys
from typing import Dict, Any, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.scenario_contract import scenario_identifier, scenarios_to_dicts
from src.pipeline.multi_agent_pipeline import MultiAgentPipeline
from src.evaluation.error_analysis import build_error_analysis_report
from src.evaluation.qa_reasoner import analyze_question_requirements

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


def _use_tqdm() -> bool:
    disable_flag = os.environ.get("PROJECTMEM_DISABLE_TQDM", "").strip().lower()
    return HAVE_TQDM and disable_flag not in {"1", "true", "yes", "on"}


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _progress_path(output_path: str) -> str:
    return f"{output_path}.progress.json"


def _partial_path(output_path: str) -> str:
    return f"{output_path}.partial.json"


def _failure_bundle_path(output_path: str) -> str:
    return f"{output_path}.failure_bundle.json"


def _scenario_bundle_dir(output_path: str) -> str:
    return f"{output_path}.scenarios"


def load_custom_benchmark(path: str) -> List[Dict[str, Any]]:
    """Load custom JSONL benchmark."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _initialize_progress_state(
    benchmark_name: str,
    scenarios: Sequence[Dict[str, Any]],
    output_path: str,
    execution_config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "benchmark": benchmark_name,
        "num_scenarios": len(scenarios),
        "output_path": output_path,
        "execution": execution_config,
        "status": "running",
        "completed_modes": [],
    }


def _instantiate_pipeline(
    pipeline_mode: str,
    result_key: str,
    execution_config: Dict[str, Any],
    variant: str,
) -> MultiAgentPipeline:
    return MultiAgentPipeline(
        mode=pipeline_mode,
        persistence_path=f"tmp_eval_{result_key}.jsonl",
        enable_persistence=False,
        agent_configs=execution_config.get("agent_configs"),
        proposal_source=execution_config.get("proposal_source", "structured"),
        strict_agent_execution=execution_config.get("strict_agent_execution", False),
        conflict_aware_variant=variant,
        track_name=execution_config.get("track_name", "oracle_structured"),
        allow_structured_fallback_in_end_to_end=execution_config.get(
            "allow_structured_fallback_in_end_to_end", False
        ),
    )


def _write_partial_report(
    output_path: str,
    benchmark_name: str,
    scenarios: Sequence[Dict[str, Any]],
    execution_config: Dict[str, Any],
    results: Dict[str, Dict[str, Any]],
    mode_specs: Sequence[Dict[str, str]],
) -> None:
    _write_json(
        _partial_path(output_path),
        {
            "benchmark": benchmark_name,
            "num_scenarios": len(scenarios),
            "timestamp": _get_timestamp(),
            "execution": execution_config,
            "results": results,
            "mode_order": [spec["result_key"] for spec in mode_specs],
        },
    )


def run_evaluation_with_scenarios(
    scenarios: List[Dict[str, Any]],
    benchmark_name: str = "custom",
    output_path: str = "reports/evaluation_report.json",
    execution_config: Optional[Dict[str, Any]] = None,
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

    execution_config = dict(execution_config or {})
    scenario_dicts = scenarios_to_dicts(scenarios)
    mode_specs = _resolve_mode_specs(execution_config)
    progress_state = _initialize_progress_state(benchmark_name, scenario_dicts, output_path, execution_config)
    _write_json(_progress_path(output_path), progress_state)

    results = {}
    raw_results = {}
    scenario_bundle_dir = _scenario_bundle_dir(output_path)
    if execution_config.get("emit_scenario_bundles"):
        os.makedirs(scenario_bundle_dir, exist_ok=True)
    try:
        for spec in mode_specs:
            result_key = spec["result_key"]
            pipeline_mode = spec["pipeline_mode"]
            variant = spec["conflict_aware_variant"]
            print(f"\n=== Running {result_key} ===")
            pipeline = _instantiate_pipeline(pipeline_mode, result_key, execution_config, variant)
            mode_results = []
            scenario_iter = (
                tqdm(
                    scenario_dicts,
                    desc=f"{result_key} progress",
                    total=len(scenario_dicts),
                    ascii=True,
                    dynamic_ncols=False,
                )
                if _use_tqdm()
                else scenario_dicts
            )
            for scenario_index, scenario_dict in enumerate(scenario_iter, start=1):
                scenario_id = scenario_identifier(scenario_dict, scenario_index)
                print(f"[{result_key}] scenario {scenario_index}/{len(scenario_dicts)}: {scenario_id}")
                progress_state.update(
                    {
                        "current_mode": result_key,
                        "current_scenario_index": scenario_index,
                        "current_scenario_id": scenario_id,
                    }
                )
                _write_json(_progress_path(output_path), progress_state)
                res = pipeline.run_scenario(
                    scenario_dict,
                    enable_retrieval_eval=bool(scenario_dict.get("queries")),
                )
                mode_results.append(res)
                if execution_config.get("emit_scenario_bundles"):
                    _write_json(
                        os.path.join(scenario_bundle_dir, f"{result_key}__{scenario_index:03d}__{scenario_id}.json"),
                        _build_scenario_bundle_entry(
                            result_key=result_key,
                            scenario_index=scenario_index,
                            scenario_dict=scenario_dict,
                            scenario_result=res,
                        ),
                    )

            raw_results[result_key] = mode_results
            results[result_key] = _compute_mode_metrics(mode_results, scenario_dicts, execution_config)

            print(f"  Scenario accuracy: {results[result_key]['scenario_accuracy']:.3f}")
            print(f"  Conflict F1: {results[result_key]['conflict_f1']:.3f}")
            print(f"  Action accuracy: {results[result_key]['action_accuracy']:.3f}")
            if results[result_key].get("qa_total", 0):
                print(f"  QA exact match: {results[result_key]['qa_exact_match']:.3f}")
                print(f"  QA SubEM: {results[result_key]['qa_subem']:.3f}")

            progress_state["completed_modes"].append(result_key)
            progress_state["last_completed_mode"] = result_key
            _write_partial_report(output_path, benchmark_name, scenario_dicts, execution_config, results, mode_specs)
            _write_json(_progress_path(output_path), progress_state)
    except BaseException as exc:
        progress_state["status"] = "failed"
        progress_state["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(_progress_path(output_path), progress_state)
        raise

    report = _build_report_payload(
        raw_results=raw_results,
        scenarios=scenario_dicts,
        benchmark_name=benchmark_name,
        execution_config=execution_config,
        mode_specs=mode_specs,
    )
    _persist_report_artifacts(output_path, report, raw_results, scenario_dicts, benchmark_name, execution_config)
    progress_state["status"] = "completed"
    progress_state["final_report"] = output_path
    _write_json(_progress_path(output_path), progress_state)

    print(f"\nReport saved to {output_path}")
    return report


def _build_scenario_bundle_entry(
    result_key: str,
    scenario_index: int,
    scenario_dict: Dict[str, Any],
    scenario_result: Dict[str, Any],
) -> Dict[str, Any]:
    qa_results = scenario_result.get("qa_results") or []
    qa_failures = [item for item in qa_results if not item.get("exact_match", False)]
    arbitration_decisions = scenario_result.get("arbitration_decisions", []) or []
    non_append_conflicts = [
        dec for dec in arbitration_decisions
        if (dec.get("result", {}) or {}).get("conflict_detected", False)
    ]
    return {
        "mode": result_key,
        "scenario_index": scenario_index,
        "scenario_id": scenario_dict.get("scenario_id"),
        "scenario_type": scenario_dict.get("scenario_type"),
        "scenario_contract_version": scenario_dict.get("_scenario_contract_version"),
        "metrics": scenario_result.get("metrics", {}),
        "num_events": len(scenario_dict.get("ordered_events", []) or []),
        "num_queries": len(scenario_dict.get("queries", []) or []),
        "qa_summary": {
            "qa_total": len(qa_results),
            "qa_failures": len(qa_failures),
        },
        "qa_results": qa_results,
        "qa_failures": qa_failures,
        "retrieval_results": scenario_result.get("retrieval_results", []),
        "detected_conflicts": scenario_result.get("detected_conflicts", []),
        "conflict_decisions": non_append_conflicts,
        "arbitration_decisions": scenario_result.get("arbitration_decisions", []),
        "write_proposals": scenario_result.get("write_proposals", []),
        "final_visible_state": scenario_result.get("final_visible_state", []),
        "execution": scenario_result.get("execution", {}),
    }


def _build_failure_bundle(
    raw_results: Dict[str, List[Dict[str, Any]]],
    scenarios: Sequence[Any],
    benchmark_name: str,
    execution_config: Dict[str, Any],
) -> Dict[str, Any]:
    scenario_dicts = scenarios_to_dicts(scenarios)

    bundle: Dict[str, Any] = {
        "benchmark": benchmark_name,
        "timestamp": _get_timestamp(),
        "execution": execution_config,
        "modes": {},
    }

    error_report = build_error_analysis_report(raw_results, scenarios)
    for mode, mode_results in raw_results.items():
        per_scenario: List[Dict[str, Any]] = []
        for idx, (scenario_dict, result) in enumerate(zip(scenario_dicts, mode_results), start=1):
            qa_results = result.get("qa_results") or []
            qa_failures = [item for item in qa_results if not item.get("exact_match", False)]
            per_scenario.append(
                {
                    "scenario_index": idx,
                    "scenario_id": scenario_dict.get("scenario_id"),
                    "scenario_type": scenario_dict.get("scenario_type"),
                    "state_match": bool((result.get("metrics") or {}).get("state_match", False)),
                    "qa_total": len(qa_results),
                    "qa_failures": len(qa_failures),
                    "qa_successes": len(qa_results) - len(qa_failures),
                    "failure_categories": [
                        failure.get("category")
                        for failure in next(
                            (
                                item.get("failures", [])
                                for item in error_report.get(mode, {}).get("detailed_failures", [])
                                if item.get("scenario_id") == scenario_dict.get("scenario_id")
                            ),
                            [],
                        )
                    ],
                }
            )
        bundle["modes"][mode] = {
            "summary_counts": error_report.get(mode, {}).get("summary_counts", {}),
            "per_scenario": per_scenario,
            "detailed_failures": error_report.get(mode, {}).get("detailed_failures", []),
        }
    return bundle


def _build_error_channel_report(
    raw_results: Dict[str, List[Dict[str, Any]]],
    scenarios: Sequence[Any],
    error_report: Dict[str, Any],
) -> Dict[str, Any]:
    scenario_dicts = scenarios_to_dicts(scenarios)
    report: Dict[str, Any] = {"modes": {}}
    for mode, results_list in raw_results.items():
        qa_outcome = {
            "arbitration_correct_qa_correct": 0,
            "arbitration_correct_qa_wrong": 0,
            "arbitration_wrong_qa_correct": 0,
            "arbitration_wrong_qa_wrong": 0,
            "no_qa": 0,
        }
        for scenario_dict, result in zip(scenario_dicts, results_list):
            qa_results = result.get("qa_results") or []
            state_match = bool((result.get("metrics") or {}).get("state_match", False))
            if not qa_results:
                qa_outcome["no_qa"] += 1
                continue
            qa_success = all(bool(item.get("exact_match", False)) for item in qa_results)
            if state_match and qa_success:
                qa_outcome["arbitration_correct_qa_correct"] += 1
            elif state_match and not qa_success:
                qa_outcome["arbitration_correct_qa_wrong"] += 1
            elif (not state_match) and qa_success:
                qa_outcome["arbitration_wrong_qa_correct"] += 1
            else:
                qa_outcome["arbitration_wrong_qa_wrong"] += 1

        summary_counts = error_report.get(mode, {}).get("summary_counts", {})
        report["modes"][mode] = {
            "qa_outcome_matrix": qa_outcome,
            "qa_failure_channels": {
                "anchor_resolution_failure": summary_counts.get("wrong_anchor_resolution", 0),
                "reverse_relation_failure": summary_counts.get("wrong_reverse_relation", 0),
                "graph_edge_failure": (
                    summary_counts.get("parser_no_edge", 0)
                    + summary_counts.get("missing_terminal_edge", 0)
                ),
                "answer_type_mismatch": summary_counts.get("answer_type_mismatch", 0),
                "answer_selection_failure": summary_counts.get("overwrite_correct_but_qa_unused", 0),
            },
        }
    return report


def _build_report_payload(
    raw_results: Dict[str, List[Dict[str, Any]]],
    scenarios: Sequence[Any],
    benchmark_name: str,
    execution_config: Dict[str, Any],
    mode_specs: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    scenario_dicts = scenarios_to_dicts(scenarios)
    results = {
        result_key: _compute_mode_metrics(mode_results, scenario_dicts, execution_config)
        for result_key, mode_results in raw_results.items()
    }
    if "conflict_aware_full" in results and "conflict_aware" not in results:
        results["conflict_aware"] = dict(results["conflict_aware_full"])

    primary_conflict_aware_key = "conflict_aware_full" if "conflict_aware_full" in results else "conflict_aware"
    deltas = _compute_deltas(results, primary_conflict_aware_key)
    error_report = build_error_analysis_report(raw_results, scenario_dicts)
    return {
        "benchmark": benchmark_name,
        "num_scenarios": len(scenario_dicts),
        "timestamp": _get_timestamp(),
        "execution": execution_config,
        "track_reporting": {
            "requested_track_name": execution_config.get("track_name", "oracle_structured"),
            "proposal_source": execution_config.get("proposal_source", "structured"),
            "fallback_policy_enabled": bool(execution_config.get("allow_structured_fallback_in_end_to_end", False)),
            "reporting_paths": [
                "oracle_structured",
                "end_to_end_extract",
                "end_to_end_extract__structured_fallback",
            ],
        },
        "scenario_contract": {
            "version": scenario_dicts[0].get("_scenario_contract_version", "unknown") if scenario_dicts else "unknown",
            "adapter_input": "ISF scenario dictionaries",
            "runtime_memory_model": "canonical MemoryEntry lifecycle",
            "qa_surface": "final_visible_state",
        },
        "results": results,
        "deltas": deltas,
        "primary_conflict_aware_key": primary_conflict_aware_key,
        "mode_order": [spec["result_key"] for spec in mode_specs],
        "per_scenario_type": _compute_per_type_breakdown(raw_results, scenario_dicts),
        "error_channels": _build_error_channel_report(raw_results, scenario_dicts, error_report),
        "error_analysis": error_report if execution_config.get("enable_error_analysis") else None,
    }


def _persist_report_artifacts(
    output_path: str,
    report: Dict[str, Any],
    raw_results: Dict[str, List[Dict[str, Any]]],
    scenarios: Sequence[Any],
    benchmark_name: str,
    execution_config: Dict[str, Any],
) -> None:
    if execution_config.get("enable_error_analysis"):
        _write_json(
            _failure_bundle_path(output_path),
            _build_failure_bundle(raw_results, scenarios, benchmark_name, execution_config),
        )
    _write_json(output_path, report)


def _reconstruct_qa_results_from_bundle(
    bundle: Dict[str, Any],
    scenario_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    qa_results = bundle.get("qa_results")
    if qa_results:
        return list(qa_results)

    queries = scenario_dict.get("queries", []) or []
    failure_items = bundle.get("qa_failures", []) or []
    failures_by_query = {item.get("query", ""): item for item in failure_items}
    reconstructed: List[Dict[str, Any]] = []
    for query in queries:
        query_text = query.get("query_text", "")
        gold_answers = query.get("gold_answers", [])
        failure_item = failures_by_query.get(query_text)
        if failure_item:
            reconstructed.append(dict(failure_item))
            continue
        reconstructed.append(
            {
                "step": "final",
                "query": query_text,
                "gold": gold_answers,
                "predicted_answers": list(gold_answers),
                "predicted_normalized": [],
                "gold_normalized": [],
                "exact_match": True,
                "substring_exact_match": True,
                "hops": 0,
                "answer_type": None,
                "path": [],
            }
        )
    return reconstructed


def _scenario_bundle_to_mode_result(bundle: Dict[str, Any], scenario_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metrics": bundle.get("metrics", {}),
        "qa_results": _reconstruct_qa_results_from_bundle(bundle, scenario_dict),
        "retrieval_results": bundle.get("retrieval_results", []),
        "detected_conflicts": bundle.get("detected_conflicts", []),
        "arbitration_decisions": bundle.get("arbitration_decisions") or bundle.get("conflict_decisions", []),
        "write_proposals": bundle.get("write_proposals", []),
        "final_visible_state": bundle.get("final_visible_state", []),
        "execution": bundle.get("execution", {}),
    }


def _discover_mode_keys_from_scenario_bundles(output_path: str) -> List[str]:
    bundle_dir = _scenario_bundle_dir(output_path)
    if not os.path.isdir(bundle_dir):
        return []
    discovered: List[str] = []
    for name in sorted(os.listdir(bundle_dir)):
        if "__" not in name or not name.endswith(".json"):
            continue
        mode_key = name.split("__", 1)[0]
        if mode_key not in discovered:
            discovered.append(mode_key)
    return discovered


def _recover_raw_results_from_scenario_bundles(
    output_path: str,
    mode_specs: Sequence[Dict[str, str]],
    scenarios: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    bundle_dir = _scenario_bundle_dir(output_path)
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(f"Scenario bundle directory not found: {bundle_dir}")

    scenario_dicts = scenarios_to_dicts(scenarios)
    raw_results: Dict[str, List[Dict[str, Any]]] = {}
    for spec in mode_specs:
        mode_key = spec["result_key"]
        bundles: List[Dict[str, Any]] = []
        prefix = f"{mode_key}__"
        for name in sorted(os.listdir(bundle_dir)):
            if not (name.startswith(prefix) and name.endswith(".json")):
                continue
            with open(os.path.join(bundle_dir, name), "r", encoding="utf-8") as f:
                bundles.append(json.load(f))
        bundles.sort(key=lambda item: int(item.get("scenario_index", 0)))
        if len(bundles) != len(scenario_dicts):
            raise ValueError(
                f"Mode {mode_key} has {len(bundles)} scenario bundles, expected {len(scenario_dicts)}."
            )
        raw_results[mode_key] = [
            _scenario_bundle_to_mode_result(bundle, scenario_dict)
            for bundle, scenario_dict in zip(bundles, scenario_dicts)
        ]
    return raw_results


def finalize_report_from_scenario_artifacts(
    scenarios: Sequence[Any],
    benchmark_name: str,
    output_path: str,
    execution_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scenario_dicts = scenarios_to_dicts(scenarios)
    partial_snapshot: Dict[str, Any] = {}
    if os.path.exists(_partial_path(output_path)):
        with open(_partial_path(output_path), "r", encoding="utf-8") as f:
            partial_snapshot = json.load(f)

    execution = dict(partial_snapshot.get("execution") or {})
    if execution_config:
        execution.update(execution_config)

    discovered_modes = partial_snapshot.get("mode_order") or _discover_mode_keys_from_scenario_bundles(output_path)
    if discovered_modes:
        execution["selected_modes"] = list(discovered_modes)
    if any(mode.startswith("conflict_aware_") for mode in discovered_modes):
        execution["include_conflict_aware_ablations"] = True

    mode_specs = _resolve_mode_specs(execution)
    raw_results = _recover_raw_results_from_scenario_bundles(output_path, mode_specs, scenario_dicts)
    report = _build_report_payload(
        raw_results=raw_results,
        scenarios=scenario_dicts,
        benchmark_name=benchmark_name,
        execution_config=execution,
        mode_specs=mode_specs,
    )
    report["recovered_from_artifacts"] = True
    _persist_report_artifacts(output_path, report, raw_results, scenario_dicts, benchmark_name, execution)

    progress_state = _initialize_progress_state(benchmark_name, scenario_dicts, output_path, execution)
    progress_state["status"] = "completed"
    progress_state["completed_modes"] = [spec["result_key"] for spec in mode_specs]
    progress_state["final_report"] = output_path
    progress_state["recovered_from_artifacts"] = True
    _write_json(_progress_path(output_path), progress_state)
    return report


def run_evaluation(
    benchmark_path: str = None,
    use_memoryagentbench: bool = False,
    mab_subset: str = "all",
    output_path: str = "reports/evaluation_report.json",
    execution_config: Optional[Dict[str, Any]] = None,
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
    return run_evaluation_with_scenarios(
        scenarios=scenarios,
        benchmark_name=benchmark_name,
        output_path=output_path,
        execution_config=execution_config,
    )


def _resolve_mode_specs(execution_config: Dict[str, Any]) -> List[Dict[str, str]]:
    if execution_config.get("include_conflict_aware_ablations"):
        specs = [
            {"result_key": "conflict_aware_full", "pipeline_mode": "conflict_aware", "conflict_aware_variant": "full"},
            {"result_key": "conflict_aware_no_lineage_edges", "pipeline_mode": "conflict_aware", "conflict_aware_variant": "no_lineage_edges"},
            {"result_key": "conflict_aware_no_query_support", "pipeline_mode": "conflict_aware", "conflict_aware_variant": "no_query_support"},
            {"result_key": "lww", "pipeline_mode": "lww", "conflict_aware_variant": "full"},
            {"result_key": "naive", "pipeline_mode": "naive", "conflict_aware_variant": "full"},
        ]
    else:
        specs = [
            {
                "result_key": "conflict_aware",
                "pipeline_mode": "conflict_aware",
                "conflict_aware_variant": execution_config.get("conflict_aware_variant", "full"),
            },
            {"result_key": "lww", "pipeline_mode": "lww", "conflict_aware_variant": "full"},
            {"result_key": "naive", "pipeline_mode": "naive", "conflict_aware_variant": "full"},
        ]

    selected = execution_config.get("selected_modes")
    if not selected:
        return specs
    selected_set = set(selected)
    filtered = [spec for spec in specs if spec["result_key"] in selected_set]
    return filtered or specs


def _compute_deltas(results: Dict[str, Dict[str, Any]], primary_conflict_aware_key: str) -> Dict[str, Dict[str, float]]:
    deltas = {}
    for metric in ["scenario_accuracy", "conflict_f1", "action_accuracy", "qa_exact_match", "qa_subem"]:
        ca = results.get(primary_conflict_aware_key, {}).get(metric, 0.0)
        lww = results.get("lww", {}).get(metric, 0.0)
        naive = results.get("naive", {}).get(metric, 0.0)
        deltas[metric] = {
            "conflict_aware_minus_lww": ca - lww,
            "conflict_aware_minus_naive": ca - naive,
            "lww_minus_naive": lww - naive,
        }
    return deltas


def _derive_report_track_name(
    execution_config: Dict[str, Any],
    proposal_source_breakdown: Dict[str, int],
) -> str:
    track_name = execution_config.get("track_name", "oracle_structured")
    if track_name != "end_to_end_extract":
        return "oracle_structured"
    fallback_count = int(proposal_source_breakdown.get("end_to_end_extract__structured_fallback", 0))
    pure_extract_count = int(proposal_source_breakdown.get("end_to_end_extract", 0))
    if fallback_count > 0:
        return "end_to_end_extract__structured_fallback"
    if pure_extract_count > 0:
        return "end_to_end_extract"
    return "end_to_end_extract"


def _compute_mode_metrics(
    mode_results: List[Dict[str, Any]],
    scenarios: List[Any],
    execution_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive metrics for a mode."""
    if len(mode_results) != len(scenarios):
        raise ValueError(f"mode_results length ({len(mode_results)}) does not match scenarios length ({len(scenarios)})")

    scenario_dicts = scenarios_to_dicts(scenarios)

    def _normalize_conflict_type(conflict_type: str) -> str:
        """Map detector conflict types to gold taxonomy."""
        gold_types = {"compatible_extension", "mutually_exclusive", "none", "semantic_overlap", "stale_read_conflict", "counterfactual_temporal"}
        if conflict_type in gold_types:
            return conflict_type
        mapping = {
            "exact_duplicate": "semantic_overlap",
            "semantic_duplicate": "semantic_overlap",
            "potential_contradiction": "mutually_exclusive",
            "concurrent_update_conflict": "mutually_exclusive",
            "concurrent_update": "mutually_exclusive",
            "temporal_inconsistency": "counterfactual_temporal",
        }
        return mapping.get(conflict_type, conflict_type)

    def _is_conflict_decision(decision: Dict[str, Any]) -> bool:
        """
        Detect whether a write decision corresponds to an actual conflict event.
        """
        result = decision.get("result", {})
        if result.get("conflict_type") in {"exact_duplicate", "semantic_duplicate"}:
            return False
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
    any_gold_conflicts = any(s.get("gold_conflict_exists", False) for s in scenario_dicts)

    # Action accuracy (only for events with conflict, i.e., candidate_count > 0)
    action_correct = 0
    action_total = 0
    first_write_total = 0
    per_action = {"overwrite": 0, "merge": 0, "keep_multiple_versions": 0, "defer": 0, "reject": 0, "append": 0}
    per_action_gold = {"overwrite": 0, "merge": 0, "keep_multiple_versions": 0, "defer": 0, "reject": 0, "append": 0}
    non_overwrite_predicate_counts: Dict[str, int] = {}
    non_overwrite_action_counts: Dict[str, int] = {}
    non_overwrite_reason_counts: Dict[str, int] = {}
    scenario_diagnostics = {
        "state_match_and_qa_success": 0,
        "state_match_but_qa_failure": 0,
        "state_mismatch_but_qa_success": 0,
        "state_mismatch_and_qa_failure": 0,
        "no_qa": 0,
    }

    # For per-scenario-level metrics
    conflict_detection_correct = 0
    conflict_type_correct = 0

    # Conflict type distribution across decisions
    conflict_type_distribution = {}

    for res, scen_dict in zip(mode_results, scenario_dicts):
        # Scenario-level conflict detection: did we detect any conflict?
        predicted_conflict_exists = res["metrics"]["num_conflicts"] > 0
        gold_conflict_exists = scen_dict.get("gold_conflict_exists", False)
        if predicted_conflict_exists == gold_conflict_exists:
            conflict_detection_correct += 1

        # Scenario-level conflict type: derive predicted type from first conflict decision, else "none"
        predicted_type = "none"
        for dec in res.get("arbitration_decisions", []):
            result = dec.get("result", {})
            if result.get("conflict_detected", False):
                predicted_type = _normalize_conflict_type(result.get("conflict_type", "none"))
                break
        gold_type = scen_dict.get("gold_conflict_type", "none")
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
            gold = scen_dict.get("gold_resolution_action", "append")
            action_total += 1
            if pred == gold:
                action_correct += 1
            else:
                proposal = ev.get("proposal", {}) or {}
                predicate = proposal.get("predicate") or "<unknown>"
                result = ev.get("result", {}) or {}
                arbitration_details = result.get("arbitration_details", {}) or {}
                conflict_details = result.get("conflict_details", {}) or {}
                reason = (
                    arbitration_details.get("reason")
                    or conflict_details.get("reason")
                    or result.get("resolution_reason", {}).get("reason")
                    or "<unknown>"
                )
                action_key = f"{pred}"
                non_overwrite_predicate_counts[predicate] = non_overwrite_predicate_counts.get(predicate, 0) + 1
                non_overwrite_action_counts[action_key] = non_overwrite_action_counts.get(action_key, 0) + 1
                non_overwrite_reason_counts[reason] = non_overwrite_reason_counts.get(reason, 0) + 1
            per_action[pred] = per_action.get(pred, 0) + 1
            per_action_gold[gold] = per_action_gold.get(gold, 0) + 1

        qa_results = res.get("qa_results") or []
        if not qa_results:
            scenario_diagnostics["no_qa"] += 1
        else:
            qa_success = all(bool(item.get("exact_match", False)) for item in qa_results)
            state_match = bool(res["metrics"]["state_match"])
            if state_match and qa_success:
                scenario_diagnostics["state_match_and_qa_success"] += 1
            elif state_match and not qa_success:
                scenario_diagnostics["state_match_but_qa_failure"] += 1
            elif (not state_match) and qa_success:
                scenario_diagnostics["state_mismatch_but_qa_success"] += 1
            else:
                scenario_diagnostics["state_mismatch_and_qa_failure"] += 1

    action_accuracy = action_correct / action_total if action_total else (0.0 if any_gold_conflicts else 1.0)
    conflict_detection_accuracy = conflict_detection_correct / total_scenarios if total_scenarios else 0.0
    conflict_type_accuracy = conflict_type_correct / total_scenarios if total_scenarios else 0.0

    # Conflict detection metrics (precision/recall/F1) already computed
    tp = sum(1 for r, s in zip(mode_results, scenario_dicts) if s.get("gold_conflict_exists") and r["metrics"]["num_conflicts"] > 0)
    fp = sum(1 for r, s in zip(mode_results, scenario_dicts) if not s.get("gold_conflict_exists") and r["metrics"]["num_conflicts"] > 0)
    fn = sum(1 for r, s in zip(mode_results, scenario_dicts) if s.get("gold_conflict_exists") and r["metrics"]["num_conflicts"] == 0)
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

    qa_exact_matches = 0
    qa_subem_matches = 0
    qa_total = 0
    qa_answered = 0
    qa_hop_sum = 0
    fc_sh_total = 0
    fc_sh_correct = 0
    fc_mh_total = 0
    fc_mh_correct = 0
    fc_unmatched_total = 0
    fc_unmatched_correct = 0
    proposal_source_breakdown: Dict[str, int] = {}
    end_to_end_fallback_writes = 0
    for r in mode_results:
        for proposal_event in r.get("write_proposals", []) or []:
            source = proposal_event.get("proposal_source", "unknown")
            proposal_source_breakdown[source] = proposal_source_breakdown.get(source, 0) + 1
            if source == "end_to_end_extract__structured_fallback":
                end_to_end_fallback_writes += 1
        qa_results = r.get("qa_results")
        if qa_results:
            for item in qa_results:
                qa_total += 1
                if item.get("predicted_answers"):
                    qa_answered += 1
                if item.get("exact_match", False):
                    qa_exact_matches += 1
                if item.get("substring_exact_match", False):
                    qa_subem_matches += 1
                qa_hop_sum += int(item.get("hops", 0) or 0)

                query_analysis = analyze_question_requirements(item.get("query", ""))
                relation_hops = len(query_analysis.get("relation_chain", []))
                exact_match = bool(item.get("exact_match", False))
                if relation_hops == 1:
                    fc_sh_total += 1
                    if exact_match:
                        fc_sh_correct += 1
                elif relation_hops >= 2:
                    fc_mh_total += 1
                    if exact_match:
                        fc_mh_correct += 1
                else:
                    fc_unmatched_total += 1
                    if exact_match:
                        fc_unmatched_correct += 1
    qa_exact_match = qa_exact_matches / qa_total if qa_total else 0.0
    qa_subem = qa_subem_matches / qa_total if qa_total else 0.0
    qa_answer_rate = qa_answered / qa_total if qa_total else 0.0
    qa_avg_hops = qa_hop_sum / qa_total if qa_total else 0.0
    fc_sh_accuracy = fc_sh_correct / fc_sh_total if fc_sh_total else 0.0
    fc_mh_accuracy = fc_mh_correct / fc_mh_total if fc_mh_total else 0.0
    fc_unmatched_accuracy = fc_unmatched_correct / fc_unmatched_total if fc_unmatched_total else 0.0

    # Stale read handling accuracy (for scenarios with gold_conflict_type == "stale_read_conflict")
    stale_correct = 0
    stale_total = 0
    for res, scen_dict in zip(mode_results, scenario_dicts):
        if scen_dict.get("gold_conflict_type") == "stale_read_conflict":
            stale_total += 1
            gold_action = scen_dict.get("gold_resolution_action")
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
    for res, scen_dict in zip(mode_results, scenario_dicts):
        # Check if scenario qualifies as temporal update scenario:
        # gold_conflict_type is "mutually_exclusive" and events are ordered (second event later than first)
        if scen_dict.get("gold_conflict_type") == "mutually_exclusive":
            events = scen_dict.get("ordered_events", [])
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

    # Counterfactual temporal accuracy (for scenarios with gold_conflict_type == "counterfactual_temporal")
    counterfactual_correct = 0
    counterfactual_total = 0
    for res, scen_dict in zip(mode_results, scenario_dicts):
        gold_type = scen_dict.get("gold_conflict_type", "none")
        if gold_type == "counterfactual_temporal":
            counterfactual_total += 1
            gold_action = scen_dict.get("gold_resolution_action")
            # Find the first conflict decision
            pred_action = None
            for dec in res.get("arbitration_decisions", []):
                result = dec.get("result", {})
                if result.get("conflict_detected", False):
                    pred_action = dec.get("resolution_action", "append")
                    break
            if pred_action is None:
                pred_action = "append"
            if pred_action == gold_action:
                counterfactual_correct += 1
    counterfactual_accuracy = counterfactual_correct / counterfactual_total if counterfactual_total else 0.0

    # Action Appropriateness Score (weighted by action correctness).
    # Score only true conflict decisions so first writes do not distort the metric.
    # overwrite: +1 (when correct), merge: +0.9, keep_multiple_versions: +0.6 (expensive),
    # defer: +0.5 (fallback), reject: +1, append: -0.5 (when conflict exists)
    action_weights = {
        "overwrite": 1.0,
        "merge": 0.9,
        "keep_multiple_versions": 0.6,
        "defer": 0.5,
        "reject": 1.0,
        "append": -0.5  # Penalty for missing conflict
    }
    appropriateness_sum = 0.0
    appropriateness_total = 0
    for res, scen_dict in zip(mode_results, scenario_dicts):
        for dec in res.get("arbitration_decisions", []):
            if not _is_conflict_decision(dec):
                continue
            pred_action = dec.get("resolution_action", "append")
            gold_action = scen_dict.get("gold_resolution_action", "append")

            # Check if prediction matches gold
            correct = pred_action == gold_action
            weight = action_weights.get(pred_action, 0.0)

            if correct:
                appropriateness_sum += weight
            else:
                # Penalize incorrect actions (use negative of weight or -1 for severe errors)
                if pred_action == "append":
                    appropriateness_sum -= 1.0  # Severe: missed conflict
                else:
                    appropriateness_sum -= 0.5  # Wrong action

            appropriateness_total += 1

    action_appropriateness_score = (
        appropriateness_sum / appropriateness_total if appropriateness_total else (0.0 if any_gold_conflicts else 1.0)
    )

    # Judge-free rate: scenarios that don't require defer (higher is better)
    defer_count = sum(1 for r in mode_results for dec in r.get("arbitration_decisions", [])
                     if dec.get("resolution_action") == "defer")
    judge_free_rate = 1.0 - (defer_count / total_writes) if total_writes else 1.0

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
    for res, scen_dict in zip(mode_results, scenario_dicts):
        gold_state = scen_dict.get("gold_visible_shared_state_after_commit", [])
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
    for res, scen_dict in zip(mode_results, scenario_dicts):
        gold_type = scen_dict.get("gold_conflict_type", "none")
        for ev in res.get("arbitration_decisions", []):
            if _is_conflict_decision(ev):
                pred = ev.get("resolution_action", "append")
                gold = scen_dict.get("gold_resolution_action", "append")
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
    execution_config = execution_config or {}
    reported_track_name = _derive_report_track_name(execution_config, proposal_source_breakdown)

    return {
        "scenario_accuracy": scenario_correct / total_scenarios if total_scenarios else 0.0,
        "conflict_detection_accuracy": conflict_detection_accuracy,
        "conflict_type_accuracy": conflict_type_accuracy,
        "conflict_precision": conflict_precision,
        "conflict_recall": conflict_recall,
        "conflict_f1": conflict_f1,
        "action_accuracy": action_accuracy,
        "action_events": action_total,
        "action_appropriateness_score": action_appropriateness_score,
        "first_writes_skipped": first_write_total,
        "total_writes": total_writes,
        "total_conflicts": total_conflicts,
        "conflict_rate": total_conflicts / total_writes if total_writes else 0.0,
        "action_distribution_pred": per_action,
        "action_distribution_gold": per_action_gold,
        "non_overwrite_predicate_counts": dict(sorted(non_overwrite_predicate_counts.items(), key=lambda item: (-item[1], item[0]))[:15]),
        "non_overwrite_action_counts": dict(sorted(non_overwrite_action_counts.items(), key=lambda item: (-item[1], item[0]))),
        "non_overwrite_reason_counts": dict(sorted(non_overwrite_reason_counts.items(), key=lambda item: (-item[1], item[0]))[:15]),
        "scenario_diagnostics": scenario_diagnostics,
        "per_conflict_type_action_accuracy": per_type_accuracy,
        "retrieval_recall_at_5": retrieval_recall_at_5,
        "qa_exact_match": qa_exact_match,
        "qa_subem": qa_subem,
        "qa_answer_rate": qa_answer_rate,
        "qa_total": qa_total,
        "qa_avg_hops": qa_avg_hops,
        "fc_sh_accuracy": fc_sh_accuracy,
        "fc_sh_correct": fc_sh_correct,
        "fc_sh_total": fc_sh_total,
        "fc_mh_accuracy": fc_mh_accuracy,
        "fc_mh_correct": fc_mh_correct,
        "fc_mh_total": fc_mh_total,
        "fc_unmatched_accuracy": fc_unmatched_accuracy,
        "fc_unmatched_correct": fc_unmatched_correct,
        "fc_unmatched_total": fc_unmatched_total,
        "proposal_source_breakdown": proposal_source_breakdown,
        "end_to_end_fallback_writes": end_to_end_fallback_writes,
        "fallback_contamination_detected": end_to_end_fallback_writes > 0,
        "reported_track_name": reported_track_name,
        "pure_end_to_end_extract": reported_track_name == "end_to_end_extract",
        "structured_fallback_present": end_to_end_fallback_writes > 0,
        "stale_handling_accuracy": stale_handling_accuracy,
        "stale_events": stale_total,
        "temporal_update_accuracy": temporal_update_accuracy,
        "counterfactual_accuracy": counterfactual_accuracy,
        "judge_free_rate": judge_free_rate,
        "final_memory_f1": memory_f1,
        "avg_branch_count": avg_branch_count,
        "conflict_type_distribution": conflict_type_distribution,
        "requires_judge_count": requires_judge_count,
    }


def _compute_per_type_breakdown(
    raw_results: Dict[str, List[Dict[str, Any]]],
    scenarios: List[Any]
) -> Dict[str, Dict[str, float]]:
    """Breakdown action accuracy by scenario type using raw results."""
    breakdown = {}
    scenario_dicts = scenarios_to_dicts(scenarios)

    # Map scenario index to its type
    idx_to_type = {i: s.get("scenario_type", "unknown") for i, s in enumerate(scenario_dicts)}

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
                gold = scenario_dicts[idx].get("gold_resolution_action", "append")
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
    from datetime import datetime, UTC
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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
