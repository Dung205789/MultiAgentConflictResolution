#!/usr/bin/env python
"""
Main entry point for multi-agent memory conflict resolution evaluation.

This unified runner supports:
- Multiple benchmarks (MemAE, MemoryAgentBench, LoCoMo, custom)
- All writer modes (conflict_aware, lww, naive)
- Primary `oracle_structured` track and secondary `end_to_end_extract` track
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
import re
from datetime import datetime, UTC
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.run_evaluation import run_evaluation_with_scenarios
from src.evaluation.run_evaluation import finalize_report_from_scenario_artifacts
from src.benchmarks.unified_loader import load_benchmark, save_scenarios_to_jsonl
from src.benchmarks.scenario_contract import scenario_to_dict


VALID_RESULT_MODES = {
    "conflict_aware",
    "conflict_aware_full",
    "conflict_aware_no_lineage_edges",
    "conflict_aware_no_query_support",
    "lww",
    "naive",
}


def resolve_execution_device(preference: str) -> str:
    """Resolve requested runtime device into `cpu` or `cuda`."""
    if preference in {"cpu", "cuda"}:
        return preference

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def detect_cuda_device_count() -> int:
    """Return the number of visible CUDA devices."""
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def resolve_agent_model_type(model_name: str) -> str:
    normalized = (model_name or "").strip().lower()
    if normalized.startswith("gemini"):
        return "gemini_api"
    if normalized.startswith("gpt-") or normalized.startswith("o1") or normalized.startswith("o3") or normalized.startswith("o4"):
        return "openai_api"
    return "transformer"


def _sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def resolve_extraction_cache_path(args: argparse.Namespace, use_model_reextract: bool) -> str:
    if not use_model_reextract:
        return ""
    configured_cache_path = getattr(args, "extraction_cache_path", "")
    if configured_cache_path:
        return os.path.abspath(configured_cache_path)

    preferred_model = args.agent1_model or args.agent2_model or "extractor"
    preferred_type = resolve_agent_model_type(preferred_model)
    cache_dir = os.path.join(PROJECT_ROOT, "reports", "extraction_cache")
    cache_name = f"{preferred_type}_{_sanitize_path_component(preferred_model)}.jsonl"
    return os.path.join(cache_dir, cache_name)


def build_execution_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate CLI agent flags into a concrete execution configuration."""
    if args.use_dummy and (args.agent1_model or args.agent2_model):
        raise ValueError("`--use-dummy` cannot be combined with `--agent1-model` or `--agent2-model`.")

    use_model_reextract = bool(args.agent1_model or args.agent2_model)
    requested_track = args.track
    if requested_track is None:
        requested_track = "end_to_end_extract" if use_model_reextract else "oracle_structured"

    if requested_track == "end_to_end_extract" and not use_model_reextract:
        raise ValueError("`--track end_to_end_extract` requires `--agent1-model` and/or `--agent2-model`.")

    if requested_track == "oracle_structured" and use_model_reextract:
        raise ValueError("Model re-extraction cannot run under `--track oracle_structured`.")

    proposal_source = "agent_extract" if requested_track == "end_to_end_extract" else "structured"
    strict_agent_execution = requested_track == "end_to_end_extract"
    extraction_cache_path = resolve_extraction_cache_path(args, use_model_reextract)
    resolved_device = resolve_execution_device(args.device)
    cuda_device_count = detect_cuda_device_count() if resolved_device == "cuda" else 0
    primary_device = "cpu"
    secondary_device = "cpu"
    if resolved_device == "cuda":
        primary_device = "cuda:0"
        secondary_device = "cuda:1" if cuda_device_count > 1 else "cuda:0"

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
        agent1_model = args.agent1_model or "gemini-2.5-flash-lite"
        agent2_model = args.agent2_model or args.agent1_model or agent1_model
        agent1_model_type = resolve_agent_model_type(agent1_model)
        agent2_model_type = resolve_agent_model_type(agent2_model)
        agent_configs["__slot_0__"].update({
            "model_type": agent1_model_type,
            "model_name": agent1_model,
            "device": primary_device,
            "extraction_cache_path": extraction_cache_path,
        })
        agent_configs["__slot_1__"].update({
            "model_type": agent2_model_type,
            "model_name": agent2_model,
            "device": secondary_device,
            "extraction_cache_path": extraction_cache_path,
        })
        if agent1_model_type == "transformer":
            agent_configs["__slot_0__"]["quantization_mode"] = "4bit" if resolved_device == "cuda" else "none"
        if agent2_model_type == "transformer":
            agent_configs["__slot_1__"]["quantization_mode"] = "4bit" if resolved_device == "cuda" else "none"
    else:
        agent_configs["__slot_0__"]["model_type"] = "structured"
        agent_configs["__slot_1__"]["model_type"] = "structured"

    selected_modes = None
    if args.modes:
        selected_modes = [item.strip() for item in args.modes.split(",") if item.strip()]
        invalid = [item for item in selected_modes if item not in VALID_RESULT_MODES]
        if invalid:
            raise ValueError(f"Unsupported values in --modes: {', '.join(invalid)}")
        if args.include_conflict_aware_ablations:
            if "conflict_aware" in selected_modes:
                raise ValueError("When --include-conflict-aware-ablations is set, use ablation result keys such as conflict_aware_full instead of conflict_aware in --modes.")
        else:
            disallowed = {
                "conflict_aware_full",
                "conflict_aware_no_lineage_edges",
                "conflict_aware_no_query_support",
            } & set(selected_modes)
            if disallowed:
                raise ValueError("Ablation-only mode keys require --include-conflict-aware-ablations.")

    return {
        "proposal_source": proposal_source,
        "strict_agent_execution": strict_agent_execution,
        "track_name": requested_track,
        "agent_configs": agent_configs,
        "device": resolved_device,
        "cuda_device_count": cuda_device_count,
        "mode_label": requested_track,
        "conflict_aware_variant": args.conflict_aware_variant,
        "include_conflict_aware_ablations": args.include_conflict_aware_ablations,
        "enable_error_analysis": args.enable_error_analysis,
        "emit_scenario_bundles": args.emit_scenario_bundles,
        "allow_structured_fallback_in_end_to_end": args.allow_structured_fallback_in_end_to_end,
        "selected_modes": selected_modes,
        "extraction_cache_path": extraction_cache_path,
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
    ordered_modes = [
        "conflict_aware",
        "conflict_aware_full",
        "conflict_aware_no_lineage_edges",
        "conflict_aware_no_query_support",
        "lww",
        "naive",
    ]
    printed = set()
    for mode in ordered_modes:
        if mode == "conflict_aware" and "conflict_aware_full" in report["results"]:
            continue
        if mode in report["results"]:
            res = report["results"][mode]
            if mode in printed:
                continue
            printed.add(mode)
            print(f"{mode:17} | Acc: {res['scenario_accuracy']:.3f} | "
                  f"F1: {res['conflict_f1']:.3f} | "
                  f"Action: {res['action_accuracy']:.3f} | "
                  f"Mem F1: {res['final_memory_f1']:.3f}")
            if res.get("qa_total", 0):
                print(
                    f"{'':17}   QA-EM: {res.get('qa_exact_match', 0.0):.3f} | "
                    f"QA-SubEM: {res.get('qa_subem', 0.0):.3f}"
                )
            if res.get("fc_sh_total", 0) or res.get("fc_mh_total", 0):
                print(
                    f"{'':17}   FC-SH: {res.get('fc_sh_accuracy', 0.0):.3f} "
                    f"({res.get('fc_sh_correct', 0)}/{res.get('fc_sh_total', 0)}) | "
                    f"FC-MH: {res.get('fc_mh_accuracy', 0.0):.3f} "
                    f"({res.get('fc_mh_correct', 0)}/{res.get('fc_mh_total', 0)}) | "
                    f"Unmatched: {res.get('fc_unmatched_total', 0)}"
                )

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


def collect_unique_extraction_texts(scenarios: List[Any]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for scenario in scenarios:
        scenario_dict = scenario_to_dict(scenario)
        for event in scenario_dict.get("ordered_events", []) or []:
            if event.get("event_type") != "write_proposal":
                continue
            proposal = event.get("proposal", {}) or {}
            seed_text = (
                proposal.get("raw_text")
                or event.get("text")
                or f"{proposal.get('subject', '')} {proposal.get('predicate', '')} {proposal.get('object_val', '')}".strip()
            )
            seed_text = str(seed_text or "").strip()
            if not seed_text or seed_text in seen:
                continue
            seen.add(seed_text)
            ordered.append(seed_text)
    return ordered


def warm_extraction_cache(
    scenarios: List[Any],
    execution_config: Dict[str, Any],
    *,
    output_dir: str,
) -> Dict[str, Any]:
    from src.local_models.runner import create_agent

    unique_texts = collect_unique_extraction_texts(scenarios)
    extractor_specs: Dict[tuple, Dict[str, Any]] = {}
    for cfg in execution_config.get("agent_configs", {}).values():
        model_type = cfg.get("model_type")
        if model_type not in {"transformer", "gemini_api", "openai_api"}:
            continue
        spec_key = (
            model_type,
            cfg.get("model_name"),
            cfg.get("device"),
            cfg.get("quantization_mode"),
            cfg.get("extraction_cache_path"),
        )
        if spec_key not in extractor_specs:
            extractor_specs[spec_key] = dict(cfg)

    if not extractor_specs:
        raise ValueError("Extraction cache warm-up requires a real extractor model configuration.")

    summary: Dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "unique_texts": len(unique_texts),
        "extractors": [],
    }
    print(f"\nWarming extraction cache for {len(unique_texts)} unique write texts...")

    for idx, cfg in enumerate(extractor_specs.values(), start=1):
        agent = create_agent(
            agent_id=f"warm_cache_agent_{idx}",
            model_type=cfg.get("model_type"),
            model_name=cfg.get("model_name"),
            device=cfg.get("device", "cpu"),
            quantization_mode=cfg.get("quantization_mode"),
            strict_loading=True,
            extraction_cache_path=cfg.get("extraction_cache_path"),
        )
        print(
            f"  Extractor {idx}/{len(extractor_specs)}: "
            f"{cfg.get('model_type')} {cfg.get('model_name')} -> {cfg.get('extraction_cache_path')}"
        )
        for text_index, text in enumerate(unique_texts, start=1):
            if text_index == 1 or text_index % 250 == 0 or text_index == len(unique_texts):
                print(f"    text {text_index}/{len(unique_texts)}")
            agent.extract_memories(text)
        stats = getattr(agent, "get_extraction_cache_stats", lambda: {})()
        summary["extractors"].append(
            {
                "model_type": cfg.get("model_type"),
                "model_name": cfg.get("model_name"),
                "cache_path": cfg.get("extraction_cache_path"),
                "stats": stats,
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "extraction_cache_warmup.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Extraction cache warm-up summary saved to {summary_path}")
    return summary


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
        help="Legacy alias for the primary `oracle_structured` track. Structured benchmark proposals are used without model re-extraction."
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        choices=["oracle_structured", "end_to_end_extract"],
        help="Research-facing execution track. Omit to infer from legacy flags."
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
        help="Optional local transformer model for agent 2. When set, the runner can execute the secondary `end_to_end_extract` track."
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
    parser.add_argument(
        "--conflict-aware-variant",
        type=str,
        default="full",
        choices=["full", "no_lineage_edges", "no_query_support"],
        help="Conflict-aware symbolic variant to run when ablations are not expanded."
    )
    parser.add_argument(
        "--include-conflict-aware-ablations",
        action="store_true",
        help="Run conflict-aware full plus no-lineage and no-query-support ablations alongside LWW and naive."
    )
    parser.add_argument(
        "--enable-error-analysis",
        action="store_true",
        help="Emit QA error-analysis artifacts for benchmark runs that include queries."
    )
    parser.add_argument(
        "--emit-scenario-bundles",
        action="store_true",
        help="Emit per-scenario JSON artifacts with final visible state, QA failures, and conflict decisions."
    )
    parser.add_argument(
        "--allow-structured-fallback-in-end-to-end",
        action="store_true",
        help="Allow explicit structured fallback when extraction fails. This is labeled separately and must not be reported as pure end-to-end extraction."
    )
    parser.add_argument(
        "--extraction-cache-path",
        type=str,
        default=None,
        help="Persistent JSONL cache for secondary-track extraction results. Default: reports/extraction_cache/<model>.jsonl"
    )
    parser.add_argument(
        "--warm-extraction-cache-only",
        action="store_true",
        help="Phase 1 only: pre-extract unique write raw_text inputs into the persistent extraction cache, then exit without running benchmark evaluation."
    )
    parser.add_argument(
        "--warm-extraction-cache-before-run",
        action="store_true",
        help="Warm the persistent extraction cache from unique write raw_text inputs before benchmark evaluation."
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Optional comma-separated subset of result modes to run, e.g. conflict_aware,lww or conflict_aware_full,lww."
    )
    parser.add_argument(
        "--finalize-report-from-artifacts",
        type=str,
        default=None,
        help="Recover and write the final report at the given report path by rebuilding it from .partial.json and .scenarios artifacts."
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

    if (args.warm_extraction_cache_only or args.warm_extraction_cache_before_run) and execution_config.get("proposal_source") != "agent_extract":
        print("Error: extraction cache warm-up requires the secondary `end_to_end_extract` track with real extractor models.")
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
            s = scenario_to_dict(scenarios[0])
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
                    t = scenario_to_dict(scen).get('scenario_type', 'unknown')
                    type_dist[t] = type_dist.get(t, 0) + 1
                print(f"  Types: {type_dist}")

            if args.warm_extraction_cache_only or args.warm_extraction_cache_before_run:
                warm_summary = warm_extraction_cache(
                    scenarios,
                    execution_config,
                    output_dir=args.output_dir,
                )
                if args.warm_extraction_cache_only:
                    all_reports[benchmark_name] = {
                        "num_scenarios": len(scenarios),
                        "cache_warmup": warm_summary,
                    }
                    print(f"[OK] Warmed extraction cache only for {benchmark_name}; skipping benchmark evaluation.")
                    continue

            if args.finalize_report_from_artifacts:
                print(f"\nFinalizing report from artifacts: {args.finalize_report_from_artifacts}")
                report = finalize_report_from_scenario_artifacts(
                    scenarios=scenarios,
                    benchmark_name=benchmark_name,
                    output_path=args.finalize_report_from_artifacts,
                    execution_config=execution_config,
                )
                print(f"[OK] Recovered final report to {args.finalize_report_from_artifacts}")
            else:
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
