"""
Structured QA error analysis for benchmark-facing symbolic runs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.benchmarks.scenario_contract import scenarios_to_dicts
from src.evaluation.qa_reasoner import (
    _follow_relation,
    _resolve_anchor_node,
    analyze_question_requirements,
    build_memory_graph,
    normalize_answer,
)


def _is_answer_type_compatible(predicted_type: Any, expected_types: Sequence[str]) -> bool:
    predicted = str(predicted_type or "").strip().lower()
    expected = {str(item or "").strip().lower() for item in expected_types if str(item or "").strip()}
    if not predicted or not expected:
        return True
    if predicted in expected:
        return True
    compatibility = {
        "geo": {"city", "country", "continent"},
        "entity": {
            "person",
            "city",
            "country",
            "continent",
            "language",
            "religion",
            "music",
            "institution",
            "org",
            "occupation",
            "position",
            "sport",
            "work",
        },
    }
    for broad, narrow in compatibility.items():
        if predicted == broad and expected & narrow:
            return True
        if predicted in narrow and broad in expected:
            return True
    return False


def _classify_single_failure(
    question: str,
    gold_answers: Sequence[Any],
    predicted_answers: Sequence[Any],
    memories: Sequence[Dict[str, Any]],
    predicted_answer_type: Any = None,
) -> Dict[str, Any]:
    analysis = analyze_question_requirements(question)
    graph, _, _ = build_memory_graph(memories)
    anchor = analysis.get("anchor")
    relation_chain = list(analysis.get("relation_chain", []))
    expected_types = list(analysis.get("expected_types", []))
    anchor_node = _resolve_anchor_node(anchor, graph) if anchor else None
    gold_norm = {normalize_answer(x) for x in gold_answers if normalize_answer(x)}
    predicted_norm = {normalize_answer(x) for x in predicted_answers if normalize_answer(x)}

    detail: Dict[str, Any] = {
        "category": "overwrite_correct_but_qa_unused",
        "anchor": anchor,
        "anchor_node": anchor_node,
        "relation_chain": relation_chain,
        "expected_types": expected_types,
        "predicted_answer_type": predicted_answer_type,
        "predicted_answers": list(predicted_answers or []),
        "gold_answers": list(gold_answers or []),
    }

    if not relation_chain:
        detail["category"] = "wrong_anchor_resolution"
        detail["reason"] = "question_template_unmatched"
        return detail

    if anchor and not anchor_node:
        detail["category"] = "wrong_anchor_resolution"
        detail["reason"] = "anchor_not_resolved_in_graph"
        return detail

    if not anchor_node:
        detail["category"] = "wrong_anchor_resolution"
        detail["reason"] = "no_anchor_node_for_symbolic_chain"
        return detail

    current_nodes = [anchor_node]
    for step_index, relation in enumerate(relation_chain):
        next_nodes, fragments = _follow_relation(graph, current_nodes, relation)
        if next_nodes:
            detail.setdefault("resolved_steps", []).append(
                {
                    "step": step_index,
                    "relation": relation,
                    "nodes": list(current_nodes),
                    "next_nodes": list(next_nodes),
                    "path_fragments": list(fragments),
                }
            )
            current_nodes = next_nodes
            continue

        available_relations = sorted(
            {
                edge.relation
                for node in current_nodes
                for edge in graph.get(node, [])
            }
        )
        reverse_relation = relation[4:] if relation.startswith("rev_") else f"rev_{relation}"
        detail["failing_step"] = step_index
        detail["failing_relation"] = relation
        detail["current_nodes"] = list(current_nodes)
        detail["available_relations"] = available_relations

        if reverse_relation in available_relations or any(rel.startswith("rev_") for rel in available_relations):
            detail["category"] = "wrong_reverse_relation"
            detail["reason"] = "reverse_relation_available_but_expected_direction_missing"
            return detail

        if not available_relations:
            detail["category"] = "parser_no_edge"
            detail["reason"] = "no_outgoing_edges_from_current_nodes"
            return detail

        if step_index == len(relation_chain) - 1:
            detail["category"] = "missing_terminal_edge"
            detail["reason"] = "terminal_relation_missing_from_graph"
            return detail

        detail["category"] = "parser_no_edge"
        detail["reason"] = "intermediate_relation_missing_from_graph"
        return detail

    resolved_norm = {normalize_answer(node) for node in current_nodes if normalize_answer(node)}
    detail["resolved_terminal_nodes"] = list(current_nodes)
    if gold_norm and resolved_norm & gold_norm:
        detail["category"] = "overwrite_correct_but_qa_unused"
        detail["reason"] = "gold_answer_reachable_but_not_selected"
        return detail

    if predicted_norm and not (predicted_norm & gold_norm):
        if not _is_answer_type_compatible(predicted_answer_type, expected_types):
            detail["category"] = "answer_type_mismatch"
            detail["reason"] = "predicted_answer_type_does_not_match_expected_question_type"
            return detail
        detail["category"] = "overwrite_correct_but_qa_unused"
        detail["reason"] = "qa_selected_wrong_reachable_answer"
        return detail

    detail["reason"] = "reachable_path_did_not_match_gold"
    return detail


def build_error_analysis_report(
    raw_results: Dict[str, List[Dict[str, Any]]],
    scenarios: Sequence[Any],
) -> Dict[str, Any]:
    scenario_dicts = scenarios_to_dicts(scenarios)

    report: Dict[str, Any] = {}
    for mode, results_list in raw_results.items():
        summary_counts: Dict[str, int] = {}
        detailed_failures: List[Dict[str, Any]] = []
        for scenario_dict, result in zip(scenario_dicts, results_list):
            qa_results = result.get("qa_results") or []
            if not qa_results:
                continue
            final_visible = result.get("final_visible_state", [])
            scenario_failures: List[Dict[str, Any]] = []
            for qa_item in qa_results:
                if qa_item.get("exact_match", False):
                    continue
                classified = _classify_single_failure(
                    qa_item.get("query", ""),
                    qa_item.get("gold", []),
                    qa_item.get("predicted_answers", []),
                    final_visible,
                    qa_item.get("answer_type"),
                )
                summary_counts[classified["category"]] = summary_counts.get(classified["category"], 0) + 1
                scenario_failures.append(
                    {
                        "query": qa_item.get("query", ""),
                        "gold": qa_item.get("gold", []),
                        "predicted_answers": qa_item.get("predicted_answers", []),
                        "path": qa_item.get("path", []),
                        **classified,
                    }
                )

            if scenario_failures:
                detailed_failures.append(
                    {
                        "scenario_id": scenario_dict.get("scenario_id"),
                        "scenario_type": scenario_dict.get("scenario_type"),
                        "num_failures": len(scenario_failures),
                        "failures": scenario_failures,
                    }
                )

        report[mode] = {
            "summary_counts": summary_counts,
            "detailed_failures": detailed_failures,
        }

    return report
