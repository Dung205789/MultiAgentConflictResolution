"""
Query-aware context extraction for answer-critical arbitration.

This module bridges the symbolic QA surface and the conflict-resolution core.
It annotates structured proposals with graph/query signals before arbitration.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

from src.evaluation.qa_reasoner import analyze_question_requirements, extract_graph_edges_from_memory


def _extract_extended_graph_edges_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Conflict-aware extended edge extraction.

    This intentionally goes beyond the base QA parser and is one of the
    symbolic advantages of the conflict-aware path over plain LWW.
    """
    clean = str(text or "").strip().rstrip(".")
    if not clean:
        return []

    patterns = (
        (
            re.compile(r"^(.+?) was written in the language of (.+?)$", re.IGNORECASE),
            "language",
            "language",
        ),
        (
            re.compile(r"^the official documents of (.+?) are written in the language of (.+?)$", re.IGNORECASE),
            "official_language",
            "language",
        ),
        (
            re.compile(r"^the official language of (.+?) is (.+?)$", re.IGNORECASE),
            "official_language",
            "language",
        ),
        (
            re.compile(r"^the headquarters of (.+?) is located in the city of (.+?)$", re.IGNORECASE),
            "headquarters_city",
            "city",
        ),
        (
            re.compile(r"^(.+?) was created in the country of (.+?)$", re.IGNORECASE),
            "origin_country",
            "country",
        ),
        (
            re.compile(r"^the univeristy where (.+?) was educated is (.+?)$", re.IGNORECASE),
            "educated_at",
            "institution",
        ),
        (
            re.compile(r"^the university where (.+?) was educated is (.+?)$", re.IGNORECASE),
            "educated_at",
            "institution",
        ),
        (
            re.compile(r"^(.+?) is famous for (.+?)$", re.IGNORECASE),
            "known_for",
            "work",
        ),
    )
    out: List[Dict[str, Any]] = []
    for pattern, relation, output_type in patterns:
        match = pattern.match(clean)
        if not match:
            continue
        out.append(
            {
                "source": match.group(1).strip(),
                "relation": relation,
                "target": match.group(2).strip(),
                "output_type": output_type,
            }
        )
    return out


def build_query_plan(queries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compile benchmark queries into a lightweight query plan."""
    plan: List[Dict[str, Any]] = []
    for idx, query in enumerate(queries or []):
        if hasattr(query, "query_text"):
            query_text = query.query_text
        else:
            query_text = query.get("query_text", "")
        if not query_text:
            continue
        analysis = analyze_question_requirements(query_text)
        analysis["query_id"] = f"query_{idx}"
        plan.append(analysis)
    return plan


def summarize_query_plan(query_plan: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact query-plan summary for report metadata."""
    relation_counts: Dict[str, int] = {}
    multi_hop_queries = 0
    for item in query_plan:
        if item.get("multi_hop"):
            multi_hop_queries += 1
        for relation in item.get("relation_chain", []):
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
    return {
        "num_queries": len(query_plan),
        "multi_hop_queries": multi_hop_queries,
        "relation_counts": relation_counts,
    }


def annotate_proposal_with_query_context(
    proposal: Dict[str, Any],
    query_plan: Sequence[Dict[str, Any]],
    variant: str = "full",
) -> Dict[str, Any]:
    """
    Inject answer-critical graph/query signals into a structured proposal.

    These signals are later consumed by the rule-based writer to avoid pure
    last-write-wins behavior on multi-hop-critical facts.
    """
    annotated = dict(proposal)
    base_edges = extract_graph_edges_from_memory(annotated)
    extra_edges = _extract_extended_graph_edges_from_text(annotated.get("raw_text", ""))
    edges = list(base_edges)
    seen = {(e["source"], e["relation"], e["target"], e["output_type"]) for e in edges}
    for edge in extra_edges:
        sig = (edge["source"], edge["relation"], edge["target"], edge["output_type"])
        if sig not in seen:
            seen.add(sig)
            edges.append(edge)
    annotated["graph_edges"] = edges
    if variant == "no_query_support" or not query_plan:
        annotated.setdefault("answer_criticality", 0.0)
        annotated.setdefault("graph_support_score", 0.0)
        annotated.setdefault("query_support_ids", [])
        annotated.setdefault("query_relation_roles", [])
        annotated.setdefault("graph_cluster_id", None)
        return annotated

    relations = [edge["relation"] for edge in edges]
    subjects = {str(edge["source"]).strip().lower() for edge in edges}
    targets = {str(edge["target"]).strip().lower() for edge in edges}
    output_types = {str(edge["output_type"]).strip().lower() for edge in edges}

    support_ids: List[str] = []
    support_roles: List[str] = []
    relation_matches = 0
    anchor_matches = 0
    type_matches = 0
    bridge_bonus = 0.0
    cluster_fragments: List[str] = []

    for item in query_plan:
        relation_chain = list(item.get("relation_chain", []))
        anchor = str(item.get("anchor") or "").strip().lower()
        expected_types = {str(x).strip().lower() for x in item.get("expected_types", [])}

        local_relation_match = False
        local_anchor_match = False
        local_type_match = False
        local_roles: List[str] = []

        for relation in relations:
            if relation in relation_chain:
                relation_matches += 1
                local_relation_match = True
                relation_idx = relation_chain.index(relation)
                if relation_idx == 0:
                    local_roles.append("anchor_edge")
                elif relation_idx == len(relation_chain) - 1:
                    local_roles.append("terminal_edge")
                else:
                    local_roles.append("bridge_edge")
                    bridge_bonus += 0.5

        if anchor and (anchor in subjects or anchor in targets):
            anchor_matches += 1
            local_anchor_match = True

        if expected_types and output_types & expected_types:
            type_matches += 1
            local_type_match = True

        if local_relation_match or local_anchor_match or local_type_match:
            support_ids.append(item["query_id"])
            support_roles.extend(local_roles or ["supporting_edge"])
            cluster_fragments.append(item["query_id"])

    unique_support_ids = list(dict.fromkeys(support_ids))
    unique_roles = list(dict.fromkeys(support_roles))
    support_ratio = len(unique_support_ids) / max(len(query_plan), 1)

    answer_criticality = min(
        1.0,
        (
            0.45 * min(1.0, relation_matches / max(len(relations), 1))
            + 0.25 * min(1.0, anchor_matches / max(len(query_plan), 1))
            + 0.20 * support_ratio
            + 0.10 * min(1.0, type_matches / max(len(query_plan), 1))
            + 0.05 * bridge_bonus
        ),
    )
    graph_support_score = min(
        1.0,
        (
            0.50 * support_ratio
            + 0.30 * min(1.0, relation_matches / max(len(query_plan), 1))
            + 0.20 * min(1.0, bridge_bonus)
        ),
    )

    if cluster_fragments and subjects:
        graph_cluster_id = f"{sorted(subjects)[0]}::{'+'.join(sorted(set(cluster_fragments)))}"
    else:
        graph_cluster_id = None

    annotated["answer_criticality"] = round(answer_criticality, 6)
    annotated["graph_support_score"] = round(graph_support_score, 6)
    annotated["query_support_ids"] = unique_support_ids
    annotated["query_relation_roles"] = unique_roles
    annotated["graph_cluster_id"] = graph_cluster_id
    return annotated
