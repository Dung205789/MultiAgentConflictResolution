from __future__ import annotations

from typing import Any, Dict, List, Optional


PROPOSAL_ONLY_KEYS = {
    "subject",
    "predicate",
    "object_val",
    "confidence",
    "provenance",
    "rationale",
    "support_spans",
    "extractor_id",
    "challenger_metadata",
}


def _normalize_support_spans(raw_spans: Any) -> List[Dict[str, Any]]:
    if raw_spans is None:
        return []
    normalized: List[Dict[str, Any]] = []
    if isinstance(raw_spans, list):
        for index, item in enumerate(raw_spans):
            if isinstance(item, dict):
                normalized.append(dict(item))
            else:
                normalized.append({"span_text": str(item), "span_index": index})
    else:
        normalized.append({"span_text": str(raw_spans), "span_index": 0})
    return normalized


def normalize_proposal(
    proposal: Dict[str, Any],
    *,
    source_label: str,
    default_extractor_id: str,
    default_provenance: str,
    default_confidence: float = 1.0,
    default_rationale: Optional[str] = None,
) -> Dict[str, Any]:
    normalized = dict(proposal or {})

    if not normalized.get("subject"):
        normalized["subject"] = "unknown"
    if not normalized.get("predicate"):
        normalized["predicate"] = "raw_statement"
    if "object_val" not in normalized:
        normalized["object_val"] = normalized.get("raw_text", "")

    normalized["confidence"] = float(normalized.get("confidence", default_confidence))
    normalized["provenance"] = normalized.get("provenance", default_provenance)
    normalized["support_spans"] = _normalize_support_spans(normalized.get("support_spans"))
    normalized["extractor_id"] = normalized.get("extractor_id", default_extractor_id)
    normalized["rationale"] = normalized.get(
        "rationale",
        default_rationale or f"{source_label}_proposal",
    )
    normalized["challenger_metadata"] = normalized.get("challenger_metadata")
    normalized["_proposal_contract"] = "proposal_only"
    return normalized
