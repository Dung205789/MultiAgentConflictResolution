from __future__ import annotations

from typing import Any, Dict, Iterable, Set, Tuple

from src.memory.canonicalization import canonicalize_memory_triplet


CanonicalFact = Tuple[str, str, str]
RawFact = Tuple[str, str, str]


def raw_state_record(record: Dict[str, Any]) -> RawFact:
    return (
        str(record.get("subject", "")),
        str(record.get("predicate", "")),
        str(record.get("object_val", record.get("object", ""))),
    )


def raw_state_facts(records: Iterable[Dict[str, Any]]) -> Set[RawFact]:
    return {raw_state_record(record) for record in (records or [])}


def canonicalize_state_record(record: Dict[str, Any]) -> CanonicalFact:
    subject, predicate, object_val = canonicalize_memory_triplet(
        record.get("subject", ""),
        record.get("predicate", ""),
        record.get("object_val", record.get("object", "")),
        raw_text=record.get("raw_text", ""),
    )
    return (str(subject), str(predicate), str(object_val))


def canonicalize_state_facts(records: Iterable[Dict[str, Any]]) -> Set[CanonicalFact]:
    return {canonicalize_state_record(record) for record in (records or [])}
