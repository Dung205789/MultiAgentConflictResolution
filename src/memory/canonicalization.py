from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import re
import unicodedata


_DIRECT_PATTERNS: Sequence[Tuple[re.Pattern[str], str]] = (
    (re.compile(r"^the chairperson of (.+?) is (.+?)$", re.IGNORECASE), "chairperson"),
    (re.compile(r"^the director of (.+?) is (.+?)$", re.IGNORECASE), "director"),
    (re.compile(r"^the author of (.+?) is (.+?)$", re.IGNORECASE), "author"),
    (re.compile(r"^the chief executive officer of (.+?) is (.+?)$", re.IGNORECASE), "ceo"),
    (re.compile(r"^the capital of (.+?) is (.+?)$", re.IGNORECASE), "capital"),
    (re.compile(r"^the original language of (.+?) is (.+?)$", re.IGNORECASE), "language"),
    (re.compile(r"^the original broadcaster of (.+?) is (.+?)$", re.IGNORECASE), "original_broadcaster"),
    (re.compile(r"^the origianl broadcaster of (.+?) is (.+?)$", re.IGNORECASE), "original_broadcaster"),
    (re.compile(r"^the head coach of (.+?) is (.+?)$", re.IGNORECASE), "head_coach"),
    (re.compile(r"^the president of (.+?) is (.+?)$", re.IGNORECASE), "head_of_state"),
    (re.compile(r"^the governor of (.+?) is (.+?)$", re.IGNORECASE), "governor"),
    (re.compile(r"^the mayor of (.+?) is (.+?)$", re.IGNORECASE), "mayor"),
    (re.compile(r"^the illinois attorney general is (.+?)$", re.IGNORECASE), "attorney_general"),
    (re.compile(r"^the head of the commonwealth is (.+?)$", re.IGNORECASE), "head_of_commonwealth"),
    (re.compile(r"^the minister of external affairs is (.+?)$", re.IGNORECASE), "minister_of_external_affairs"),
    (re.compile(r"^the pope is (.+?)$", re.IGNORECASE), "pope_holder"),
    (re.compile(r"^the t[aá]naiste is (.+?)$", re.IGNORECASE), "tanaiste"),
    (re.compile(r"^the official language of (.+?) is (.+?)$", re.IGNORECASE), "official_language"),
    (re.compile(r"^the name of the current head of state in (.+?) is (.+?)$", re.IGNORECASE), "head_of_state"),
    (re.compile(r"^the current head of state in (.+?) is (.+?)$", re.IGNORECASE), "head_of_state"),
    (re.compile(r"^the name of the current head of the (.+?) government is (.+?)$", re.IGNORECASE), "government_head"),
    (re.compile(r"^the prime minister of (.+?) is (.+?)$", re.IGNORECASE), "prime_minister"),
    (re.compile(r"^the headquarters of (.+?) is located in the city of (.+?)$", re.IGNORECASE), "headquarters_city"),
    (re.compile(r"^the univeristy where (.+?) was educated is (.+?)$", re.IGNORECASE), "educated_at"),
    (re.compile(r"^the university where (.+?) was educated is (.+?)$", re.IGNORECASE), "educated_at"),
    (re.compile(r"^the company that produced (.+?) is (.+?)$", re.IGNORECASE), "producer_company"),
    (re.compile(r"^the company that originally broadcasted (.+?) is (.+?)$", re.IGNORECASE), "original_broadcaster"),
)

_SUBJECT_PATTERNS: Sequence[Tuple[re.Pattern[str], str]] = (
    (re.compile(r"^(.+?) is married to (.+?)$", re.IGNORECASE), "spouse"),
    (re.compile(r"^(.+?) is a citizen of (.+?)$", re.IGNORECASE), "citizenship"),
    (re.compile(r"^(.+?) is affiliated with the religion of (.+?)$", re.IGNORECASE), "religion"),
    (re.compile(r"^(.+?) is associated with the sport of (.+?)$", re.IGNORECASE), "sport"),
    (re.compile(r"^(.+?) plays the position of (.+?)$", re.IGNORECASE), "position"),
    (re.compile(r"^(.+?) was born in the city of (.+?)$", re.IGNORECASE), "birth_place"),
    (re.compile(r"^(.+?) died in the city of (.+?)$", re.IGNORECASE), "death_place"),
    (re.compile(r"^(.+?) was founded by (.+?)$", re.IGNORECASE), "founder"),
    (re.compile(r"^(.+?) was founded in the city of (.+?)$", re.IGNORECASE), "origin_city"),
    (re.compile(r"^(.+?) was created in the country of (.+?)$", re.IGNORECASE), "origin_country"),
    (re.compile(r"^(.+?) was performed by (.+?)$", re.IGNORECASE), "performer"),
    (re.compile(r"^(.+?) was created by (.+?)$", re.IGNORECASE), "creator"),
    (re.compile(r"^(.+?) was developed by (.+?)$", re.IGNORECASE), "producer_company"),
    (re.compile(r"^(.+?) was written in the language of (.+?)$", re.IGNORECASE), "language"),
    (re.compile(r"^(.+?) was written by (.+?)$", re.IGNORECASE), "author"),
    (re.compile(r"^(.+?) is famous for (.+?)$", re.IGNORECASE), "known_for"),
    (re.compile(r"^(.+?) is located in the continent of (.+?)$", re.IGNORECASE), "continent"),
    (re.compile(r"^(.+?) speaks the language of (.+?)$", re.IGNORECASE), "language"),
    (re.compile(r"^(.+?) is employed by (.+?)$", re.IGNORECASE), "employer"),
    (re.compile(r"^(.+?) worked in the city of (.+?)$", re.IGNORECASE), "work_location"),
    (re.compile(r"^(.+?) works in the field of (.+?)$", re.IGNORECASE), "occupation"),
    (re.compile(r"^(.+?)'s child is (.+?)$", re.IGNORECASE), "child"),
    (re.compile(r"^the type of music that (.+?) plays is (.+?)$", re.IGNORECASE), "music_type"),
)

_PREDICATE_ALIASES = {
    "location": "continent",
    "founder_location": "origin_city",
    "is_associated_with": "sport",
    "is_affiliated_with": "religion",
    "affiliated_with": "religion",
    "plays_the_position_of": "position",
    "was_created_in": "origin_country",
    "was_created_in_the_country_of": "origin_country",
    "created_in": "origin_country",
    "was_created_by": "creator",
    "created_by": "creator",
    "developed_by": "producer_company",
    "was_developed_by": "producer_company",
    "produced_by": "producer_company",
    "is_employed_by": "employer",
    "works_in_the_field_of": "occupation",
    "was_educated_at": "educated_at",
    "speaks_the_language_of": "language",
    "was_written_in_the_language_of": "language",
    "headquarters_is_located_in": "headquarters_city",
    "is_famous_for": "known_for",
    "is_married_to": "spouse",
    "headquarters_city": "headquarters_city",
    "original_broadcaster": "original_broadcaster",
    "was_founded_by": "founder",
    "written_by": "author",
    "head_of_state": "government_head",
    "prime_minister": "government_head",
    "head": "government_head",
}

_OBJECT_WRAPPERS = (
    "the city of ",
    "city of ",
    "the country of ",
    "country of ",
    "the continent of ",
    "continent of ",
    "the language of ",
    "language of ",
    "the religion of ",
    "religion of ",
    "the sport of ",
    "sport of ",
)


def canonicalize_subject(subject: Any) -> str:
    text = str(subject or "").strip().rstrip(".")
    return unicodedata.normalize("NFKC", text)


def canonicalize_object_value(value: Any) -> Any:
    text = str(value or "").strip().rstrip(".")
    if not text:
        return value
    lowered = text.lower()
    for wrapper in _OBJECT_WRAPPERS:
        if lowered.startswith(wrapper):
            return text[len(wrapper):].strip()
    return unicodedata.normalize("NFKC", text)


def canonicalize_predicate_name(predicate: Any) -> str:
    pred = str(predicate or "").strip().lower().replace("-", "_")
    pred = re.sub(r"\s+", "_", pred)
    if not pred:
        return "raw_statement"
    return _PREDICATE_ALIASES.get(pred, pred)


def parse_fact_from_raw_text(raw_text: Any) -> Optional[Tuple[str, str, Any]]:
    text = str(raw_text or "").strip().rstrip(".")
    if not text:
        return None
    for pattern, predicate in _DIRECT_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        if match.lastindex == 1:
            synthetic_subject = {
                "attorney_general": "Illinois",
                "head_of_commonwealth": "Commonwealth",
                "minister_of_external_affairs": "India",
                "pope_holder": "Papacy",
                "tanaiste": "Ireland",
            }.get(predicate)
            if synthetic_subject is not None:
                return canonicalize_subject(synthetic_subject), predicate, canonicalize_object_value(match.group(1).strip())
        return (
            canonicalize_subject(match.group(1).strip()),
            predicate,
            canonicalize_object_value(match.group(2).strip()),
        )
    for pattern, predicate in _SUBJECT_PATTERNS:
        match = pattern.match(text)
        if match:
            return (
                canonicalize_subject(match.group(1).strip()),
                predicate,
                canonicalize_object_value(match.group(2).strip()),
            )
    return None


def canonicalize_memory_triplet(
    subject: Any,
    predicate: Any,
    object_val: Any,
    *,
    raw_text: Any = "",
) -> Tuple[str, str, Any]:
    parsed = parse_fact_from_raw_text(raw_text)
    if parsed is not None:
        parsed_subject, parsed_predicate, parsed_object = parsed
        return (
            canonicalize_subject(parsed_subject or "unknown"),
            canonicalize_predicate_name(parsed_predicate),
            canonicalize_object_value(parsed_object),
        )
    return (
        canonicalize_subject(subject or "unknown"),
        canonicalize_predicate_name(predicate),
        canonicalize_object_value(object_val),
    )


def _slot_key_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_entity_id(subject: Any, predicate: Any, *, raw_text: Any = "", object_val: Any = "") -> str:
    canonical_subject, canonical_predicate, _ = canonicalize_memory_triplet(
        subject,
        predicate,
        object_val,
        raw_text=raw_text,
    )
    return f"{_slot_key_text(canonical_subject)}::{canonical_predicate}"


def build_canonical_claim(subject: Any, predicate: Any, object_val: Any, *, raw_text: Any = "") -> str:
    canonical_subject, canonical_predicate, canonical_object = canonicalize_memory_triplet(
        subject,
        predicate,
        object_val,
        raw_text=raw_text,
    )
    return f"{canonical_subject} {canonical_predicate} {canonical_object}"
