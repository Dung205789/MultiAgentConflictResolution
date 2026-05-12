"""
Enhanced conflict detection module with principled, comprehensive coverage.

Design principles:
1. Rule-based first: Fast, deterministic, explainable
2. Semantic fallback: When rules inconclusive, use similarity
3. Complete coverage: Every possible case handled
"""
from typing import Dict, Any, List, Tuple, Optional
import json
import re

# Don't import sentence_transformers at module level
HAVE_EMBEDDINGS = False
SentenceTransformer = None
cosine_similarity = None

# Lazy import for retriever
_overlap_score_func = None

def _get_overlap_score():
    global _overlap_score_func
    if _overlap_score_func is None:
        from src.utils.retriever import overlap_score
        _overlap_score_func = overlap_score
    return _overlap_score_func


# Thresholds
DUPLICATE_THRESHOLD = 0.95
SEMANTIC_OVERLAP_THRESHOLD = 0.5  # lowered from 0.7 for better semantic_overlap detection
COMPATIBLE_THRESHOLD = 0.5
CONTRADICTION_THRESHOLD = 0.3

# Embedding model (lazy loaded)
_EMBED_MODEL = None


def _try_import_embeddings():
    global HAVE_EMBEDDINGS, SentenceTransformer, cosine_similarity
    if HAVE_EMBEDDINGS:
        return True
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        HAVE_EMBEDDINGS = True
        return True
    except ImportError:
        HAVE_EMBEDDINGS = False
        return False


def _load_embed_model():
    global _EMBED_MODEL
    if not _try_import_embeddings():
        return None
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return _EMBED_MODEL
    except Exception:
        return None


def semantic_similarity(text1: str, text2: str, mode: str = "debug_fallback") -> float:
    """Calculate semantic similarity with embedding fallback to lexical."""
    if _try_import_embeddings():
        model = _load_embed_model()
        if model is not None:
            try:
                embeddings = model.encode([text1, text2], normalize_embeddings=True)
                sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
                return sim
            except Exception:
                pass
    if mode == "research_strict":
        raise RuntimeError("Semantic similarity requires sentence-transformers in research_strict mode")
    overlap_func = _get_overlap_score()
    return overlap_func(text1, text2)


# Predicate classification for mutually exclusive detection
MUTUALLY_EXCLUSIVE_PREDICATES = {
    # Location and temporal
    "city", "study_time", "location", "current_location", "current_task",
    "current_focus", "current_activity", "current_mood", "current_status",
    "born_in", "death_place", "place_of_birth", "place_of_death",
    # Identity (single-valued)
    "ssn", "email", "phone", "passport_number", "id_number", "birth_date", "death_date",
    # Position/title (typically singular)
    "chairperson", "president", "director", "ceo", "founder", "author",
    "nationality", "primary_role", "main_focus",
    # Measurements
    "height", "weight", "age", "salary", "price", "cost", "temperature",
    # Financial identifiers
    "bank_account", "routing_number", "credit_card_number", "iban",
    # Binary states
    "is_married", "is_employed", "is_student", "employment_status",
    # Time-bound single states
    "current_company", "current_school", "current_project",
}

ADDITIVE_PREDICATES = {
    "skill", "skills", "interest", "interests", "hobby", "hobbies",
    "language", "languages", "known_for", "achievement", "achievements",
    "publication", "publications", "project", "projects", "tool", "tools",
    "friend", "friends", "colleague", "colleagues", "certification", "certifications",
    "role", "roles",  # can have multiple roles
    "technology", "technologies",  # overlapping tech stacks
    "product", "products",  # companies can have multiple products
}

# Copular verbs that need object-level analysis
COPULAR_VERBS = {"is", "was", "are", "were", "be", "been", "being"}

# Single-value cues in natural language
SINGLE_VALUE_CUES = {
    "born in", "located in", "located at", "capital of",
    "chairperson", "president", "ceo", "director", "founder",
    "nationality", "plays", "died in", "works at", "works as",
    "worked at", "worked as", "language of", "married to",
    "a citizen of", "citizen of", "famous for", "employed by",
    "educated is", "created by", "performed by", "hails from",
}


def is_mutually_exclusive_predicate(predicate: str) -> bool:
    """Check if predicate is inherently mutually exclusive."""
    pred = predicate.lower()
    return pred in MUTUALLY_EXCLUSIVE_PREDICATES


def is_additive_predicate(predicate: str) -> bool:
    """Check if predicate is additive (can have multiple values)."""
    pred = predicate.lower()
    return pred in ADDITIVE_PREDICATES


def _tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    return [tok for tok in re.split(r"[^a-z0-9]+", (text or "").lower()) if tok]


def _looks_like_single_valued_claim(old_obj: str, new_obj: str) -> bool:
    """
    Heuristic for detecting single-valued claims.
    Two different values for similar template likely indicate mutual exclusion.
    """
    old_norm = (old_obj or "").strip().lower()
    new_norm = (new_obj or "").strip().lower()
    if not old_norm or not new_norm or old_norm == new_norm:
        return False

    # Check for single-value cues
    if any(cue in old_norm or cue in new_norm for cue in SINGLE_VALUE_CUES):
        return True

    # Token overlap analysis
    old_tokens = _tokenize(old_norm)
    new_tokens = _tokenize(new_norm)
    if len(old_tokens) >= 3 and len(new_tokens) >= 3:
        overlap = len(set(old_tokens) & set(new_tokens)) / max(len(old_tokens), len(new_tokens))
        if overlap >= 0.6 and abs(len(old_tokens) - len(new_tokens)) <= 3:
            return True

    # Conservative: default to True for conflicting atomic claims
    return True


def _is_concurrent_update(proposal: Dict[str, Any], candidates: List[Dict[str, Any]]) -> bool:
    """
    Detect concurrent updates: multiple writes without intervening reads.
    Candidates are all with same (subject, predicate), ordered by commit time.
    Concurrent if timestamps are very close (< 1 second apart).
    """
    if len(candidates) < 1:
        return False

    # Get proposal timestamp
    prop_ts = proposal.get("timestamp", 0)
    latest_ts = candidates[-1].get("timestamp", 0) if candidates else 0

    # If timestamps are within 1 second, consider concurrent
    if prop_ts and latest_ts and abs(prop_ts - latest_ts) < 1.0:
        return True

    # Also check committed_at if available
    prop_committed = proposal.get("committed_at", 0)
    latest_committed = candidates[-1].get("committed_at", 0) if candidates else 0
    if prop_committed and latest_committed and abs(prop_committed - latest_committed) < 1.0:
        return True

    return False


def _is_temporal_inconsistency(proposal: Dict[str, Any], candidates: List[Dict[str, Any]]) -> bool:
    """
    Detect temporal inconsistency: newer write contradicts older without staleness.
    This is a special case of mutually exclusive where order matters.
    """
    if not candidates:
        return False

    latest = candidates[-1]
    prop_ts = proposal.get("timestamp", 0)
    latest_ts = latest.get("timestamp", 0)

    # If proposal is newer than latest (out of order), could be temporal inconsistency
    if prop_ts > latest_ts:
        # Check if they're mutually exclusive values
        old_obj = str(latest.get("object_val", ""))
        new_obj = str(proposal.get("object_val", ""))
        if old_obj != new_obj:
            predicate = proposal.get("predicate", "")
            if is_likely_mutually_exclusive(predicate, old_obj, new_obj):
                return True

    return False


def is_likely_mutually_exclusive(predicate: str, old_obj: str, new_obj: str) -> bool:
    """
    Determine if two values for the same predicate are mutually exclusive.
    """
    pred = (predicate or "").strip().lower()

    # Check explicit mutually exclusive predicates
    if is_mutually_exclusive_predicate(pred):
        return True

    # Check strongly single-valued predicates
    strongly_single_valued = {
        "plays", "died", "works", "worked", "speaks the language of",
        "is", "was", "are", "were",  # copular verbs need object check
    }
    if pred in strongly_single_valued:
        return _looks_like_single_valued_claim(old_obj, new_obj)

    # For additive predicates, NOT mutually exclusive
    if is_additive_predicate(pred):
        return False

    # For unknown predicates, use heuristic
    return _looks_like_single_valued_claim(old_obj, new_obj)


def _detect_rule_based(
    proposal: Dict[str, Any],
    latest: Dict[str, Any],
    stale_info: Optional[Dict[str, Any]],
    all_candidates: List[Dict[str, Any]] = None
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Comprehensive rule-based conflict detection.

    Returns: (conflict_type, details) or (None, {}) if need semantic tier
    """
    predicate = proposal.get("predicate", "")
    old_obj = str(latest.get("object_val", ""))
    new_obj = str(proposal.get("object_val", ""))

    # 1. Exact duplicate check
    if old_obj == new_obj:
        return "exact_duplicate", {"reason": "identical_object_val"}

    # 2. Stale read conflict
    if stale_info and stale_info.get("stale"):
        return "stale_read_conflict", {
            "reason": "read_before_latest_commit",
            "stale_info": stale_info
        }

    # 3. Read-your-writes consistency check
    # If proposal agent_id matches latest agent_id and timestamps are close,
    # this should be an overwrite, not a conflict
    prop_agent = proposal.get("agent_id")
    latest_agent = latest.get("agent_id")
    if prop_agent == latest_agent:
        prop_ts = proposal.get("timestamp", 0)
        latest_ts = latest.get("timestamp", 0)
        if prop_ts >= latest_ts and abs(prop_ts - latest_ts) < 10.0:
            # Same agent, newer timestamp - should overwrite, not conflict
            return None, {"reason": "read_your_writes_sequential"}

    # 4. Concurrent update detection
    if all_candidates and _is_concurrent_update(proposal, all_candidates):
        # Concurrent updates by different agents → mutually exclusive
        return "mutually_exclusive", {
            "reason": "concurrent_writes_different_agents",
            "candidate_count": len(all_candidates)
        }

    # 5. Counterfactual temporal conflict
    # Newer information contradicts older (out-of-order or revision)
    if _is_temporal_inconsistency(proposal, all_candidates or [latest]):
        return "counterfactual_temporal", {
            "reason": "newer_contradicts_older_temporal",
            "timestamps": {
                "proposal": proposal.get("timestamp"),
                "latest": latest.get("timestamp")
            }
        }

    # 6. Mutually exclusive predicate check - SKIP for additive predicates
    # Additive predicates should be handled by semantic tier to detect overlap
    if not is_additive_predicate(predicate) and is_likely_mutually_exclusive(predicate, old_obj, new_obj):
        return "mutually_exclusive", {
            "reason": "predicate_is_mutually_exclusive_any_difference_conflicts",
            "predicate_type": predicate,
            "value_comparison": {"old": old_obj[:50], "new": new_obj[:50]}
        }

    # 7. Additive predicate - defer to semantic to check for overlap
    if is_additive_predicate(predicate):
        return None, {}  # Let semantic tier decide

    # No rule-based conflict found - needs semantic check
    return None, {}


def _detect_semantic(
    proposal: Dict[str, Any],
    latest: Dict[str, Any],
    mode: str = "debug_fallback"
) -> Tuple[str, Dict[str, Any]]:
    """
    Semantic similarity-based conflict detection.
    Handles cases where rule-based detection is inconclusive.
    """
    old_obj = str(latest.get("object_val", ""))
    new_obj = str(proposal.get("object_val", ""))
    predicate = proposal.get("predicate", "")

    # Calculate similarity
    sim_score = semantic_similarity(old_obj, new_obj, mode=mode)

    # Check JSON mergeability first
    try:
        old_json = json.loads(old_obj) if old_obj.strip().startswith("{") else None
        new_json = json.loads(new_obj) if new_obj.strip().startswith("{") else None
        if isinstance(old_json, dict) and isinstance(new_json, dict):
            # Even low similarity but both JSON objects can be merged
            return "semantic_overlap", {
                "similarity": sim_score,
                "reason": "json_objects_mergeable",
                "json_mergeable": True
            }
    except (json.JSONDecodeError, TypeError):
        pass

    # For additive predicates, always treat as semantic_overlap (merge)
    if is_additive_predicate(predicate):
        return "semantic_overlap", {
            "similarity": sim_score,
            "reason": "additive_predicate_always_merge"
        }

    # For mutually exclusive predicates, semantic similarity shouldn't force merge
    if is_likely_mutually_exclusive(predicate, old_obj, new_obj):
        # Even if similar, they're contradictory
        return "mutually_exclusive", {
            "similarity": sim_score,
            "reason": "mutually_exclusive_predicate_different_values"
        }

    # Non-additive, non-mutually exclusive predicates:
    # Use similarity thresholds
    if sim_score >= DUPLICATE_THRESHOLD:
        return "semantic_duplicate", {
            "similarity": sim_score,
            "reason": "near_identical_semantic"
        }
    elif sim_score >= SEMANTIC_OVERLAP_THRESHOLD:
        return "semantic_overlap", {
            "similarity": sim_score,
            "reason": "significant_semantic_overlap"
        }
    elif sim_score >= COMPATIBLE_THRESHOLD:
        return "compatible_extension", {
            "similarity": sim_score,
            "reason": "moderate_semantic_overlap_compatible"
        }
    else:  # sim_score < COMPATIBLE_THRESHOLD
        return "potential_contradiction", {
            "similarity": sim_score,
            "reason": "low_similarity_potential_conflict"
        }


def detect_conflict_type(
    proposal: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    read_snapshot_time: float,
    stale_detector: Any,
    mode: str = "debug_fallback"
) -> Tuple[str, Dict[str, Any]]:
    """
    Comprehensive conflict detection with rule/semantic tiers.

    Args:
        proposal: Proposed memory entry
        candidates: Existing entries with same (subject, predicate)
        read_snapshot_time: When agent read before proposing
        stale_detector: StalenessDetector instance
        mode: "research_strict" or "debug_fallback"

    Returns:
        (conflict_type, details) - always a valid conflict type
    """
    if not candidates:
        return "none", {"reason": "no_existing_candidates"}

    latest = candidates[-1]

    # Tier 1: Rule-based detection with staleness check
    stale_info = stale_detector.detect_stale_for_proposal(
        proposal, read_snapshot_time, candidates
    )
    rule_result, rule_details = _detect_rule_based(proposal, latest, stale_info, candidates)
    if rule_result is not None:
        return rule_result, rule_details

    # Tier 2: Semantic similarity
    return _detect_semantic(proposal, latest, mode=mode)
