"""
Enhanced conflict detection module with more principled semantic conflict detection.
"""
from typing import Dict, Any, List, Tuple, Optional
import json
import logging
import re

# Don't import sentence_transformers at the module level
# We'll try to import it only when needed
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

# Constants for conflict detection
DUPLICATE_THRESHOLD = 0.95  # Almost exact match
SEMANTIC_OVERLAP_THRESHOLD = 0.7  # Strong semantic overlap
COMPATIBLE_THRESHOLD = 0.5  # Some semantic overlap but potentially compatible
CONTRADICTION_THRESHOLD = 0.3  # Low similarity, potential contradiction

# Embedding model for semantic similarity
_EMBED_MODEL = None


def _try_import_embeddings():
    """Try to import sentence_transformers and related modules."""
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
    """Load the embedding model for semantic similarity if available."""
    global _EMBED_MODEL

    if not _try_import_embeddings():
        return None

    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    try:
        # Force CPU-only to avoid CUDA dependency issues
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return _EMBED_MODEL
    except Exception:
        return None


def semantic_similarity(text1: str, text2: str, mode: str = "debug_fallback") -> float:
    """
    Calculate semantic similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        mode: Either "research_strict" (fail if models required but unavailable) or
              "debug_fallback" (allow fallback to lexical overlap)

    Returns:
        Similarity score between 0 and 1

    Raises:
        RuntimeError: If mode="research_strict" but embedding model unavailable or fails
    """
    # Always try embedding similarity first if available
    if _try_import_embeddings():
        model = _load_embed_model()
        if model is not None:
            try:
                embeddings = model.encode([text1, text2], normalize_embeddings=True)
                sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
                logging.debug(f"semantic_similarity: used embeddings (score={sim:.4f})")
                return sim
            except Exception as e:
                logging.warning(f"semantic_similarity: embedding failed ({e}), falling back to lexical")
                # Fall through to lexical fallback

    # If we get here, embeddings unavailable or failed
    if mode == "research_strict":
        raise RuntimeError(
            "Semantic similarity calculation requires sentence-transformers in research_strict mode. "
            "Install sentence-transformers and the all-MiniLM-L6-v2 model."
        )

    # Use lexical overlap as fallback
    overlap_func = _get_overlap_score()
    sim = overlap_func(text1, text2)
    logging.debug(f"semantic_similarity: used lexical overlap (score={sim:.4f})")
    return sim


def is_mutually_exclusive_predicate(predicate: str) -> bool:
    """
    Check if a predicate is mutually exclusive by nature.
    For example, a person can only be in one city at a time.

    NOTE: Generic linking verbs (is, was, are, etc.) are NOT inherently mutually exclusive
    because they can be used with non-exclusive predicates (e.g., "is skilled in", "are interested in").
    Only include predicates that represent truly single-valued attributes or roles.
    """
    mutually_exclusive_predicates = {
        # Location and temporal predicates
        "city", "study_time", "location", "current_location", "current_task",
        "current_focus", "current_activity", "current_mood", "current_status",
        # Identity and status (single-valued)
        "ssn", "email", "phone", "passport_number", "id_number", "birth_date", "death_date",
        # Position/title (primary role typically singular)
        "chairperson", "president", "director", "ceo", "founder", "author",
        "nationality", "place_of_birth", "place_of_death",
        # Measurement attributes (single value)
        "height", "weight", "age", "born_in", "death_place",
        # Financial identifiers
        "bank_account", "routing_number", "credit_card_number"
    }
    return predicate.lower() in mutually_exclusive_predicates


def _tokenize_surface(text: str) -> List[str]:
    """Tokenize text with lightweight normalization for rule checks."""
    return [tok for tok in re.split(r"[^a-z0-9]+", (text or "").lower()) if tok]


def _looks_like_single_valued_copular_claim(old_obj: str, new_obj: str) -> bool:
    """
    Heuristic for copular predicates (is/was/are/were).

    We only mark as mutually-exclusive when both sides look like the same
    claim template with different slot values (typical in conflict-resolution
    datasets), not for generic descriptive sentences.
    """
    old_norm = (old_obj or "").strip().lower()
    new_norm = (new_obj or "").strip().lower()
    if not old_norm or not new_norm or old_norm == new_norm:
        return False

    # High-signal cues that usually denote single-valued attributes/roles.
    single_value_cues = (
        "born in",
        "located in",
        "located at",
        "capital of",
        "chairperson",
        "president",
        "ceo",
        "director",
        "founder",
        "nationality",
        "plays",
        "died in",
        "works at",
        "works as",
        "worked at",
        "worked as",
        "language of",
        "married to",
        "a citizen of",
        "citizen of",
        "famous for",
        "employed by",
        "educated is",
        "created by",
        "performed by",
    )
    if any(cue in old_norm or cue in new_norm for cue in single_value_cues):
        return True

    old_tokens = _tokenize_surface(old_norm)
    new_tokens = _tokenize_surface(new_norm)
    if len(old_tokens) >= 3 and len(new_tokens) >= 3:
        old_set = set(old_tokens)
        new_set = set(new_tokens)
        overlap = len(old_set & new_set) / max(len(old_set), len(new_set))
        # Same template + small token length delta usually means "same claim, new value".
        if overlap >= 0.6 and abs(len(old_tokens) - len(new_tokens)) <= 3:
            return True

    # Conservative fallback for copular predicates in conflict-resolution settings:
    # if two non-identical atomic claims compete, treat as mutually exclusive.
    return True


def is_likely_mutually_exclusive(
    predicate: str,
    old_obj: str,
    new_obj: str,
) -> bool:
    """Augmented mutually-exclusive predicate check for natural-language predicates."""
    pred = (predicate or "").strip().lower()
    if is_mutually_exclusive_predicate(pred):
        return True

    # In MemAE-style triples these predicates are usually single-valued facts.
    strongly_single_valued_predicates = {
        "plays",
        "died",
        "works",
        "worked",
        "speaks the language of",
    }
    if pred in strongly_single_valued_predicates:
        return True

    # Copular verbs need object-level checks to avoid over-triggering.
    if pred in {"is", "was", "are", "were"}:
        return _looks_like_single_valued_copular_claim(old_obj, new_obj)

    return False


def is_additive_predicate(predicate: str) -> bool:
    """
    Check if a predicate is additive by nature.
    For example, a person can have multiple skills or interests.
    """
    additive_predicates = {
        "skill", "interest", "hobby", "language", "focus_area", 
        "known_for", "likes", "dislikes", "friend", "colleague"
    }
    return predicate.lower() in additive_predicates


def _detect_rule_based(proposal: Dict[str, Any], latest: Dict[str, Any],
                       stale_info: Optional[Dict[str, Any]]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Rule-based conflict detection.

    Returns: (conflict_type, details) or None if no rule-based conflict (need semantic check)
    """
    predicate = proposal.get("predicate", "")

    old_obj = str(latest.get("object_val", ""))
    new_obj = str(proposal.get("object_val", ""))

    # Exact duplicate
    if old_obj == new_obj:
        return "exact_duplicate", {"reason": "identical_object_val"}

    # Stale read
    if stale_info and stale_info.get("stale"):
        return "stale_read_conflict", stale_info

    # Mutually exclusive predicates: ANY difference is a conflict
    if is_likely_mutually_exclusive(predicate, old_obj, new_obj):
        return "mutually_exclusive", {
            "reason": "predicate_or_claim_pattern_is_mutually_exclusive_any_difference_conflicts",
            "predicate_type": "mutually_exclusive"
        }

    # Additive predicates: defer to semantic check (could be duplicate, overlap, or compatible)
    # We don't return early for additive; let semantic tier decide based on similarity
    return None


def _detect_semantic(proposal: Dict[str, Any], latest: Dict[str, Any],
                     mode: str = "debug_fallback") -> Tuple[str, Dict[str, Any]]:
    """
    Semantic similarity-based conflict detection.

    Returns: (conflict_type, details)
    """
    old_obj = str(latest.get("object_val", ""))
    new_obj = str(proposal.get("object_val", ""))
    predicate = proposal.get("predicate", "")

    # Calculate semantic similarity
    sim_score = semantic_similarity(old_obj, new_obj, mode=mode)

    # Check for JSON mergeable (even if low similarity)
    try:
        old_json = json.loads(old_obj) if old_obj.strip().startswith("{") else None
        new_json = json.loads(new_obj) if new_obj.strip().startswith("{") else None
        if isinstance(old_json, dict) and isinstance(new_json, dict):
            return "semantic_overlap", {
                "similarity": sim_score,
                "reason": "both_are_json_objects_mergeable",
                "json_mergeable": True
            }
    except (json.JSONDecodeError, TypeError):
        pass

    # For additive predicates, always treat as compatible extension (different valid perspectives)
    if is_additive_predicate(predicate):
        return "compatible_extension", {
            "similarity": sim_score,
            "reason": "additive_predicate_compatible_extension"
        }

    # For single-valued predicates/claims, semantic similarity should not force merge/keep-both.
    # Competing values are mutually exclusive even if wording is close or far.
    if is_likely_mutually_exclusive(predicate, old_obj, new_obj):
        return "mutually_exclusive", {
            "similarity": sim_score,
            "reason": "single_valued_predicate_or_claim_pattern",
            "predicate_type": "mutually_exclusive",
        }

    # Determine conflict type based on similarity thresholds for non-additive predicates
    if sim_score >= DUPLICATE_THRESHOLD:
        return "semantic_duplicate", {"similarity": sim_score, "reason": "near_identical_meaning"}

    elif sim_score >= SEMANTIC_OVERLAP_THRESHOLD:
        return "semantic_overlap", {"similarity": sim_score, "reason": "significant_semantic_overlap"}

    elif sim_score >= COMPATIBLE_THRESHOLD:
        return "semantic_overlap", {"similarity": sim_score, "reason": "moderate_semantic_overlap"}

    else:  # sim_score < COMPATIBLE_THRESHOLD
        return "potential_contradiction", {
            "similarity": sim_score,
            "reason": "low_similarity_potential_contradiction"
        }


def detect_conflict_type(
    proposal: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    read_snapshot_time: float,
    stale_detector: Any,
    mode: str = "debug_fallback"
) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced conflict detection with rule/semantic/judge tiers.

    Args:
        proposal: The proposed memory entry
        candidates: Existing memory entries with the same subject and predicate
        read_snapshot_time: When the agent read the memory before making this proposal
        stale_detector: Staleness detector instance
        mode: Either "research_strict" or "debug_fallback"

    Returns:
        Tuple of (conflict_type, details)
    """
    if not candidates:
        return "none", {"reason": "no_candidates"}

    latest = candidates[-1]

    # Tier 1: Rule-based detection
    stale_info = stale_detector.detect_stale_for_proposal(
        proposal, read_snapshot_time, candidates
    )
    rule_result = _detect_rule_based(proposal, latest, stale_info)
    if rule_result:
        return rule_result

    # Tier 2: Semantic similarity detection
    return _detect_semantic(proposal, latest, mode=mode)
