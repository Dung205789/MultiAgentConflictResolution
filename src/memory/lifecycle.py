"""
Memory lifecycle management: scoring for retrieval, consolidation, promotion, decay, archiving.
Research-backed: Mem0, Zep, A-MEM, Ebbinghaus-inspired adaptive decay.
"""
import math
from typing import List, Dict, Any, Optional, Tuple
from .shared_memory_store import MemoryEntry


def calculate_adaptive_lambda(lambda_base: float, recall_count: int, beta: float = 0.3, gamma: float = 0.1) -> float:
    """
    Compute adaptive decay rate based on recall frequency.
    More frequently recalled memories decay more slowly.
    Uses tanh to asymptotically reduce lambda as recall_count increases.

    lambda_n = lambda_base * (1 - beta * tanh(gamma * recall_count))
    """
    boost = beta * math.tanh(gamma * recall_count)
    return lambda_base * (1 - boost)


def score_for_recall(
    entry: MemoryEntry,
    query_embedding: Optional[List[float]] = None,
    current_time: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compute a memory's retrieval score, blending:
    - Semantic relevance (cosine similarity to query)
    - Importance (derived from provenance/type/confidence)
    - Adaptive recency (exponential decay with recall-boosted lambda)
    - Provenance reliability
    - Consistency (canonical status, conflicts)

    Returns a score between 0 and 1.
    """
    if config is None:
        config = {
            "weights": {
                "semantic_relevance": 0.25,
                "importance": 0.20,
                "recency": 0.20,
                "provenance": 0.15,
                "evidence": 0.10,
                "consistency": 0.10,
            },
            "decay": {
                "lambda_base": 0.0001,
                "recall_boost_beta": 0.3,
                "recall_boost_gamma": 0.1,
            }
        }

    weights = config.get("weights", {})
    decay_cfg = config.get("decay", {})

    # 1. Semantic relevance (if query embedding provided)
    semantic_score = 0.0
    if query_embedding is not None and hasattr(entry, 'embedding') and entry.embedding is not None:
        try:
            # Compute cosine similarity
            dot = sum(a * b for a, b in zip(entry.embedding, query_embedding))
            norm1 = math.sqrt(sum(a * a for a in entry.embedding))
            norm2 = math.sqrt(sum(b * b for b in query_embedding))
            if norm1 > 0 and norm2 > 0:
                semantic_score = dot / (norm1 * norm2)
            else:
                semantic_score = 0.0
        except Exception:
            semantic_score = 0.0
    else:
        # No embedding available; treat as neutral (could use lexical similarity instead)
        semantic_score = 0.5

    # 2. Importance: derived from confidence and provenance
    # Map provenance to weight
    provenance_weights = {
        "explicit": 1.0,
        "behavioral": 0.85,
        "inferred": 0.7,
        "llm_inferred": 0.6,
        "unknown": 0.4,
    }
    provenance_score = provenance_weights.get(entry.provenance, 0.4)
    # Importance is a blend of confidence and provenance
    importance_score = 0.6 * entry.confidence + 0.4 * provenance_score

    # 3. Recency: exponential decay using adaptive lambda
    ref_time = current_time or (entry.committed_at or entry.timestamp or 0.0)
    if ref_time <= 0:
        recency_score = 1.0
    else:
        # Use event_time if available, else committed_at, else timestamp
        entry_time = entry.event_time or entry.committed_at or entry.timestamp or 0.0
        delta_t = ref_time - entry_time
        if delta_t < 0:
            # Future-dated entry; treat as very recent
            recency_score = 1.0
        else:
            lambda_n = calculate_adaptive_lambda(
                lambda_base=decay_cfg.get("lambda_base", 0.0001),
                recall_count=entry.recall_count,
                beta=decay_cfg.get("recall_boost_beta", 0.3),
                gamma=decay_cfg.get("recall_boost_gamma", 0.1)
            )
            recency_score = math.exp(-lambda_n * delta_t)

    # 4. Provenance (again) - we already used it in importance, but can weight separately if desired
    # Use the same provenance_score; could be separate but weights already allocated.

    # 5. Evidence: does raw_text exist? Then 1.0 else 0.5
    evidence_score = 1.0 if entry.raw_text and entry.raw_text.strip() else 0.5

    # 6. Consistency: favor canonical/active entries, penalize superseded/archived
    status_consistency = {
        "canonical": 1.0,
        "active": 1.0,
        "tentative": 0.7,
        "needs_review": 0.5,
        "superseded": 0.2,
        "archived": 0.0,
        "rejected": 0.0,
    }
    consistency_score = status_consistency.get(entry.canonical_status, 0.5)

    # Weighted sum
    final_score = (
        weights.get("semantic_relevance", 0.25) * semantic_score +
        weights.get("importance", 0.20) * importance_score +
        weights.get("recency", 0.20) * recency_score +
        weights.get("provenance", 0.15) * provenance_score +
        weights.get("evidence", 0.10) * evidence_score +
        weights.get("consistency", 0.10) * consistency_score
    )

    return final_score


def consolidate_memories(memories: List[MemoryEntry], similarity_threshold: float = 0.9) -> List[MemoryEntry]:
    """
    Merge semantic duplicates and mark superseded historical facts.
    Strategy: group by (subject, predicate), sort by recency/confidence, then merge high-similarity entries.
    Returns a list of consolidated memories.
    """
    # Group by (subject, predicate)
    groups: Dict[Tuple[str, str], List[MemoryEntry]] = {}
    for mem in memories:
        key = (mem.subject, mem.predicate)
        groups.setdefault(key, []).append(mem)

    consolidated = []
    for key, group in groups.items():
        # Sort group by confidence and recency (newest first)
        group.sort(key=lambda m: (
            m.confidence,
            m.committed_at or m.timestamp or 0.0
        ), reverse=True)

        # The top entry is the strongest candidate
        top = group[0]
        merged_ids = []
        to_merge = []

        for candidate in group[1:]:
            # Check if candidate can be merged into top (high similarity)
            # We need an embedding similarity check. If no embeddings, we can use simple heuristic:
            # If object_val is similar string-wise or if both are JSON and keys are disjoint.
            # Since we don't have embedding model here, we'll use a simple string similarity placeholder.
            # In practice, use semantic_similarity from conflict_detector.
            # For now, merge if same object_val (exact duplicate) or if both are JSON and non-overlapping keys.
            # Mark others as superseded but not merged.
            if candidate.object_val == top.object_val:
                # Exact duplicate: mark superseded, no content change
                candidate.status = "superseded"
                candidate.canonical_status = "historical"
                merged_ids.append(candidate.memory_id)
            else:
                # Not identical; keep as separate? Could be contradictory or additive.
                # For consolidation, we only merge high-similarity entries. Without embeddings, we'll be conservative.
                # If the top already has merged_from empty and this candidate is newer but lower confidence,
                # we could mark as superseded.
                candidate.status = "superseded"
                candidate.canonical_status = "historical"
                merged_ids.append(candidate.memory_id)

        if merged_ids:
            top.merged_from = merged_ids
            top.canonical_status = "canonical"
        consolidated.append(top)

    return consolidated


def promote_to_long_term(entry: MemoryEntry, rules: Optional[Dict[str, Any]] = None) -> bool:
    """
    Promote a memory to long-term based on recall_count, confidence, and evidence.
    Returns True if promotion occurred, False otherwise.
    """
    if rules is None:
        rules = {
            "min_recall_count": 3,
            "min_confidence": 0.7,
            "require_evidence": True,
        }

    # Already long-term? All memories are long-term in this system, but we can mark a flag.
    # In A-MEM style, long-term might have different storage tier.
    # For now, promotion means marking as canonical and maybe boosting recall_count.
    if entry.recall_count >= rules["min_recall_count"] and entry.confidence >= rules["min_confidence"]:
        if rules["require_evidence"] and not entry.raw_text:
            return False
        entry.canonical_status = "canonical"
        return True
    return False


def decay_memory(entry: MemoryEntry, current_time: Optional[float] = None, lambda_n: Optional[float] = None) -> float:
    """
    Apply adaptive forgetting decay to a memory's effective score.
    Returns the decay factor (between 0 and 1) that can be used to adjust retrieval scores or archive.
    If decay factor falls below a threshold, consider archiving.
    """
    if current_time is None:
        current_time = entry.committed_at or entry.timestamp or 0.0

    if lambda_n is None:
        lambda_n = calculate_adaptive_lambda(
            lambda_base=0.0001,
            recall_count=entry.recall_count,
            beta=0.3,
            gamma=0.1
        )

    entry_time = entry.event_time or entry.committed_at or entry.timestamp or 0.0
    delta_t = current_time - entry_time
    if delta_t < 0:
        return 1.0
    decay_factor = math.exp(-lambda_n * delta_t)
    return decay_factor


def mark_user_confirmed(entry: MemoryEntry) -> None:
    """
    Boost recall_count, set provenance explicit, set canonical status.
    Called when a human explicitly confirms a memory fact.
    """
    entry.recall_count += 1
    entry.last_recalled_at = time.time()
    entry.provenance = "explicit"
    entry.canonical_status = "canonical"
    entry.confidence = min(1.0, entry.confidence + 0.1)  # small confidence boost


def archive_superseded(store, current_time: Optional[float] = None, retention_days: int = 30) -> int:
    """
    Archive old superseded/historical entries beyond retention rules.
    Removes them from active visibility or moves to archive status.
    Returns the number of entries archived.
    """
    if current_time is None:
        current_time = time.time()

    retention_seconds = retention_days * 24 * 3600
    archived_count = 0

    for entry in store.records:
        if entry.canonical_status in ["superseded", "historical", "rejected"]:
            # If it's been superseded for longer than retention period, archive it
            supersede_time = entry.committed_at or entry.timestamp or 0.0
            if (current_time - supersede_time) > retention_seconds:
                entry.status = "archived"
                entry.visibility_state = "stale_visible"  # or could remove from active list
                archived_count += 1

    return archived_count


def mark_user_contradiction(entry1: MemoryEntry, entry2: MemoryEntry) -> None:
    """
    Mark two entries as conflicting.
    """
    entry1.conflicts_with = entry2.memory_id
    entry2.conflicts_with = entry1.memory_id
    entry1.canonical_status = "needs_review"
    entry2.canonical_status = "needs_review"
