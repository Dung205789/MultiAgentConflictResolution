"""
Enhanced retriever with conflict-awareness and metadata-based ranking.
"""
from typing import Dict, List, Tuple, Optional, Any
import math
import time
import os
import yaml

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from src.memory.lifecycle import score_for_recall as lifecycle_score

try:
    from src.memory.shared_memory_store import MemoryEntry
    HAVE_MEMORY_ENTRY = True
except ImportError:
    HAVE_MEMORY_ENTRY = False


_EMBED_MODEL = None

# Load arbitration config for lifecycle weights
def _load_arbitration_config() -> Dict[str, Any]:
    """Load arbitration configuration from YAML."""
    config_path = "configs/arbitration.yaml"
    if not os.path.isabs(config_path):
        # Assume relative to project root (two levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(project_root, config_path)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            return config
    except Exception:
        # Fallback to defaults matching arbitration.yaml
        return {
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
            },
            "provenance_weights": {
                "explicit": 1.0,
                "behavioral": 0.85,
                "inferred": 0.7,
                "llm_inferred": 0.6,
                "unknown": 0.4,
            }
        }

_ARBITRATION_CONFIG = _load_arbitration_config()


def _safe_text(x: str) -> str:
    """Convert input to safe, normalized text."""
    return (x or "").strip().lower()


def tokenize(text: str) -> List[str]:
    """Simple tokenization for text."""
    return [w for w in _safe_text(text).replace(".", " ").replace(",", " ").split() if w]


def overlap_score(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa or not sb:
        return 0.0
    inter = len(sa.intersection(sb))
    union = len(sa.union(sb))
    return inter / union if union else 0.0


def _load_embed_model():
    """Load the embedding model if available."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    if SentenceTransformer is None:
        return None
    try:
        # Force CPU-only to avoid CUDA dependency issues
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return _EMBED_MODEL
    except Exception:
        return None


def _build_memory_text(m: Dict) -> str:
    """Build a text representation of a memory entry."""
    return f"{m.get('subject','user')} {m.get('predicate','')} {m.get('object_val','')} {m.get('memory_type','')}".strip()


def _dict_to_memory_entry(mem_dict: Dict[str, Any]) -> MemoryEntry:
    """Convert a dictionary to a MemoryEntry object for lifecycle scoring."""
    entry = MemoryEntry(
        subject=mem_dict.get("subject", ""),
        predicate=mem_dict.get("predicate", ""),
        object_val=mem_dict.get("object_val", ""),
        agent_id=mem_dict.get("agent_id", "unknown"),
        confidence=float(mem_dict.get("confidence", 0.5)),
        provenance=mem_dict.get("provenance", "inferred"),
        raw_text=mem_dict.get("raw_text", ""),
        canonical_claim=mem_dict.get("canonical_claim"),
        memory_type=mem_dict.get("memory_type", "fact"),
        event_time=mem_dict.get("event_time"),
        ingestion_time=mem_dict.get("ingestion_time"),
        valid_from=mem_dict.get("valid_from"),
        valid_until=mem_dict.get("valid_until"),
        session_id=mem_dict.get("session_id"),
        turn_index=mem_dict.get("turn_index"),
        recall_count=int(mem_dict.get("recall_count", 0)),
        last_recalled_at=mem_dict.get("last_recalled_at"),
        supersedes=mem_dict.get("supersedes"),
        merged_from=mem_dict.get("merged_from", []),
        conflicts_with=mem_dict.get("conflicts_with"),
        canonical_status=mem_dict.get("canonical_status", "tentative")
    )
    # Set timestamps
    entry.timestamp = mem_dict.get("timestamp", entry.timestamp)
    entry.committed_at = mem_dict.get("committed_at", entry.committed_at)
    entry.indexed_at = mem_dict.get("indexed_at", entry.indexed_at)
    # Optional embedding - skip if not present
    if "embedding" in mem_dict:
        entry.embedding = mem_dict["embedding"]
    return entry


def _calculate_metadata_score(memory: Dict[str, Any]) -> float:
    """Calculate a quality score based on memory metadata."""
    # Base score starts at 0.5
    score = 0.5
    
    # Confidence boost (0 to 0.3)
    confidence = float(memory.get("confidence", 0.5))
    score += 0.3 * confidence
    
    # Provenance quality boost (0 to 0.2)
    provenance = memory.get("provenance", "unknown")
    provenance_scores = {
        "explicit": 0.2,
        "behavioral": 0.15,
        "inferred": 0.1,
        "llm_inferred": 0.05,
        "unknown": 0.0
    }
    score += provenance_scores.get(provenance, 0.0)
    
    # Status penalty for non-active or tentative entries
    status = memory.get("status", "active")
    if status != "active":
        score *= 0.7  # 30% penalty
    elif memory.get("visibility_state") != "visible":
        score *= 0.8  # 20% penalty
    
    # Cap at 1.0
    return min(1.0, score)


def retrieve_topk_keyword(memories: List[Dict], query: str, k: int = 3) -> List[Tuple[Dict, float]]:
    """Retrieve top-k memories using keyword matching."""
    scored = []
    for m in memories:
        # Skip non-active entries unless they have special metadata
        if m.get("status", "active") != "active" and not m.get("arbitration_metadata"):
            continue
            
        text = _build_memory_text(m)
        match_score = overlap_score(text, query)
        
        # Apply metadata-based adjustments
        metadata_score = _calculate_metadata_score(m)
        final_score = 0.7 * match_score + 0.3 * metadata_score
        
        scored.append((m, final_score))
        
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def retrieve_topk_embedding(memories: List[Dict], query: str, k: int = 3) -> List[Tuple[Dict, float]]:
    """Retrieve top-k memories using embedding similarity."""
    model = _load_embed_model()
    if model is None:
        return retrieve_topk_keyword(memories, query, k=k)

    # Filter for active memories or those with special metadata
    active_mem = [m for m in memories if m.get("status", "active") == "active" or m.get("arbitration_metadata")]
    if not active_mem:
        return []

    mem_texts = [_build_memory_text(m) for m in active_mem]
    try:
        q_emb = model.encode([query], normalize_embeddings=True)
        m_emb = model.encode(mem_texts, normalize_embeddings=True)
    except Exception:
        return retrieve_topk_keyword(memories, query, k=k)

    import numpy as np
    sims = (m_emb @ q_emb[0]).tolist()
    
    # Apply metadata-based adjustments
    adjusted_scores = []
    for i, (mem, sim) in enumerate(zip(active_mem, sims)):
        metadata_score = _calculate_metadata_score(mem)
        final_score = 0.7 * sim + 0.3 * metadata_score
        adjusted_scores.append((mem, final_score))
    
    adjusted_scores.sort(key=lambda x: x[1], reverse=True)
    return adjusted_scores[:k]


def retrieve_topk(
    memories: List[Dict], 
    query: str, 
    k: int = 3, 
    method: str = "hybrid",
    mode: str = "debug_fallback",
    include_conflicts: bool = True
) -> List[Tuple[Dict, float]]:
    """
    Enhanced retrieval with conflict awareness.
    
    Args:
        memories: List of memory entries
        query: Search query
        k: Number of results to return
        method: "keyword", "embedding", or "hybrid"
        mode: "research_strict" or "debug_fallback"
        include_conflicts: Whether to include conflicting entries
        
    Returns:
        List of (memory, score) tuples
    """
    if method == "keyword":
        results = retrieve_topk_keyword(memories, query, k=k)
        
    elif method == "embedding" or method == "hybrid":
        if mode == "research_strict":
            model = _load_embed_model()
            if model is None:
                raise RuntimeError(
                    f"Embedding retrieval requested ({method}) in research_strict mode but no model available. "
                    "Install sentence-transformers and the all-MiniLM-L6-v2 model."
                )
        
        if method == "embedding":
            results = retrieve_topk_embedding(memories, query, k=k)
        else:
            # Hybrid method
            emb = retrieve_topk_embedding(memories, query, k=max(k * 2, 6))
            kw = retrieve_topk_keyword(memories, query, k=max(k * 2, 6))

            merged = {}
            for m, s in emb:
                merged[m.get("memory_id")] = (m, 0.7 * float(s))
            for m, s in kw:
                if m.get("memory_id") in merged:
                    old_m, old_s = merged[m.get("memory_id")]
                    merged[m.get("memory_id")] = (old_m, old_s + 0.3 * float(s))
                else:
                    merged[m.get("memory_id")] = (m, 0.3 * float(s))

            results = list(merged.values())
            results.sort(key=lambda x: x[1], reverse=True)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")

    # Apply lifecycle reranking to boost high-importance, well-maintained memories
    if HAVE_MEMORY_ENTRY:
        try:
            # Prepare lifecycle config from arbitration config
            life_config = {
                "weights": {
                    "semantic_relevance": 0.0,  # already accounted in base score
                    "importance": _ARBITRATION_CONFIG.get("weights", {}).get("importance", 0.20),
                    "recency": _ARBITRATION_CONFIG.get("weights", {}).get("recency", 0.20),
                    "provenance": _ARBITRATION_CONFIG.get("weights", {}).get("provenance", 0.15),
                    "evidence": _ARBITRATION_CONFIG.get("weights", {}).get("evidence", 0.10),
                    "consistency": _ARBITRATION_CONFIG.get("weights", {}).get("consistency", 0.10),
                },
                "decay": _ARBITRATION_CONFIG.get("decay", {
                    "lambda_base": 0.0001,
                    "recall_boost_beta": 0.3,
                    "recall_boost_gamma": 0.1,
                }),
                "provenance_weights": _ARBITRATION_CONFIG.get("provenance_weights", {
                    "explicit": 1.0,
                    "behavioral": 0.85,
                    "inferred": 0.7,
                    "llm_inferred": 0.6,
                    "unknown": 0.4,
                })
            }

            reranked = []
            for mem, score in results:
                # Convert dict to MemoryEntry for scoring
                try:
                    entry = _dict_to_memory_entry(mem)
                except Exception:
                    # If conversion fails, keep original score
                    reranked.append((mem, score))
                    continue

                # Use lifecycle scoring to adjust score based on importance/recency/provenance
                life_boost = lifecycle_score(entry, current_time=time.time(), config=life_config)
                final_score = 0.6 * score + 0.4 * life_boost
                reranked.append((mem, final_score))
            results = reranked
            results.sort(key=lambda x: x[1], reverse=True)
        except Exception:
            # If lifecycle scoring fails, keep original scores
            pass

    # Handle conflicts - group by entity_id and ensure diversity
    if include_conflicts:
        # Group by entity_id
        entity_groups = {}
        for mem, score in results:
            entity_id = f"{mem.get('subject')}_{mem.get('predicate')}"
            if entity_id not in entity_groups:
                entity_groups[entity_id] = []
            entity_groups[entity_id].append((mem, score))
        
        # For each entity with multiple versions, ensure we include diverse perspectives
        final_results = []
        for entity_id, group in entity_groups.items():
            if len(group) == 1:
                final_results.append(group[0])
            else:
                # Sort by score
                group.sort(key=lambda x: x[1], reverse=True)
                
                # Always include the highest scoring one
                final_results.append(group[0])
                
                # If there are explicit conflicts, include the top alternative
                has_conflicts = any(
                    m.get("arbitration_metadata", {}).get("branch_mode") == "parallel_active_versions" 
                    for m, _ in group
                )
                
                if has_conflicts and len(group) > 1:
                    final_results.append(group[1])
        
        # Re-sort and limit
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]
    
    return results[:k]
