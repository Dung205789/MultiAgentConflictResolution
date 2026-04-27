"""
LongMemEval adapter for Hugging Face dataset.

Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Paper/Repo: https://github.com/xiaowu0162/LongMemEval

LongMemEval evaluates long-term memory in conversational agents with
multi-session dialogues and memory retrieval tasks.
"""
import json
from typing import Dict, List, Any, Optional

from src.benchmarks.generator_core import generate_scenario


def load_longmemeval(subset: str = "all", num_samples: int = None) -> List[Dict[str, Any]]:
    """
    Load LongMemEval from Hugging Face.

    Args:
        subset: Which subset to load ("all", "s", "m", "oracle", etc.)
        num_samples: Maximum number of samples to load (None for all)

    Returns:
        List of scenarios in the repository's benchmark format.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found.")
        print("Please install: pip install datasets")
        return []

    # Determine which dataset split to load
    dataset_name = "xiaowu0162/longmemeval-cleaned"
    if subset == "s":
        split = "longmemeval_s_cleaned"
    elif subset == "m":
        split = "longmemeval_m_cleaned"
    elif subset == "oracle":
        split = "longmemeval_oracle"
    else:
        split = "longmemeval_s_cleaned"  # default to a valid split

    try:
        # Use streaming mode to avoid loading full dataset into memory
        ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"Error loading LongMemEval with split '{split}': {e}")
        print("Attempting to load s_cleaned split...")
        try:
            ds = load_dataset(dataset_name, split="longmemeval_s_cleaned", streaming=True)
        except Exception as e2:
            print(f"Failed to load LongMemEval: {e2}")
            print("Make sure you have internet connection and datasets library is installed.")
            return []

    scenarios = []
    for idx, item in enumerate(ds):
        scenario = _convert_longmemeval_item(item, idx, subset)
        if scenario:
            scenarios.append(scenario)
        # Stop if we reached the sample limit
        if num_samples is not None and len(scenarios) >= num_samples:
            break

    print(f"Loaded {len(scenarios)} LongMemEval scenarios (subset={subset})")
    return scenarios


def _convert_longmemeval_item(item: Dict[str, Any], idx: int, subset_tag: str) -> Optional[Dict[str, Any]]:
    """
    Convert a LongMemEval item to the repository's scenario format.

    Expected LongMemEval format (based on repo):
    - dialogues: list of dialogue turns with speaker, text, session info
    - memories: ground truth memory facts that should be retained
    - queries: retrieval questions to test memory
    - profile: user/agent profile information

    The dataset tests memory retention across long conversations.
    """
    # Extract basic info
    dialogue_id = item.get("dialogue_id", f"longmemeval_{subset_tag}_{idx}")
    dialogues = item.get("dialogues", [])
    memories = item.get("memories", [])
    queries = item.get("queries", [])

    # Build ordered events from dialogue turns
    ordered_events = []
    current_agent = None

    for turn in dialogues:
        speaker = turn.get("speaker", "agent")
        text = turn.get("text", "")
        turn_type = turn.get("type", "utterance")

        # Determine agent_id (alternate between two agents or use speaker mapping)
        if speaker.lower() in ["user", "human"]:
            agent_id = "agent_a"
        else:
            agent_id = "agent_b"

        # Convert dialogue turns into write proposals or reads
        if turn_type == "utterance" and text:
            # Extract memory facts from the utterance
            # In LongMemEval, some utterances contain memory-worthy information
            extracted_memories = _extract_memories_from_text(text, speaker)

            for mem in extracted_memories:
                proposal = {
                    "subject": mem.get("subject", "user"),
                    "predicate": mem.get("predicate", "info"),
                    "object_val": mem.get("object", text),
                    "confidence": mem.get("confidence", 0.8),
                    "provenance": mem.get("provenance", "explicit")
                }
                ordered_events.append({
                    "step": len(ordered_events) + 1,
                    "agent_id": agent_id,
                    "event_type": "write_proposal",
                    "proposal": proposal,
                    "timestamp": turn.get("timestamp", len(ordered_events) * 60.0)
                })

        elif turn_type == "query":
            # Memory retrieval query
            ordered_events.append({
                "step": len(ordered_events) + 1,
                "agent_id": agent_id,
                "event_type": "read",
                "query": text,
                "timestamp": turn.get("timestamp", len(ordered_events) * 60.0)
            })

    # If no events extracted, skip this item
    if not ordered_events:
        return None

    # Build queries for evaluation from the queries field
    eval_queries = []
    for q in queries:
        query_text = q.get("query", "")
        gold_answers = q.get("answers", [])
        if query_text:
            eval_queries.append({
                "query_text": query_text,
                "gold_answers": gold_answers if isinstance(gold_answers, list) else [gold_answers],
                "expected_retrieval_style": "best"
            })

    # Build gold memory state from memories field
    gold_memory_state = []
    for mem in memories:
        gold_memory_state.append({
            "subject": mem.get("subject", "user"),
            "predicate": mem.get("predicate", "info"),
            "object_val": mem.get("object", ""),
            "status": "active"
        })

    # Determine if there are conflicts
    # LongMemEval focuses on memory retention and retrieval, not necessarily conflicts
    # But we can treat multiple writes to same (subject, predicate) as potential conflicts
    conflict_exists = _detect_memory_conflicts(ordered_events)

    # Create scenario
    scenario = generate_scenario(
        scenario_id=dialogue_id,
        scenario_type="longmem_eval_memory_retention",
        agents=["agent_a", "agent_b"],
        ordered_events=ordered_events,
        gold_conflict_exists=conflict_exists,
        gold_conflict_type="memory_update" if conflict_exists else "none",
        gold_resolution_action="merge" if conflict_exists else "append",
        gold_reconciled_memory_state=gold_memory_state,
        gold_visible_shared_state_after_commit=gold_memory_state,
        description=f"LongMemEval dialogue: {len(dialogues)} turns, {len(memories)} memory facts",
        agent_profiles=None,
        queries=eval_queries,
        base_timestamp=1000.0
    )

    return scenario


def _extract_memories_from_text(text: str, speaker: str) -> List[Dict[str, Any]]:
    """
    Extract memory-worthy facts from dialogue text.

    LongMemEval may have structured memory annotations, but we also need
    to handle plain text. Simple heuristic extraction.
    """
    memories = []

    # Simple pattern matching for common memory predicates
    # This should be enhanced based on actual dataset format
    text_lower = text.lower()

    # Check for common memory-indicating phrases
    memory_indicators = ["i am", "i have", "my name is", "i live", "i work", "i like", "i hate",
                         "i want", "i need", "i remember", "i forgot", "i learned"]

    if any(indicator in text_lower for indicator in memory_indicators):
        # Extract as a general memory fact
        memories.append({
            "subject": "user" if "i" in text_lower[:20] else speaker,
            "predicate": "info",
            "object": text.strip(),
            "confidence": 0.8,
            "provenance": "explicit"
        })

    return memories


def _detect_memory_conflicts(ordered_events: List[Dict[str, Any]]) -> bool:
    """Check if there are conflicting write proposals for same (subject, predicate)."""
    writes = [ev for ev in ordered_events if ev.get("event_type") == "write_proposal"]
    seen = {}
    for w in writes:
        prop = w.get("proposal", {})
        key = (prop.get("subject"), prop.get("predicate"))
        if key in seen:
            return True
        seen[key] = w
    return False


if __name__ == "__main__":
    # Test loading
    print("Testing LongMemEval loader...")
    for subset in ["s", "m", "oracle", "all"]:
        print(f"\nSubset: {subset}")
        scenarios = load_longmemeval(subset=subset)
        if scenarios:
            print(json.dumps(scenarios[0], indent=2, ensure_ascii=False))
            break
