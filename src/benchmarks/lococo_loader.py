"""
LoCoMo adapter for the official dataset release.

Primary source:
- Repo: https://github.com/snap-research/locomo
- Dataset file: data/locomo10.json

The repository previously assumed a Hugging Face schema that does not match the
official release. This loader now prefers the official `locomo10.json` layout
when present locally and falls back to the older streaming path only if needed.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterable

from src.benchmarks.generator_core import generate_scenario


DEFAULT_LOCAL_PATH = Path("data/raw/locomo/locomo10.json")


def load_lococo(
    subset: str = "all",
    num_samples: int = None,
    dataset_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load LoCoMo from Hugging Face.

    Args:
        subset: Which subset to load ("all", "test", "validation", etc.)
        num_samples: Maximum number of samples to load (None for all)

    Returns:
        List of scenarios in the repository's benchmark format.
    """
    local_path = Path(dataset_path) if dataset_path else DEFAULT_LOCAL_PATH
    if local_path.exists():
        scenarios = _load_from_official_json(local_path, num_samples=num_samples)
        print(f"Loaded {len(scenarios)} LoCoMo scenarios from {local_path}")
        return scenarios

    scenarios = _load_from_legacy_hf(subset=subset, num_samples=num_samples)
    print(f"Loaded {len(scenarios)} LoCoMo scenarios")
    return scenarios


def _load_from_official_json(dataset_path: Path, num_samples: Optional[int]) -> List[Dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    scenarios = []
    for idx, item in enumerate(data):
        scenario = _convert_official_locomo_item(item, idx)
        if scenario:
            scenarios.append(scenario)
        if num_samples is not None and len(scenarios) >= num_samples:
            break
    return scenarios


def _load_from_legacy_hf(subset: str, num_samples: Optional[int]) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found.")
        print("Please install: pip install datasets")
        return []

    dataset_name = "Aman279/Locomo"

    try:
        if subset == "all":
            ds = [load_dataset(dataset_name, split="train", streaming=True)]
        else:
            ds = [load_dataset(dataset_name, split=subset, streaming=True)]
    except Exception as e:
        print(f"Error loading LoCoMo: {e}")
        print("Make sure you have internet connection and datasets library is installed.")
        return []

    scenarios = []
    for stream in ds:
        for idx, item in enumerate(stream):
            scenario = _convert_lococo_item(item, idx)
            if scenario:
                scenarios.append(scenario)
            if num_samples is not None and len(scenarios) >= num_samples:
                break
        if num_samples is not None and len(scenarios) >= num_samples:
            break

    return scenarios


def _convert_official_locomo_item(item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    sample_id = item.get("sample_id", f"locomo_{idx}")
    conversation = item.get("conversation", {})
    qa_items = item.get("qa", [])
    event_summary = item.get("event_summary", {})

    if not conversation:
        return None

    speaker_a = conversation.get("speaker_a", "speaker_a")
    speaker_b = conversation.get("speaker_b", "speaker_b")
    session_keys = sorted(
        [
            key for key in conversation.keys()
            if key.startswith("session_") and not key.endswith("_date_time")
        ],
        key=_session_sort_key,
    )
    if not session_keys:
        return None

    evidence_ids = {
        str(evidence_id)
        for qa in qa_items
        for evidence_id in qa.get("evidence", [])
        if evidence_id
    }

    ordered_events: List[Dict[str, Any]] = []
    step = 1
    base_timestamp = 1000.0
    time_step = 60.0

    for session_key in session_keys:
        session_date = conversation.get(f"{session_key}_date_time", "")
        ordered_events.append({
            "step": step,
            "agent_id": "system_agent",
            "event_type": "write_proposal",
            "timestamp": base_timestamp + (step - 1) * time_step,
            "proposal": {
                "subject": session_key,
                "predicate": "session_date",
                "object_val": session_date,
                "confidence": 1.0,
                "provenance": "locomo_session_metadata",
                "session_id": session_key,
            },
        })
        step += 1

        for turn in conversation.get(session_key, []):
            dia_id = str(turn.get("dia_id", f"{session_key}:{step}"))
            ordered_events.append({
                "step": step,
                "agent_id": "agent_a" if turn.get("speaker") == speaker_a else "agent_b",
                "event_type": "write_proposal",
                "timestamp": base_timestamp + (step - 1) * time_step,
                "proposal": {
                    "subject": turn.get("speaker", "unknown"),
                    "predicate": "utterance",
                    "object_val": turn.get("text", ""),
                    "confidence": 0.9,
                    "provenance": "locomo_conversation",
                    "raw_text": turn.get("text", ""),
                    "dia_id": dia_id,
                    "session_id": session_key,
                    "session_date": session_date,
                    "supports_answer": dia_id in evidence_ids,
                },
            })
            step += 1

    queries = []
    for qa in qa_items:
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        if not question:
            continue
        gold_answers = answer if isinstance(answer, list) else [answer]
        queries.append({
            "query_text": question,
            "gold_answers": gold_answers,
            "expected_retrieval_style": "best",
        })

    gold_memory_state = []
    for session_key in session_keys:
        session_date = conversation.get(f"{session_key}_date_time", "")
        gold_memory_state.append({
            "subject": session_key,
            "predicate": "session_date",
            "object_val": session_date,
            "status": "active",
            "confidence": 1.0,
            "provenance": "locomo_session_metadata",
        })
        session_events = event_summary.get(f"events_{session_key}", {})
        if isinstance(session_events, dict):
            for speaker, summaries in session_events.items():
                if speaker == "date":
                    continue
                for summary in _ensure_list(summaries):
                    gold_memory_state.append({
                        "subject": speaker,
                        "predicate": "session_event",
                        "object_val": summary,
                        "status": "active",
                        "confidence": 1.0,
                        "provenance": "locomo_event_summary",
                    })

    scenario_type = "locomo_long_conversation"
    if len(session_keys) >= 15:
        scenario_type = "locomo_very_long_term_memory"

    return generate_scenario(
        scenario_id=sample_id,
        scenario_type=scenario_type,
        agents=["agent_a", "agent_b"],
        ordered_events=ordered_events,
        gold_conflict_exists=False,
        gold_conflict_type="none",
        gold_resolution_action="append",
        gold_reconciled_memory_state=gold_memory_state,
        gold_visible_shared_state_after_commit=gold_memory_state,
        description=f"LoCoMo conversation: {len(session_keys)} sessions, {len(qa_items)} QA items",
        queries=queries,
        base_timestamp=base_timestamp,
    )


def _convert_lococo_item(item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """
    Convert a LoCoMo item to the repository's scenario format.

    Expected LoCoMo format (based on typical structure of memory benchmarks):
    - conversations: list of conversation turns with speaker and text
    - facts: ground truth facts that should be remembered
    - questions: memory-related questions with answers
    - metadata: conversation metadata (participants, duration, etc.)
    """
    conversation_id = item.get("conversation_id", f"lococo_{idx}")
    conversations = item.get("conversations", [])
    facts = item.get("facts", [])
    questions = item.get("questions", [])

    # Build ordered events
    ordered_events = []
    speaker_mapping = {}  # Map speakers to agent_a or agent_b

    for turn in conversations:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        turn_type = turn.get("type", "utterance")

        # Map speaker to agent_id (alternating for up to 2 speakers)
        if speaker not in speaker_mapping:
            # Assign next available agent
            agent_idx = len(speaker_mapping) % 2
            speaker_mapping[speaker] = f"agent_{'a' if agent_idx == 0 else 'b'}"

        agent_id = speaker_mapping[speaker]

        if turn_type == "utterance" and text:
            # Extract memory facts from utterance
            extracted_memories = _extract_facts_from_text(text, speaker)

            for mem in extracted_memories:
                proposal = {
                    "subject": mem.get("subject", speaker),
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

        elif turn_type == "query" or turn.get("is_question"):
            # This turn is a memory query
            query_text = text
            ordered_events.append({
                "step": len(ordered_events) + 1,
                "agent_id": agent_id,
                "event_type": "read",
                "query": query_text,
                "timestamp": turn.get("timestamp", len(ordered_events) * 60.0)
            })

    if not ordered_events:
        return None

    # Build evaluation queries from questions field
    eval_queries = []
    for q in questions:
        query_text = q.get("question", "")
        gold_answers = q.get("answers", [])
        if query_text:
            eval_queries.append({
                "query_text": query_text,
                "gold_answers": gold_answers if isinstance(gold_answers, list) else [gold_answers],
                "expected_retrieval_style": "best"
            })

    # Build gold memory state from facts
    gold_memory_state = []
    for fact in facts:
        gold_memory_state.append({
            "subject": fact.get("subject", "user"),
            "predicate": fact.get("predicate", "info"),
            "object_val": fact.get("object", ""),
            "status": "active"
        })

    # Detect conflicts
    conflict_exists = _detect_conflicts(ordered_events)

    # Determine scenario type based on conversation characteristics
    scenario_type = "lococo_conversation"
    if len(conversations) > 100:
        scenario_type = "long_conversation_memory"
    elif any("story" in c.get("text", "").lower() for c in conversations[:10]):
        scenario_type = "narrative_memory"

    scenario = generate_scenario(
        scenario_id=conversation_id,
        scenario_type=scenario_type,
        agents=list(set(speaker_mapping.values())) if speaker_mapping else ["agent_a", "agent_b"],
        ordered_events=ordered_events,
        gold_conflict_exists=conflict_exists,
        gold_conflict_type="memory_update" if conflict_exists else "none",
        gold_resolution_action="merge" if conflict_exists else "append",
        gold_reconciled_memory_state=gold_memory_state,
        gold_visible_shared_state_after_commit=gold_memory_state,
        description=f"LoCoMo conversation: {len(conversations)} turns, {len(facts)} facts",
        agent_profiles=None,
        queries=eval_queries,
        base_timestamp=1000.0
    )

    return scenario


def _session_sort_key(key: str) -> int:
    try:
        return int(key.split("_")[1])
    except Exception:
        return 0


def _ensure_list(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_facts_from_text(text: str, speaker: str) -> List[Dict[str, Any]]:
    """Extract factual statements from text."""
    facts = []

    text_lower = text.lower()

    # Memory-indicative patterns
    memory_patterns = [
        "i am", "i'm", "i have", "i've", "i live", "i work", "i study",
        "i like", "i love", "i hate", "i prefer", "my name is",
        "i'm from", "i am from", "i'm a", "i'm an", "i'm the"
    ]

    if any(pattern in text_lower for pattern in memory_patterns):
        # Extract as memory fact
        facts.append({
            "subject": "user" if "i " in text_lower[:30] else speaker,
            "predicate": "info",
            "object": text.strip(),
            "confidence": 0.8,
            "provenance": "explicit"
        })

    return facts


def _detect_conflicts(ordered_events: List[Dict[str, Any]]) -> bool:
    """Check for conflicting writes."""
    writes = [ev for ev in ordered_events if ev.get("event_type") == "write_proposal"]
    seen_keys = set()
    for w in writes:
        prop = w.get("proposal", {})
        key = (prop.get("subject"), prop.get("predicate"))
        if key in seen_keys:
            return True
        seen_keys.add(key)
    return False


if __name__ == "__main__":
    print("Testing LoCoMo loader...")
    scenarios = load_lococo(subset="test")
    if scenarios:
        print(json.dumps(scenarios[0], indent=2, ensure_ascii=False))
