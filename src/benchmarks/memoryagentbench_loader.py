"""
MemoryAgentBench adapter for Hugging Face dataset.

Integrates the MemoryAgentBench dataset into the multi-agent memory conflict evaluation harness.

Dataset: https://huggingface.co/datasets/THUDM/MemoryAgentBench
"""
import json
from typing import Dict, List, Any, Optional
from src.benchmarks.generator_core import generate_scenario


def load_memoryagentbench(subset: str = "all", num_samples: int = None) -> List[Dict[str, Any]]:
    """
    Load MemoryAgentBench from Hugging Face.

    Args:
        subset: Which subset to load ("all", "conflict", "temporal", "update", etc.)
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

    try:
        # Try streaming to save memory
        ds = load_dataset("THUDM/MemoryAgentBench", split="test", streaming=True)
    except Exception as e:
        print(f"Error loading MemoryAgentBench: {e}")
        print("Streaming not available, falling back to regular load (may use more memory)...")
        try:
            ds = load_dataset("THUDM/MemoryAgentBench", split="test")
        except Exception as e2:
            print(f"Failed to load MemoryAgentBench: {e2}")
            print("Make sure you have internet connection and the datasets library is properly installed.")
            return []

    scenarios = []
    for idx, item in enumerate(ds):
        # Convert MemoryAgentBench format to repository format
        scenario = _convert_memoryagentbench_item(item, idx, subset)
        if scenario:
            scenarios.append(scenario)
        # Stop if we reached the sample limit
        if num_samples is not None and len(scenarios) >= num_samples:
            break

    return scenarios


def _convert_memoryagentbench_item(item: Dict[str, Any], idx: int, subset_filter: str) -> Optional[Dict[str, Any]]:
    """
    Convert a MemoryAgentBench item to the repository's scenario format.

    MemoryAgentBench item structure (expected):
    - scenario_type: e.g., "conflict", "temporal", "update"
    - agents: list of agent names
    - events: list of events with agent_id, event_type, content/turn, timestamp
    - query: question to answer after processing events
    - gold_answer: correct answer
    - gold_update_action: expected resolution action
    - gold_state: expected final memory state

    Returns:
        Scenario dict compatible with repository evaluation harness.
    """
    scenario_type = item.get("scenario_type", "unknown")
    if subset_filter != "all" and scenario_type != subset_filter:
        return None

    agents = item.get("agents", ["agent_a", "agent_b"])
    events = item.get("events", [])

    # Convert events to ordered_events format
    ordered_events = []
    for ev in events:
        event_type = ev.get("event_type", "write_proposal")
        agent_id = ev.get("agent_id", "agent_a")
        content = ev.get("content", "")
        timestamp = ev.get("timestamp", 0.0)

        # Parse content into proposal if it's a write event
        if event_type == "write_proposal":
            # MemoryAgentBench may have structured content
            # We'll parse it into subject, predicate, object_val
            # For now, assume content is a simple text that we parse into a claim
            proposal = _parse_memory_content(content, agent_id)
            ordered_events.append({
                "step": len(ordered_events) + 1,
                "agent_id": agent_id,
                "event_type": "write_proposal",
                "proposal": proposal,
                "timestamp": timestamp,
                "read_snapshot_time": timestamp  # Assume read at same time for simplicity
            })
        elif event_type == "read":
            # Include read events
            ordered_events.append({
                "step": len(ordered_events) + 1,
                "agent_id": agent_id,
                "event_type": "read",
                "query": ev.get("query", ""),
                "timestamp": timestamp
            })

    # Determine gold values
    gold_conflict_exists = item.get("gold_conflict_exists", len([e for e in events if e.get("event_type") == "write_proposal"]) > 1)
    gold_conflict_type = item.get("gold_conflict_type", "none")
    gold_resolution_action = item.get("gold_update_action", "append")

    gold_state = item.get("gold_state", [])
    gold_reconciled_memory_state = gold_state
    gold_visible_shared_state_after_commit = gold_state

    # Add queries for downstream QA evaluation
    query = item.get("query")
    gold_answer = item.get("gold_answer")
    queries = []
    if query and gold_answer:
        queries.append({
            "query_text": query,
            "gold_answers": [gold_answer] if isinstance(gold_answer, str) else gold_answer,
            "expected_retrieval_style": "best"
        })

    scenario = generate_scenario(
        scenario_id=f"memoryagentbench_{subset_filter}_{idx}",
        scenario_type=scenario_type,
        agents=agents,
        ordered_events=ordered_events,
        gold_conflict_exists=gold_conflict_exists,
        gold_conflict_type=gold_conflict_type,
        gold_resolution_action=gold_resolution_action,
        gold_reconciled_memory_state=gold_reconciled_memory_state,
        gold_visible_shared_state_after_commit=gold_visible_shared_state_after_commit,
        description=f"MemoryAgentBench item {idx}: {scenario_type}",
        agent_profiles=None,  # Could be added if dataset provides agent metadata
        queries=queries,
        base_timestamp=item.get("base_timestamp", 1000.0)
    )

    return scenario


def _parse_memory_content(content: str, agent_id: str) -> Dict[str, Any]:
    """
    Parse memory content into a structured proposal.

    MemoryAgentBench content format can vary. We need to extract:
    - subject
    - predicate
    - object_val
    - confidence (if available)
    - provenance (if available)

    Simple heuristic: split by commas or use LLM extraction.
    For now, use a simple rule-based parser.
    """
    # Simple parsing: "subject predicate object" or JSON-like
    text = content.strip()

    # Try to parse as JSON
    if text.startswith('{') and text.endswith('}'):
        try:
            data = json.loads(text)
            return {
                "subject": data.get("subject", "user"),
                "predicate": data.get("predicate", "status"),
                "object_val": data.get("object", text),
                "confidence": data.get("confidence", 0.8),
                "provenance": data.get("provenance", "explicit")
            }
        except json.JSONDecodeError:
            pass

    # Simple heuristic parsing (very basic)
    # This should be enhanced based on actual MemoryAgentBench format
    parts = text.split(maxsplit=2)
    if len(parts) >= 3:
        subject = parts[0]
        predicate = parts[1]
        object_val = parts[2]
    else:
        subject = "user"
        predicate = "info"
        object_val = text

    return {
        "subject": subject,
        "predicate": predicate,
        "object_val": object_val,
        "confidence": 0.8,
        "provenance": "explicit"
    }


if __name__ == "__main__":
    # Test loading
    scenarios = load_memoryagentbench(subset="conflict")
    print(f"Loaded {len(scenarios)} MemoryAgentBench scenarios")
    if scenarios:
        print(json.dumps(scenarios[0], indent=2))
