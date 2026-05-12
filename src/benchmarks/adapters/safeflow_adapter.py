"""
Adapter for SAFEFLOWBENCH dataset.

SAFEFLOWBENCH is an adversarial benchmark for evaluating
multi-agent conflict resolution under noisy, concurrent conditions.

This adapter converts SAFEFLOWBENCH data to ISF (Internal Standard Format).
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.format import Scenario, MemoryEntry, Event, Query, CONFLICT_TYPES, RESOLUTION_ACTIONS


class SAFEFLOWAdapter:
    """Adapter for SAFEFLOWBENCH dataset."""

    def __init__(self, dataset_path: Optional[str] = None, use_huggingface: bool = True):
        """
        Initialize SAFEFLOW adapter.

        Args:
            dataset_path: Path to local JSONL file (fallback if HuggingFace unavailable)
            use_huggingface: Try loading from HuggingFace first
        """
        self.dataset_path = dataset_path
        self.use_huggingface = use_huggingface
        self.dataset = None

    def load_data(self) -> List[Dict[str, Any]]:
        """Load the dataset from HuggingFace or local file."""
        if self.use_huggingface:
            try:
                from datasets import load_dataset
                print("Loading SAFEFLOWBENCH from HuggingFace...")
                dataset = load_dataset("lsflowers/SAFEFLOWBENCH", split="train")
                self.dataset = list(dataset)
                print(f"Loaded {len(self.dataset)} scenarios from HuggingFace")
                return self.dataset
            except ImportError:
                print("datasets library not installed. Install with: pip install datasets")
                print("Falling back to local file...")
            except Exception as e:
                print(f"Failed to load from HuggingFace: {e}")
                print("Falling back to local file...")

        # Fallback to local file
        if self.dataset_path and Path(self.dataset_path).exists():
            print(f"Loading from local file: {self.dataset_path}")
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = [json.loads(line) for line in f if line.strip()]
            print(f"Loaded {len(self.dataset)} scenarios from local file")
            return self.dataset

        raise FileNotFoundError(
            "No dataset found. Please either:\n"
            "1. Install datasets: pip install datasets\n"
            "2. Provide a local JSONL file path"
        )

    def convert_to_scenario(self, item: Dict[str, Any], index: int) -> Scenario:
        """
        Convert a SAFEFLOWBENCH item to ISF Scenario.

        Expected SAFEFLOWBENCH format:
        - scenario_id: Unique identifier
        - agents: List of agent IDs
        - events: List of events with agent_id, event_type, proposal, timestamp
        - conflict_type: Type of conflict
        - resolution: Expected resolution action
        - memory_state: Expected final memory state
        """
        scenario_id = item.get("scenario_id", f"safeflow_{index:04d}")

        # Extract agents
        agents = item.get("agents", [])
        if not agents and "events" in item:
            agents = list(set(e.get("agent_id") for e in item["events"] if e.get("agent_id")))

        # Convert events to ISF format
        events = []
        # SAFEFLOW uses "ordered_events" in raw format
        raw_events = item.get("ordered_events", item.get("events", []))
        for i, evt in enumerate(raw_events):
            event = Event(
                step=i + 1,
                agent_id=evt.get("agent_id", "unknown"),
                event_type=evt.get("event_type", "write_proposal"),
                timestamp=evt.get("timestamp", 1000.0 + i * 10),
                proposal=evt.get("proposal"),
                read_snapshot_time=evt.get("read_snapshot_time"),
            )
            events.append(event)

        # Map conflict type - use gold_conflict_type directly
        conflict_type = item.get("gold_conflict_type", "unknown")
        if conflict_type not in CONFLICT_TYPES:
            # Try to map similar types
            type_mapping = {
                "duplicate": "exact_duplicate",
                "overlap": "semantic_overlap",
                "extension": "compatible_extension",
                "exclusive": "mutually_exclusive",
                "stale": "stale_read_conflict",
            }
            conflict_type = type_mapping.get(conflict_type, "unknown")

        # Map resolution action - use gold_resolution_action directly
        resolution = item.get("gold_resolution_action", "defer")
        if resolution not in RESOLUTION_ACTIONS:
            action_mapping = {
                "accept": "overwrite",
                "reject": "reject",
                "merge": "merge",
                "keep": "keep_multiple_versions",
                "defer": "defer",
            }
            resolution = action_mapping.get(resolution, "defer")

        # Convert memory state
        memory_state = []
        for mem in item.get("memory_state", []):
            entry = MemoryEntry(
                subject=mem.get("subject", "unknown"),
                predicate=mem.get("predicate", "unknown"),
                object_val=mem.get("object_val", ""),
                status=mem.get("status", "active"),
                confidence=mem.get("confidence"),
                provenance=mem.get("provenance"),
                timestamp=mem.get("timestamp"),
                agent_id=mem.get("agent_id"),
            )
            memory_state.append(entry)

        # Convert queries
        queries = []
        for q in item.get("queries", []):
            query = Query(
                query_text=q.get("query_text", ""),
                gold_answers=q.get("gold_answers", []),
                expected_retrieval_style=q.get("expected_retrieval_style", "best"),
            )
            queries.append(query)

        return Scenario(
            scenario_id=scenario_id,
            scenario_type=item.get("scenario_type", conflict_type),
            description=item.get("description", ""),
            agents=agents,
            ordered_events=events,
            gold_conflict_exists=item.get("gold_conflict_exists", conflict_type != "none"),
            gold_conflict_type=conflict_type,
            gold_resolution_action=resolution,
            gold_reconciled_memory_state=memory_state,
            gold_visible_shared_state_after_commit=memory_state,
            queries=queries,
            agent_profiles=item.get("agent_profiles"),
            base_timestamp=item.get("base_timestamp", 1000.0),
        )

    def convert_all_to_scenarios(self, max_scenarios: Optional[int] = None) -> List[Scenario]:
        """Convert all dataset items to ISF Scenarios."""
        if self.dataset is None:
            self.load_data()

        scenarios = []
        for i, item in enumerate(self.dataset):
            if max_scenarios and i >= max_scenarios:
                break
            try:
                scenario = self.convert_to_scenario(item, i)
                scenarios.append(scenario)
            except Exception as e:
                print(f"Error converting scenario {i}: {e}")
                continue

        print(f"Converted {len(scenarios)} scenarios to ISF")
        return scenarios
