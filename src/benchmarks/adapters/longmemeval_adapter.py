"""
Adapter for LongMemEval benchmark.

LongMemEval tests multi-session memory retrieval across long conversation histories.
Each entry contains multiple conversation sessions and a question that requires
retrieving information from these sessions.

This adapter converts LongMemEval data to ISF (Internal Standard Format).
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.format import Scenario, MemoryEntry, Event, Query


class LongMemEvalAdapter:
    """Adapter for LongMemEval benchmark."""

    def __init__(self, dataset_path: str):
        """
        Initialize LongMemEval adapter.

        Args:
            dataset_path: Path to the JSON file (e.g., longmemeval_s_cleaned.json)
        """
        self.dataset_path = dataset_path
        self.data = None

    def load_data(self):
        """Load the JSON file."""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        return self.data

    def conversation_to_memory_events(self, sessions: List[List[Dict]], base_timestamp: float, time_step: float, agent_id: str = "user_agent") -> List[Event]:
        """
        Convert conversation sessions into memory events.

        Strategy:
        - Each user message that contains factual information becomes a write_proposal
        - Assistant responses that summarize or state facts also become write_proposals
        - The question at the end becomes a read query event
        """
        events = []
        step = 0

        for session_idx, session in enumerate(sessions):
            for msg_idx, message in enumerate(session):
                role = message['role']
                content = message['content']
                # Simple heuristic: messages with "I" statements or factual content are memories
                # In production, would use NLP to extract facts
                timestamp = base_timestamp + (step * time_step)

                # For now, treat each message as a memory write
                # A more sophisticated approach would extract specific facts/triples
                event = Event(
                    step=step,
                    agent_id=agent_id if role == "user" else "assistant_agent",
                    event_type="write_proposal",
                    timestamp=timestamp,
                    proposal={
                        'subject': f"msg_{session_idx}_{msg_idx}",
                        'predicate': 'statement',
                        'object_val': content[:500],  # Limit length
                        'confidence': 0.9,
                        'provenance': 'longmemeval_conversation',
                        'role': role
                    }
                )
                events.append(event)
                step += 1

        return events

    def extract_answer_from_sessions(self, sessions: List[List[Dict]], answer: str) -> List[MemoryEntry]:
        """
        Create gold memory entries that contain the answer.
        In a real implementation, would extract the specific memory that answers the question.
        For now, we'll create a synthetic memory entry with the answer.
        """
        # This is a simplification: the gold state should contain the actual facts
        # that support the answer, but we don't have explicit fact annotations
        # So we'll create a placeholder
        return [
            MemoryEntry(
                subject="gold_answer",
                predicate="answer_value",
                object_val=answer,
                status="active",
                confidence=1.0,
                provenance="longmemeval_gold",
                timestamp=1000.0,
                agent_id="gold_annotator"
            )
        ]

    def convert_entry_to_scenario(self, entry: Dict[str, Any], num_agents: int = 2) -> Scenario:
        """
        Convert a single LongMemEval entry to an ISF Scenario.
        """
        scenario_id = f"longmemeval_{entry['question_id']}"

        agents = ["agent_0", "assistant_agent"]
        base_timestamp = 1000.0
        time_step = 10.0

        # Convert sessions to events
        sessions = entry['haystack_sessions']
        ordered_events = self.conversation_to_memory_events(sessions, base_timestamp, time_step, agents[0])

        # Create query from the question
        query = Query(
            query_text=entry['question'],
            gold_answers=[entry['answer']],
            expected_retrieval_style="best"
        )

        # Build gold memory state from the ordered_events themselves
        # This ensures exact match with what the system will produce (assuming no conflicts)
        gold_reconciled_memory_state = []
        gold_visible_shared_state_after_commit = []

        for event in ordered_events:
            if event.event_type == "write_proposal":
                prop = event.proposal
                mem_entry = MemoryEntry(
                    subject=prop['subject'],
                    predicate=prop['predicate'],
                    object_val=prop['object_val'],
                    status="active",
                    confidence=prop.get('confidence', 1.0),
                    provenance=prop.get('provenance', 'inferred'),
                    timestamp=event.timestamp,
                    agent_id=event.agent_id
                )
                gold_reconciled_memory_state.append(mem_entry)
                gold_visible_shared_state_after_commit.append(mem_entry)

        # No extra gold_answer entry - retrieval is evaluated separately

        # LongMemEval doesn't have explicit conflicts, but could have outdated information
        has_conflict = False
        conflict_type = "none"
        resolution_action = "append"

        scenario = Scenario(
            scenario_id=scenario_id,
            agents=agents,
            ordered_events=ordered_events,
            gold_conflict_exists=has_conflict,
            gold_conflict_type=conflict_type,
            gold_resolution_action=resolution_action,
            gold_reconciled_memory_state=gold_reconciled_memory_state,
            gold_visible_shared_state_after_commit=gold_visible_shared_state_after_commit,
            scenario_type="longmemeval_" + entry['question_type'],
            description=entry['question'],
            queries=[query],
            base_timestamp=base_timestamp
        )

        return scenario

    def convert_all_to_scenarios(self, num_agents: int = 2, limit: int = None) -> List[Scenario]:
        """Convert entire dataset to list of ISF scenarios."""
        if self.data is None:
            self.load_data()

        scenarios = []
        data_subset = self.data[:limit] if limit else self.data

        for entry in data_subset:
            try:
                scenario = self.convert_entry_to_scenario(entry, num_agents)
                scenarios.append(scenario)
            except Exception as e:
                print(f"Error converting entry {entry.get('question_id', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return scenarios


def main():
    """Test the adapter."""
    import json

    print("=== Testing LongMemEval Adapter ===")
    adapter = LongMemEvalAdapter('data/raw/longmemeval/longmemeval_s_cleaned.json')
    scenarios = adapter.convert_all_to_scenarios(num_agents=2, limit=5)  # Test with 5 entries

    print(f"Converted {len(scenarios)} scenarios")
    if scenarios:
        scenario = scenarios[0]
        print(f"\nScenario ID: {scenario.scenario_id}")
        print(f"Type: {scenario.scenario_type}")
        print(f"Agents: {scenario.agents}")
        print(f"Number of events: {len(scenario.ordered_events)}")
        print(f"Number of queries: {len(scenario.queries)}")
        print(f"Memory entries: {len(scenario.gold_reconciled_memory_state)}")
        print(f"Question: {scenario.queries[0].query_text[:80]}...")
        print(f"Answer: {scenario.queries[0].gold_answers[0][:50]}...")

        # Save example
        scenario_dict = scenario.to_dict()
        output_path = 'output_longmemeval_scenario.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_dict, f, indent=2, ensure_ascii=False)
        print(f"\nSaved example to {output_path}")


if __name__ == "__main__":
    main()
