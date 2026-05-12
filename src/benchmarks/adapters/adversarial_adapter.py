"""
Adversarial Benchmark Generator for Multi-Agent Conflict Resolution.

This module generates synthetic scenarios with controlled conflict types and parameters.
Useful for systematic testing and ablation studies.
"""

import json
import random
import sys
import os
import uuid
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add project root to path
# __file__ is in src/benchmarks/adapters/, so go up 3 levels to src, then one more to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.format import Scenario, MemoryEntry, Event, Query, CONFLICT_TYPES

# Predicate templates for different conflict types
PREDICATE_TEMPLATES = {
    "mutually_exclusive": [
        {"subject": "person", "predicate": "birth_place", "values": ["London", "Paris", "Tokyo", "NYC"]},
        {"subject": "person", "predicate": "current_location", "values": ["Boston", "Seattle", "SF", "LA"]},
        {"subject": "person", "predicate": "job_title", "values": ["Engineer", "Doctor", "Lawyer", "Teacher"]},
        {"subject": "product", "predicate": "price", "values": ["$10", "$20", "$30", "$40"]},
        {"subject": "event", "predicate": "date", "values": ["2024-01-01", "2024-02-01", "2024-03-01"]},
    ],
    "semantic_overlap": [
        {"subject": "person", "predicate": "skill", "values": [["Python", "ML"], ["Python", "AI"], ["ML", "AI"], ["Python", "ML", "AI"]]},
        {"subject": "project", "predicate": "technology", "values": [["React", "Node"], ["React", "Express"], ["Vue", "Node"], ["Angular", "Node"]]},
    ],
    "compatible_extension": [
        {"subject": "person", "predicate": "hobby", "values": [["reading"], ["reading", "hiking"], ["hiking", "reading", "cooking"]]},
        {"subject": "company", "predicate": "product", "values": [["CRM"], ["CRM", "Analytics"], ["CRM", "Analytics", "AI"]]},
    ],
    "stale_read_conflict": [
        {"subject": "order", "predicate": "status", "values": ["pending", "shipped", "delivered"]},
        {"subject": "task", "predicate": "completion", "values": ["0%", "50%", "100%"]},
    ],
    "concurrent_update": [
        {"subject": "counter", "predicate": "value", "values": ["1", "2", "3", "4"]},
        {"subject": "stock", "predicate": "quantity", "values": ["100", "101", "102", "103"]},
    ],
    "exact_duplicate": [
        {"subject": "constant", "predicate": "value", "values": ["42", "42", "42", "42"]},
    ],
    "potential_contradiction": [
        {"subject": "opinion", "predicate": "rating", "values": ["good", "bad", "neutral", "terrible"]},
    ],
}


class AdversarialGenerator:
    """Generate controlled adversarial scenarios for conflict resolution."""

    def __init__(
        self,
        num_agents: int = 2,
        base_timestamp: float = 1000.0,
        time_step: float = 10.0,
        confidence_range: Tuple[float, float] = (0.7, 1.0),
        seed: int = None
    ):
        self.num_agents = num_agents
        self.base_timestamp = base_timestamp
        self.time_step = time_step
        self.confidence_range = confidence_range
        self.rng = random.Random(seed)

    def generate_scenario(
        self,
        conflict_type: str,
        difficulty: str = "medium",
        num_writes: int = 5,
        scenario_id: str = None
    ) -> Scenario:
        """
        Generate a single scenario with specified conflict type.

        Args:
            conflict_type: One of CONFLICT_TYPES
            difficulty: "easy", "medium", "hard" - controls ambiguity
            num_writes: Number of write events to generate
            scenario_id: Optional scenario ID

        Returns:
            Scenario object
        """
        if scenario_id is None:
            scenario_id = f"adversarial_{conflict_type}_{uuid.uuid4().hex[:8]}"

        agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Generate template based on conflict type
        template = self._select_template(conflict_type)
        if not template:
            raise ValueError(f"No template for conflict type: {conflict_type}")

        # Generate events and determine gold resolution
        ordered_events, gold_action = self._generate_events(
            template, conflict_type, num_writes, difficulty
        )

        # Build gold memory state
        gold_reconciled = self._build_gold_state(ordered_events, conflict_type, gold_action)
        # gold_visible should only contain active entries
        gold_visible = [m for m in gold_reconciled if m.status == "active"]

        # Create query (retrieval test)
        query = self._create_query(template, conflict_type)

        scenario = Scenario(
            scenario_id=scenario_id,
            agents=agents,
            ordered_events=ordered_events,
            gold_conflict_exists=(conflict_type != "none"),
            gold_conflict_type=conflict_type,
            gold_resolution_action=gold_action,
            gold_reconciled_memory_state=gold_reconciled,
            gold_visible_shared_state_after_commit=gold_visible,
            scenario_type=f"adversarial_{conflict_type}",
            description=f"Adversarial {conflict_type} scenario, difficulty={difficulty}",
            queries=[query],
            base_timestamp=self.base_timestamp
        )

        return scenario

    def _select_template(self, conflict_type: str) -> Dict:
        """Select a random template for the given conflict type."""
        templates = PREDICATE_TEMPLATES.get(conflict_type, [])
        if not templates:
            return None
        return self.rng.choice(templates)

    def _generate_events(
        self,
        template: Dict,
        conflict_type: str,
        num_writes: int,
        difficulty: str
    ) -> Tuple[List[Event], str]:
        """Generate write events and determine the gold resolution action."""
        events = []
        values = template["values"]
        subject = template["subject"]
        predicate = template["predicate"]

        # Determine resolution based on conflict type
        if conflict_type == "mutually_exclusive":
            # Multiple writes with different values - newest with high confidence should overwrite
            gold_action = "overwrite"
        elif conflict_type == "semantic_overlap":
            # Overlapping values - should merge
            gold_action = "merge"
        elif conflict_type == "compatible_extension":
            # Additive - keep multiple versions or merge
            gold_action = "keep_multiple_versions" if difficulty == "hard" else "merge"
        elif conflict_type == "stale_read_conflict":
            # Agent reads stale, writes should be rejected
            gold_action = "reject"
        elif conflict_type == "concurrent_update":
            # Simultaneous writes - newer or higher confidence wins
            gold_action = "overwrite"
        elif conflict_type == "exact_duplicate":
            # Identical values - reject duplicate
            gold_action = "reject"
        elif conflict_type == "potential_contradiction":
            # Ambiguous - often defer in hard mode
            gold_action = "defer" if difficulty == "hard" else "overwrite"
        elif conflict_type == "none":
            gold_action = "append"
        else:
            gold_action = "defer"

        # Generate write events
        timestamp = self.base_timestamp

        for i in range(num_writes):
            agent_id = f"agent_{i % self.num_agents}"

            # Determine value for this write
            if conflict_type in ["mutually_exclusive", "concurrent_update", "exact_duplicate", "potential_contradiction"]:
                # Cycle through different values (or same for duplicate)
                if conflict_type == "exact_duplicate":
                    value = values[0]  # All same
                else:
                    value_idx = i % len(values)
                    value = values[value_idx]
            elif conflict_type in ["semantic_overlap", "compatible_extension"]:
                # Overlapping sets - each adds more
                value = values[min(i, len(values) - 1)]
            elif conflict_type == "stale_read_conflict":
                # Simulate stale read: first write ok, then another agent writes, then stale read
                if i == 0:
                    value = values[0]
                elif i == 1:
                    value = values[1]  # Another agent updates
                else:
                    value = values[0]  # Stale read tries to re-write old value
            else:
                # Default: cycle through values
                value_idx = i % len(values)
                value = values[value_idx]

            # Convert value to appropriate format
            if isinstance(value, list):
                object_val = json.dumps(value)
            else:
                object_val = str(value)

            # Adjust confidence based on conflict type to match gold action
            base_confidence = self.rng.uniform(*self.confidence_range)
            if conflict_type in ["mutually_exclusive", "concurrent_update"]:
                # For overwrite scenarios, ensure later writes have higher confidence
                # So that the arbitration favors the latest
                confidence = min(1.0, base_confidence + (i * 0.05))
            elif conflict_type == "stale_read_conflict":
                # Stale read (write3) should have moderate confidence but gets rejected due to staleness
                if i == 2:  # The stale write
                    confidence = base_confidence
                else:
                    confidence = min(1.0, base_confidence + 0.1)  # First two writes have higher confidence
            else:
                confidence = base_confidence

            confidence = round(confidence, 2)

            event = Event(
                step=i,
                agent_id=agent_id,
                event_type="write_proposal",
                timestamp=timestamp,
                proposal={
                    'subject': subject,
                    'predicate': predicate,
                    'object_val': object_val,
                    'confidence': confidence,
                    'provenance': 'adversarial_generated'
                }
            )
            events.append(event)
            timestamp += self.time_step

        # For stale_read_conflict, add read-before-write marker
        if conflict_type == "stale_read_conflict" and num_writes >= 2:
            # We need to insert a read event before the stale write
            # The pattern: write1 → write2 (by other agent) → read (snapshot old) → write3 (stale) → possibly more stale writes
            if len(events) >= 3:
                # Insert read event before the last write (index 2)
                read_timestamp = events[2].timestamp - 5.0
                read_event = Event(
                    step=events[2].step,
                    agent_id=events[2].agent_id,  # Same agent as the stale writer
                    event_type="read",
                    timestamp=read_timestamp,
                    query=None,
                    read_snapshot_time=None  # The snapshot time is implicit in the read
                )
                events.insert(2, read_event)
                # After insertion, all writes from index 3 onward are considered stale
                # because they occur after the read without a fresh read.
                for i in range(3, len(events)):
                    if events[i].event_type == "write_proposal":
                        events[i].read_snapshot_time = read_timestamp
                # Adjust steps for events after the insertion
                for i in range(4, len(events)):
                    events[i].step = i

        return events, gold_action

    def _build_gold_state(self, events: List[Event], conflict_type: str, gold_action: str) -> List[MemoryEntry]:
        """Build the gold memory state after all events."""
        # Filter only write proposals
        write_events = [e for e in events if e.event_type == "write_proposal"]

        if not write_events:
            return []

        # Collect all unique values for merge scenarios
        all_values = []
        subject = write_events[0].proposal['subject']
        predicate = write_events[0].proposal['predicate']

        for event in write_events:
            obj = event.proposal['object_val']
            # Parse JSON if it's a list
            try:
                val_list = json.loads(obj) if obj.startswith('[') else [obj]
            except:
                val_list = [obj]
            all_values.extend(val_list)

        # Build final state based on conflict_type and gold_action
        gold = []

        if conflict_type == "none":
            # All writes are active
            for event in write_events:
                mem = self._entry_from_event(event)
                mem.status = "active"
                gold.append(mem)

        elif conflict_type == "exact_duplicate":
            # Only the latest active
            for i, event in enumerate(write_events):
                mem = self._entry_from_event(event)
                if i == len(write_events) - 1:
                    mem.status = "active"
                else:
                    mem.status = "superseded"
                gold.append(mem)

        elif conflict_type == "stale_read_conflict":
            if gold_action == "reject":
                # Find the first read event timestamp (if any) to demarcate stale region
                first_read_time = None
                for ev in events:
                    if ev.event_type == "read":
                        first_read_time = ev.timestamp
                        break
                # Classify writes: any write after a read is stale; writes before read are fresh
                stale_indices = set()
                fresh_indices = []
                for i, event in enumerate(write_events):
                    if first_read_time is not None and event.timestamp > first_read_time:
                        stale_indices.add(i)
                    else:
                        fresh_indices.append(i)
                # Fresh writes: apply normal overwrite logic (latest wins)
                if fresh_indices:
                    latest_fresh_idx = fresh_indices[-1]
                    for i, event in enumerate(write_events):
                        if i in fresh_indices:
                            mem = self._entry_from_event(event)
                            if i == latest_fresh_idx:
                                mem.status = "active"
                            else:
                                mem.status = "superseded"
                            gold.append(mem)
                # Stale writes are rejected
                for i, event in enumerate(write_events):
                    if i in stale_indices:
                        mem = self._entry_from_event(event)
                        mem.status = "rejected"
                        gold.append(mem)
            else:
                # For gold_action "overwrite" or "defer", treat like other conflict types
                if gold_action == "overwrite":
                    for i, event in enumerate(write_events):
                        mem = self._entry_from_event(event)
                        if i == len(write_events) - 1:
                            mem.status = "active"
                        else:
                            mem.status = "superseded"
                        gold.append(mem)
                else:  # defer or others - keep all active for now
                    for event in write_events:
                        mem = self._entry_from_event(event)
                        mem.status = "active"
                        gold.append(mem)

        elif conflict_type == "semantic_overlap":
            if gold_action == "merge":
                # Create a single merged entry from all values
                # Merge by concatenating unique values
                unique_vals = []
                for v in all_values:
                    if v not in unique_vals:
                        unique_vals.append(v)
                merged_object = json.dumps(unique_vals) if len(unique_vals) > 1 else str(unique_vals[0])

                # Use the latest event as base for the merged entry
                base_event = write_events[-1]
                merged_mem = MemoryEntry(
                    subject=subject,
                    predicate=predicate,
                    object_val=merged_object,
                    agent_id=base_event.agent_id,
                    confidence=base_event.proposal.get('confidence', 1.0),
                    provenance='merged',
                    timestamp=base_event.timestamp
                )
                merged_mem.status = "active"
                gold.append(merged_mem)

                # Supersede all previous
                for event in write_events[:-1]:
                    mem = self._entry_from_event(event)
                    mem.status = "superseded"
                    gold.append(mem)
            else:
                # Keep all active (no merge)
                for event in write_events:
                    mem = self._entry_from_event(event)
                    mem.status = "active"
                    gold.append(mem)

        elif conflict_type == "compatible_extension":
            if gold_action == "keep_multiple_versions":
                # All active
                for event in write_events:
                    mem = self._entry_from_event(event)
                    mem.status = "active"
                    gold.append(mem)
            elif gold_action == "merge":
                # Merge all values into a single entry (same as semantic_overlap)
                unique_vals = []
                for v in all_values:
                    if v not in unique_vals:
                        unique_vals.append(v)
                merged_object = json.dumps(unique_vals) if len(unique_vals) > 1 else str(unique_vals[0])

                base_event = write_events[-1]
                merged_mem = MemoryEntry(
                    subject=subject,
                    predicate=predicate,
                    object_val=merged_object,
                    agent_id=base_event.agent_id,
                    confidence=base_event.proposal.get('confidence', 1.0),
                    provenance='merged',
                    timestamp=base_event.timestamp
                )
                merged_mem.status = "active"
                gold.append(merged_mem)

                for event in write_events[:-1]:
                    mem = self._entry_from_event(event)
                    mem.status = "superseded"
                    gold.append(mem)
            else:
                # Default: latest active
                for i, event in enumerate(write_events):
                    mem = self._entry_from_event(event)
                    if i == len(write_events) - 1:
                        mem.status = "active"
                    else:
                        mem.status = "superseded"
                    gold.append(mem)

        elif conflict_type == "potential_contradiction":
            # Ambiguous - defer means all tentative
            if gold_action == "defer":
                for event in write_events:
                    mem = self._entry_from_event(event)
                    mem.status = "tentative"
                    gold.append(mem)
            else:
                for event in write_events:
                    mem = self._entry_from_event(event)
                    mem.status = "active"
                    gold.append(mem)

        elif conflict_type == "potential_contradiction":
            if gold_action == "defer":
                # All tentative (not active)
                for event in write_events:
                    mem = self._entry_from_event(event)
                    mem.status = "tentative"
                    gold.append(mem)
            elif gold_action == "reject":
                for event in write_events:
                    mem = self._entry_from_event(event)
                    mem.status = "rejected"
                    gold.append(mem)
            else:
                # overwrite or append - latest active
                for i, event in enumerate(write_events):
                    mem = self._entry_from_event(event)
                    if i == len(write_events) - 1:
                        mem.status = "active"
                    else:
                        mem.status = "superseded"
                    gold.append(mem)

        else:
            # Default: all active
            for event in write_events:
                mem = self._entry_from_event(event)
                mem.status = "active"
                gold.append(mem)

        return gold

    def _entry_from_event(self, event: Event) -> MemoryEntry:
        """Create MemoryEntry from write event."""
        proposal = event.proposal
        return MemoryEntry(
            subject=proposal['subject'],
            predicate=proposal['predicate'],
            object_val=proposal['object_val'],
            agent_id=event.agent_id,
            confidence=proposal.get('confidence', 1.0),
            provenance=proposal.get('provenance', 'adversarial'),
            timestamp=event.timestamp
        )

    def _create_query(self, template: Dict, conflict_type: str) -> Query:
        """Create a retrieval query for this scenario."""
        subject = template["subject"]
        predicate = template["predicate"]

        # Query asks about the subject-predicate pair
        query_text = f"What is the {predicate} of {subject}?"

        # Gold answer is from the final visible state (last active entry)
        gold_values = []
        if isinstance(template["values"][0], list):
            # Take the last/most complete set
            gold_values = [json.dumps(template["values"][-1])]
        else:
            gold_values = [str(template["values"][-1])]

        return Query(
            query_text=query_text,
            gold_answers=gold_values,
            expected_retrieval_style="best"
        )

    def generate_dataset(
        self,
        num_scenarios: int,
        conflict_type_distribution: Dict[str, float] = None,
        difficulty: str = "medium"
    ) -> List[Scenario]:
        """
        Generate a full dataset with specified distribution.

        Args:
            num_scenarios: Total number of scenarios
            conflict_type_distribution: Dict mapping conflict_type to proportion (must sum to 1.0)
            difficulty: "easy", "medium", "hard"

        Returns:
            List of Scenario objects
        """
        if conflict_type_distribution is None:
            # Default balanced distribution
            conflict_types = list(CONFLICT_TYPES - {"none", "exact_duplicate", "semantic_duplicate"})
            n_types = len(conflict_types)
            conflict_type_distribution = {ct: 1.0 / n_types for ct in conflict_types}

        scenarios = []
        counts = {ct: 0 for ct in conflict_type_distribution}

        for _ in range(num_scenarios):
            # Choose conflict type according to distribution
            conflict_type = self.rng.choices(
                list(conflict_type_distribution.keys()),
                weights=list(conflict_type_distribution.values()),
                k=1
            )[0]

            num_writes = self.rng.randint(3, 10) if difficulty == "easy" else self.rng.randint(5, 15)

            scenario = self.generate_scenario(
                conflict_type=conflict_type,
                difficulty=difficulty,
                num_writes=num_writes
            )
            scenarios.append(scenario)
            counts[conflict_type] += 1

        print(f"Generated {len(scenarios)} scenarios:")
        for ct, count in sorted(counts.items()):
            print(f"  {ct}: {count} ({count/len(scenarios)*100:.1f}%)")

        return scenarios


def save_scenarios(scenarios: List[Scenario], output_path: str) -> None:
    """Save generated scenarios to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario.to_dict(), ensure_ascii=False) + '\n')
    print(f"Saved {len(scenarios)} scenarios to {output_path}")


def generate_benchmark(
    num_scenarios: int = 100,
    output_path: str = "data/processed/adversarial_benchmark.jsonl",
    difficulty: str = "medium",
    seed: int = 42
) -> List[Scenario]:
    """
    Generate and save a complete adversarial benchmark.

    Args:
        num_scenarios: Number of scenarios to generate
        output_path: Where to save the JSONL file
        difficulty: "easy", "medium", or "hard"
        seed: Random seed for reproducibility

    Returns:
        List of generated scenarios
    """
    generator = AdversarialGenerator(num_agents=2, seed=seed)

    # Balanced distribution across conflict types
    conflict_types = [
        "mutually_exclusive",
        "semantic_overlap",
        "compatible_extension",
        "stale_read_conflict",
        "concurrent_update",
        "potential_contradiction",
    ]
    distribution = {ct: 1.0 / len(conflict_types) for ct in conflict_types}

    scenarios = generator.generate_dataset(
        num_scenarios=num_scenarios,
        conflict_type_distribution=distribution,
        difficulty=difficulty
    )

    save_scenarios(scenarios, output_path)
    return scenarios


def load_adversarial_benchmark(
    num_scenarios: int = 100,
    difficulty: str = "medium",
    output_path: str = None
) -> List[Scenario]:
    """
    Load or generate adversarial benchmark.

    If output_path is provided and file exists, load from file.
    Otherwise generate new benchmark and optionally save to output_path.

    Args:
        num_scenarios: Number of scenarios
        difficulty: "easy", "medium", "hard"
        output_path: Optional path to save/load JSONL

    Returns:
        List of Scenario objects
    """
    if output_path and os.path.exists(output_path):
        print(f"Loading adversarial benchmark from {output_path}")
        # Load from JSONL
        scenarios = []
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    scenario = Scenario.from_dict(data)
                    scenarios.append(scenario)
                    if num_scenarios and len(scenarios) >= num_scenarios:
                        break
        print(f"Loaded {len(scenarios)} scenarios from {output_path}")
        return scenarios

    # Generate new benchmark
    scenarios = generate_benchmark(
        num_scenarios=num_scenarios,
        output_path=output_path if output_path else f"data/processed/adversarial_{difficulty}.jsonl",
        difficulty=difficulty
    )
    return scenarios


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate adversarial benchmark")
    parser.add_argument("--num-scenarios", type=int, default=100, help="Number of scenarios to generate")
    parser.add_argument("--output", type=str, default="data/processed/adversarial_benchmark.jsonl", help="Output path")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_benchmark(
        num_scenarios=args.num_scenarios,
        output_path=args.output,
        difficulty=args.difficulty,
        seed=args.seed
    )
