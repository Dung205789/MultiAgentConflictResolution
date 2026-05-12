"""
Adapter for MemAB (Memory Ability Benchmark) datasets.

MemAB contains two types:
1. Long_Range_Understanding: Long contexts with QA
2. Conflict_Resolution: List of facts with multi-hop reasoning questions

This adapter converts MemAB data to ISF (Internal Standard Format).
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.format import Scenario, MemoryEntry, Event, Query, CONFLICT_TYPES, RESOLUTION_ACTIONS


class MemABAdapter:
    """Adapter for MemAB benchmark datasets."""

    def __init__(self, dataset_path: str, variant: str = "conflict_resolution"):
        """
        Initialize MemAB adapter.

        Args:
            dataset_path: Path to the parquet file
            variant: Either "long_range_understanding" or "conflict_resolution"
        """
        self.dataset_path = dataset_path
        self.variant = variant
        self.df = None

    def load_data(self):
        """Load the parquet file."""
        import pandas as pd
        self.df = pd.read_parquet(self.dataset_path)
        return self.df

    def parse_facts_from_context(self, context: str) -> List[Dict[str, str]]:
        """
        Parse facts from context string.

        MemAB Conflict_Resolution contexts are formatted as:
        "Here is a list of facts:\n0. Thomas Kyd was born in the city of London.\n1. ..."

        Returns:
            List of fact dicts with subject, predicate, object_val
        """
        facts = []
        lines = context.split('\n')

        for line in lines:
            # Match pattern: "0. Subject predicate object."
            match = re.match(r'\s*\d+\.\s+(.+?)(?:\.|$)', line.strip())
            if match:
                fact_text = match.group(1).strip()
                # Simple parsing: split by first verb-like pattern
                # This is a simplified parser - a real one would need NLP
                # For now, keep the whole text as a fact
                facts.append({
                    'raw_text': fact_text,
                    'fact_id': len(facts)
                })

        return facts

    def _extract_entity_and_predicate(self, fact_text: str) -> Tuple[str, str, str]:
        """
        Extract (entity, predicate_type, object) from fact text.
        Returns: (entity, predicate, object_val)
        """
        text = fact_text.strip()
        if not text:
            return "unknown", "raw_statement", text

        # Pattern matching for common fact formats
        # "X was born in Y", "X lives in Y", "X works at Y", etc.
        verb_patterns = {
            r'\b(was|is|are|were)\s+born\s+in\b': ('birth_place', 2),
            r'\b(was|is|are|were)\s+born\s+at\b': ('birth_place', 2),
            r'\b(died|passed away)\s+in\b': ('death_place', 2),
            r'\blives\s+in\b': ('current_location', 1),
            r'\bworks?\s+(?:in|at)\b': ('work_location', 1),
            r'\bworks?\s+as\b': ('occupation', 1),
            r'\bplays?\s+(?:for|with|at)\b': ('affiliation', 1),
            r'\b(studies|studied)\s+(?:at|in)\b': ('education', 1),
            r'\bfounded\s+in\b': ('founder_location', 1),
            r'\bfounded\s+by\b': ('founder', -1),  # swap: X founded by Y -> subject=X, founder=Y
            r'\bauthored?\s+by\b': ('author', -1),
            r'\bdirected?\s+by\b': ('director', -1),
            r'\bmarried\s+to\b': ('spouse', 1),
            r'\bcitizen\s+of\b': ('citizenship', 1),
            r'\b(created|developed)\s+by\b': ('creator', -1),
            r'\b(performed|sang)\s+by\b': ('performer', -1),
            r'\bis\s+located\s+in\b': ('location', 2),
            r'\bcapital\s+of\b': ('capital', 1),
            r'\bspeaks\b': ('language', 1),
        }

        lower_text = text.lower()
        for pattern, (pred_type, entity_pos) in verb_patterns.items():
            match = re.search(pattern, lower_text)
            if match:
                # Extract entity before verb and object after
                parts = text.split(match.group(0), 1)
                before = parts[0].strip() if len(parts) > 1 else ""
                after = parts[1].strip() if len(parts) > 1 else ""

                # For "X founded by Y" pattern, entity is before "founded", object is after "by"
                if entity_pos == -1:
                    entity = before if before else "Unknown"
                    object_val = after
                else:
                    entity = before if before else "Unknown"
                    object_val = after

                return entity, pred_type, object_val

        # Fallback for "X of Y is Z" or "X of Y are Z" patterns (e.g., "The director of X is Y")
        # These are "raw_statements" but we should extract the full "X of Y" as entity
        match = re.match(r'^(.+?)\s+(?:of|in|at)\s+(.+?)\s+(?:is|are|was|were)\s+(.+?)$', text)
        if match:
            subject_phrase = match.group(1).strip()
            # location_phrase = match.group(2).strip()  # not used in entity
            object_val = match.group(3).strip()
            # Use full subject phrase as entity to avoid grouping unrelated facts
            entity = f"{subject_phrase} {match.group(2).strip()}"
            return entity, "raw_statement", object_val

        # Another fallback: "X is Y" - use X as entity if X is a proper noun phrase
        match = re.match(r'^(.+?)\s+(?:is|are|was|were)\s+(.+?)$', text)
        if match:
            subject_phrase = match.group(1).strip()
            object_val = match.group(2).strip()
            # If subject is a single word or short phrase, use it as is
            # But to avoid grouping, we may want to keep more context
            return subject_phrase, "raw_statement", object_val

        # Last resort: use first few words as entity
        words = text.split()
        if len(words) >= 3:
            # Use first 3 words as entity to provide more context than just first word
            entity = ' '.join(words[:min(3, len(words))])
            return entity, "raw_statement", text
        else:
            return "unknown", "raw_statement", text

    def fact_to_memory_entry(self, fact: Dict[str, str], agent_id: str, timestamp: float) -> MemoryEntry:
        """Convert a fact dictionary to MemoryEntry."""
        raw_text = fact['raw_text']
        # Extract entity to use as subject
        entity, predicate, object_val = self._extract_entity_and_predicate(raw_text)

        # Normalize subject to avoid too many unique subjects
        subject = entity[:100] if entity else f"fact_{fact['fact_id']}"

        return MemoryEntry(
            subject=subject,
            predicate=predicate,
            object_val=object_val if predicate != "raw_statement" else raw_text,
            status="active",
            confidence=1.0,
            provenance="memab_benchmark",
            timestamp=timestamp,
            agent_id=agent_id
        )

    def _analyze_facts_for_conflicts(self, facts: List[Dict], fact_entities: List[Tuple[str, str, str]]) -> Tuple[bool, str, str]:
        """
        Analyze facts to detect actual conflicts.

        Strategy:
        1. Use pre-extracted fact_entities (entity, predicate, object)
        2. Group facts by entity-predicate pairs
        3. Check for contradictory values on same predicate
        4. Determine conflict type and expected resolution

        Args:
            facts: List of fact dictionaries with raw_text
            fact_entities: List of (entity, predicate, object_val) tuples

        Returns: (has_conflict, conflict_type, resolution_action)
        """
        if len(facts) < 2:
            return False, "none", "append"

        # Group by (entity, predicate) to find mutually exclusive facts
        facts_by_key = {}
        for i, (entity, predicate, object_val) in enumerate(fact_entities):
            if entity and predicate:
                key = (entity[:50], predicate)  # Normalize entity length
                if key not in facts_by_key:
                    facts_by_key[key] = []
                facts_by_key[key].append({
                    'idx': i,
                    'object': object_val,
                    'raw_text': facts[i]['raw_text']
                })

        # Check for conflicts: same entity+predicate with different objects
        conflict_count = 0
        conflict_types = []

        for key, entries in facts_by_key.items():
            if len(entries) > 1:
                # Compare objects (case-insensitive)
                objects = [e['object'].lower().strip(' .,!?;:') if e['object'] else '' for e in entries]
                unique_objects = set(objects)
                if len(unique_objects) > 1:
                    conflict_count += 1
                    conflict_types.append('mutually_exclusive')

        # Check for semantic conflicts (many facts about same entity)
        if conflict_count == 0:
            # If we have multiple facts about the same entity but different predicates,
            # this could be semantic overlap
            entity_counts = {}
            for entity, _, _ in fact_entities:
                if entity:
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1

            if any(count > 5 for count in entity_counts.values()):
                conflict_count = 1
                conflict_types.append('semantic_overlap')

        if conflict_count == 0:
            return False, "none", "append"
        else:
            primary_type = max(set(conflict_types), key=conflict_types.count) if conflict_types else "mutually_exclusive"

            if primary_type == 'mutually_exclusive':
                return True, "mutually_exclusive", "overwrite"
            elif primary_type == 'semantic_overlap':
                return True, "semantic_overlap", "merge"
            else:
                return True, primary_type, "defer"

    def convert_row_to_scenario(self, row_idx: int, num_agents: int = 2) -> Scenario:
        """
        Convert a single row from MemAB to an ISF Scenario.

        For Conflict_Resolution:
        - Each fact is written by a different agent (cycling through num_agents)
        - Facts are written in order
        - Queries are derived from the questions
        - Conflict detection: We can simulate conflicts by having overlapping facts
        """
        if self.df is None:
            self.load_data()

        row = self.df.iloc[row_idx]
        scenario_id = f"memab_{self.variant}_{row_idx}"

        # Parse facts
        if self.variant == "conflict_resolution":
            facts = self.parse_facts_from_context(row['context'])
        else:
            # For Long_Range_Understanding, the context is a long narrative
            # We'll need a different strategy (maybe chunk into facts)
            facts = [{'raw_text': row['context'], 'fact_id': 0}]

        # Create agents
        agents = [f"agent_{i}" for i in range(num_agents)]

        # Create events: each fact is written by an agent
        ordered_events = []
        base_timestamp = 1000.0
        time_step = 10.0

        # Analyze facts for structured extraction
        fact_entities = []
        for fact in facts:
            entity, predicate, object_val = self._extract_entity_and_predicate(fact['raw_text'])
            fact_entities.append((entity, predicate, object_val))

        for i, fact in enumerate(facts):
            agent_id = agents[i % num_agents]
            entity, predicate, object_val = fact_entities[i]
            subject = entity[:100] if entity else f"fact_{fact['fact_id']}"

            event = Event(
                step=i,
                agent_id=agent_id,
                event_type="write_proposal",
                timestamp=base_timestamp + (i * time_step),
                proposal={
                    'subject': subject,
                    'predicate': predicate,
                    'object_val': object_val,
                    'confidence': 1.0,
                    'provenance': 'memab_benchmark'
                }
            )
            ordered_events.append(event)

        # Extract queries from questions
        queries = []
        questions = row['questions']
        answers = row['answers']

        # Convert numpy arrays to lists if needed
        if hasattr(questions, 'tolist'):
            questions = questions.tolist()
        if hasattr(answers, 'tolist'):
            answers = answers.tolist()

        # Ensure they are lists
        if not isinstance(questions, list):
            questions = [questions]
        if not isinstance(answers, list):
            answers = [answers]

        for q, a in zip(questions, answers):
            # Convert any non-serializable objects to strings
            q_text = str(q) if not isinstance(q, str) else q
            a_val = a.tolist() if hasattr(a, 'tolist') else a
            query = Query(
                query_text=q_text,
                gold_answers=[str(a_val)] if not isinstance(a_val, list) else [str(x) for x in a_val],
                expected_retrieval_style="best"
            )
            queries.append(query)

        # Analyze facts for conflicts using pre-extracted entities
        has_conflict, conflict_type, resolution_action = self._analyze_facts_for_conflicts(
            facts, fact_entities
        )

        # Build gold memory state based on conflict analysis
        gold_reconciled_memory_state = []
        gold_visible_shared_state_after_commit = []

        if has_conflict and conflict_type in ["mutually_exclusive", "counterfactual_temporal"]:
            # For mutually exclusive: latest wins per (entity, predicate)
            latest_by_key = {}
            for i, (entity, predicate, object_val) in enumerate(fact_entities):
                key = (entity, predicate)
                latest_by_key[key] = i

            for i, (fact, (entity, predicate, object_val)) in enumerate(zip(facts, fact_entities)):
                agent_id = agents[i % num_agents]
                subject = entity[:100] if entity else f"fact_{fact['fact_id']}"
                key = (entity, predicate)
                is_latest = (i == latest_by_key.get(key, i))

                status = "active" if is_latest else "superseded"
                mem = MemoryEntry(
                    subject=subject,
                    predicate=predicate,
                    object_val=object_val,
                    status=status,
                    confidence=1.0,
                    provenance="memab_benchmark",
                    timestamp=base_timestamp + (i * time_step),
                    agent_id=agent_id
                )
                gold_reconciled_memory_state.append(mem)
                if status == "active":
                    gold_visible_shared_state_after_commit.append(mem)
        else:
            # All active
            for i, fact in enumerate(facts):
                agent_id = agents[i % num_agents]
                entity, predicate, object_val = fact_entities[i]
                subject = entity[:100] if entity else f"fact_{fact['fact_id']}"

                mem = MemoryEntry(
                    subject=subject,
                    predicate=predicate,
                    object_val=object_val,
                    status="active",
                    confidence=1.0,
                    provenance="memab_benchmark",
                    timestamp=base_timestamp + (i * time_step),
                    agent_id=agent_id
                )
                gold_reconciled_memory_state.append(mem)
                gold_visible_shared_state_after_commit.append(mem)

        # Create scenario
        scenario = Scenario(
            scenario_id=scenario_id,
            agents=agents,
            ordered_events=ordered_events,
            gold_conflict_exists=has_conflict,
            gold_conflict_type=conflict_type,
            gold_resolution_action=resolution_action,
            gold_reconciled_memory_state=gold_reconciled_memory_state,
            gold_visible_shared_state_after_commit=gold_visible_shared_state_after_commit,
            scenario_type="memab_" + self.variant,
            description=f"MemAB benchmark sample {row_idx}",
            queries=queries,
            base_timestamp=base_timestamp
        )

        return scenario

    def convert_all_to_scenarios(self, num_agents: int = 2) -> List[Scenario]:
        """Convert entire dataset to list of ISF scenarios."""
        if self.df is None:
            self.load_data()

        scenarios = []
        for idx in range(len(self.df)):
            try:
                scenario = self.convert_row_to_scenario(idx, num_agents)
                scenarios.append(scenario)
            except Exception as e:
                print(f"Error converting row {idx}: {e}")
                continue

        return scenarios


def main():
    """Test the adapter."""
    import json

    # Test Conflict_Resolution adapter
    print("=== Testing MemAB Conflict_Resolution Adapter ===")
    adapter = MemABAdapter('data/raw/memab/Conflict_Resolution-00000-of-00001.parquet', 'conflict_resolution')
    scenarios = adapter.convert_all_to_scenarios(num_agents=2)

    print(f"Converted {len(scenarios)} scenarios")
    if scenarios:
        scenario = scenarios[0]
        print(f"\nScenario ID: {scenario.scenario_id}")
        print(f"Agents: {scenario.agents}")
        print(f"Number of events: {len(scenario.ordered_events)}")
        print(f"Number of queries: {len(scenario.queries)}")
        print(f"Conflict exists: {scenario.gold_conflict_exists}")
        print(f"Memory entries: {len(scenario.gold_reconciled_memory_state)}")

        # Save first scenario as JSON example
        scenario_dict = scenario.to_dict()
        output_path = 'output_memab_conflict_scenario.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_dict, f, indent=2, ensure_ascii=False)
        print(f"\nSaved example to {output_path}")

        # Print query safely using encode/decode
        try:
            print(f"\nFirst query: {scenario.queries[0].query_text[:100]}...")
            print(f"First answer: {scenario.queries[0].gold_answers[0][:50]}...")
        except UnicodeEncodeError:
            query_text = scenario.queries[0].query_text[:100].encode('utf-8', errors='ignore').decode('utf-8')
            answer_text = str(scenario.queries[0].gold_answers[0])[:50].encode('utf-8', errors='ignore').decode('utf-8')
            print(f"\nFirst query (unicode-safe): {query_text}...")
            print(f"First answer (unicode-safe): {answer_text}...")


if __name__ == "__main__":
    main()
