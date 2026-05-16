"""
MemoryAgentBench adapter for Hugging Face dataset.

Dataset: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
Correct dataset ID: ai-hyz/MemoryAgentBench (not THUDM/MemoryAgentBench)

Integrates the MemoryAgentBench dataset into the multi-agent memory conflict evaluation harness.
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from src.format import Scenario, MemoryEntry, Event, Query
from src.benchmarks.generator_core import generate_scenario


def load_memoryagentbench(subset: str = "all", num_samples: int = None) -> List[Scenario]:
    """
    Load MemoryAgentBench from Hugging Face.

    Args:
        subset: Which subset to load ("all", "Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding", "Conflict_Resolution")
        num_samples: Maximum number of samples to load (None for all)

    Returns:
        List of Scenario objects in ISF format.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found.")
        print("Please install: pip install datasets")
        return []

    # Correct dataset ID
    dataset_name = "ai-hyz/MemoryAgentBench"

    # Determine which splits to load
    if subset == "all":
        splits = ["Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding", "Conflict_Resolution"]
    else:
        splits = [subset]

    scenarios = []
    for split in splits:
        try:
            ds = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"Warning: Could not load split '{split}': {e}")
            continue

        for idx, item in enumerate(ds):
            scenario = _convert_memoryagentbench_item(item, split, idx)
            if scenario:
                scenarios.append(scenario)
            if num_samples is not None and len(scenarios) >= num_samples:
                break
        if num_samples is not None and len(scenarios) >= num_samples:
            break

    print(f"Loaded {len(scenarios)} MemoryAgentBench scenarios (subset={subset})")
    return scenarios


def _parse_facts_from_context(context: str) -> List[Dict[str, Any]]:
    """Parse facts from context string.

    Context format: "Here is a list of facts:\n0. Thomas Kyd was born in London.\n1. ..."
    Returns list of fact dicts with raw_text and fact_id.
    """
    facts = []
    lines = context.split('\n')

    for line in lines:
        # Match pattern: "0. Thomas Kyd was born in London."
        match = re.match(r'\s*\d+\.\s+(.+?)(?:\.|$)', line.strip())
        if match:
            fact_text = match.group(1).strip()
            facts.append({
                'raw_text': fact_text,
                'fact_id': len(facts)
            })

    return facts


def _extract_entity_and_predicate(fact_text: str) -> Tuple[str, str, str]:
    """
    Extract (entity, predicate_type, object) from fact text.
    """
    text = fact_text.strip()
    if not text:
        return "unknown", "raw_statement", text

    direct_patterns = [
        (r"^The chairperson of (.+?) is (.+?)$", "chairperson"),
        (r"^The director of (.+?) is (.+?)$", "director"),
        (r"^The author of (.+?) is (.+?)$", "author"),
        (r"^The chief executive officer of (.+?) is (.+?)$", "ceo"),
        (r"^The capital of (.+?) is (.+?)$", "capital"),
        (r"^The official language of (.+?) is (.+?)$", "official_language"),
        (r"^The name of the current head of state in (.+?) is (.+?)$", "head_of_state"),
        (r"^The Prime Minister of (.+?) is (.+?)$", "prime_minister"),
        (r"^The headquarters of (.+?) is located in the city of (.+?)$", "headquarters_city"),
        (r"^The univeristy where (.+?) was educated is (.+?)$", "educated_at"),
        (r"^The university where (.+?) was educated is (.+?)$", "educated_at"),
        (r"^The company that produced (.+?) is (.+?)$", "producer_company"),
        (r"^The company that originally broadcasted (.+?) is (.+?)$", "original_broadcaster"),
    ]
    for pattern, predicate in direct_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip(), predicate, match.group(2).strip()

    subject_patterns = [
        (r"^(.+?) is married to (.+?)$", "spouse"),
        (r"^(.+?) is a citizen of (.+?)$", "citizenship"),
        (r"^(.+?) is affiliated with the religion of (.+?)$", "religion"),
        (r"^(.+?) is associated with the sport of (.+?)$", "sport"),
        (r"^(.+?) plays the position of (.+?)$", "position"),
        (r"^(.+?) was born in the city of (.+?)$", "birth_place"),
        (r"^(.+?) died in the city of (.+?)$", "death_place"),
        (r"^(.+?) was founded by (.+?)$", "founder"),
        (r"^(.+?) was founded in the city of (.+?)$", "founder_location"),
        (r"^(.+?) was created in the country of (.+?)$", "origin_country"),
        (r"^(.+?) was performed by (.+?)$", "performer"),
        (r"^(.+?) was created by (.+?)$", "creator"),
        (r"^(.+?) is famous for (.+?)$", "known_for"),
        (r"^(.+?) is located in the continent of (.+?)$", "location"),
        (r"^(.+?) speaks the language of (.+?)$", "language"),
        (r"^(.+?) is employed by (.+?)$", "employer"),
        (r"^(.+?) worked in the city of (.+?)$", "work_location"),
        (r"^(.+?) works in the field of (.+?)$", "occupation"),
        (r"^(.+?)'s child is (.+?)$", "child"),
        (r"^The type of music that (.+?) plays is (.+?)$", "music_type"),
    ]
    for pattern, predicate in subject_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip(), predicate, match.group(2).strip()

    # Fallback
    words = text.split()
    if len(words) >= 3:
        entity = ' '.join(words[:2])
    else:
        entity = "unknown"
    return entity, "raw_statement", text


def _detect_conflicts_in_facts(facts: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
    """
    Analyze facts to detect conflicts.

    Returns: (has_conflict, conflict_type, resolution_action)
    """
    if len(facts) < 2:
        return False, "none", "append"

    # Build simple subject-predicate groups
    # Use heuristic to extract "subject predicate object" pattern
    fact_groups = {}  # key: (subject_guess, predicate_type) -> list of fact indices

    # Predicates that are typically mutually exclusive
    mutually_exclusive_verbs = {
        'born': 'birth_place',
        'died': 'death_place',
        'lives': 'current_location',
        'works': 'work_location',
        'plays': 'position',
        'is': 'identity',
        'founded': 'founder',
        'author': 'author',
        'director': 'director',
        'ceo': 'ceo',
        'chairperson': 'chairperson',
        'married': 'spouse',
        'citizen': 'citizenship',
        'located': 'location',
        'capital': 'capital',
        'language': 'language',
        'employed': 'employer',
        'educated': 'education',
    }

    for fact in facts:
        text = fact['raw_text'].lower()

        # Try to extract subject and predicate using verb patterns
        subject_guess = None
        predicate_type = None
        object_val = None
        subject_key = None  # Initialize

        # Look for key verbs
        for verb, pred_type in mutually_exclusive_verbs.items():
            if verb in text:
                # Extract subject (text before verb)
                parts = text.split(verb, 1)
                before_verb = parts[0].strip()
                # Subject is typically last noun phrase before verb
                subject_guess = before_verb.split()[-3:]  # Last few words
                subject_key = ' '.join(subject_guess)

                # Extract object (text after verb)
                if len(parts) > 1:
                    after_verb = parts[1].strip(' .,!?;:')
                    # First few words as object
                    object_val = ' '.join(after_verb.split()[:5])

                predicate_type = pred_type
                break

        if subject_key and predicate_type:
            key = (subject_key, predicate_type)
            if key not in fact_groups:
                fact_groups[key] = []
            fact_groups[key].append({
                'fact_idx': fact['fact_id'],
                'object': object_val,
                'full_text': fact['raw_text']
            })

    # Check for conflicts: same subject+predicate with different objects
    conflict_count = 0
    conflict_details = []

    for key, entries in fact_groups.items():
        if len(entries) > 1:
            # Check if objects differ
            objects = [e['object'].lower() if e['object'] else '' for e in entries]
            unique_objects = set(objects)
            if len(unique_objects) > 1:
                conflict_count += 1
                conflict_details.append({
                    'subject_predicate': key,
                    'entries': entries
                })

    if conflict_count > 0:
        # Determine primary conflict type
        # Most MemoryAgentBench conflicts are mutually exclusive facts
        return True, "mutually_exclusive", "overwrite"

    # Also check for semantic conflicts (similar but contradictory statements)
    # For now, if many facts (>50), likely some semantic overlap
    if len(facts) > 50:
        return True, "semantic_overlap", "merge"

    return False, "none", "append"


def _convert_memoryagentbench_item(item: Dict[str, Any], split: str, idx: int) -> Optional[Scenario]:
    """
    Convert a MemoryAgentBench item to ISF Scenario.

    The dataset has:
    - context: string with numbered facts
    - questions: list of questions
    - answers: list of answers
    - metadata: optional
    """
    context = item.get("context", "")
    if not context:
        return None

    # Parse facts from context
    facts = _parse_facts_from_context(context)
    if not facts:
        return None

    # Detect conflicts
    has_conflict, conflict_type, resolution_action = _detect_conflicts_in_facts(facts)

    # Create agents (alternating)
    num_agents = 2
    agents = [f"agent_{i}" for i in range(num_agents)]

    # Create ordered events (each fact as write proposal)
    ordered_events = []
    base_timestamp = 1000.0
    time_step = 10.0

    for i, fact in enumerate(facts):
        agent_id = agents[i % num_agents]
        raw_text = fact['raw_text']

        # Extract entity as subject
        entity, predicate, object_val = _extract_entity_and_predicate(raw_text)
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
                'confidence': 0.9,
                'provenance': 'memoryagentbench',
                'raw_text': raw_text,
            }
        )
        ordered_events.append(event)

    # Create queries from questions
    queries = []
    questions = item.get("questions", [])
    answers = item.get("answers", [])

    # Ensure lists
    if not isinstance(questions, list):
        questions = [questions] if questions else []
    if not isinstance(answers, list):
        answers = [answers] if answers else []

    for q, a in zip(questions, answers):
        if q:  # Only add if question exists
            query = Query(
                query_text=str(q),
                gold_answers=[str(a)] if not isinstance(a, list) else [str(x) for x in a],
                expected_retrieval_style="best"
            )
            queries.append(query)

    # Build gold memory state based on conflict type
    gold_reconciled_memory_state = []
    gold_visible_shared_state_after_commit = []

    # Extract entity for each fact
    fact_entities = []
    for fact in facts:
        entity, predicate, object_val = _extract_entity_and_predicate(fact['raw_text'])
        fact_entities.append((entity, predicate, object_val))

    if has_conflict and conflict_type == "mutually_exclusive":
        # For mutually exclusive: only latest active for each conflicting entity-predicate
        # Track latest write per (entity, predicate)
        latest_by_key = {}
        for i, (entity, predicate, object_val) in enumerate(fact_entities):
            key = (entity, predicate)
            latest_by_key[key] = i  # Will end up with last index

        for i, (fact, (entity, predicate, object_val)) in enumerate(zip(facts, fact_entities)):
            agent_id = agents[i % num_agents]
            subject = entity[:100] if entity else f"fact_{fact['fact_id']}"

            # Determine if this is the latest for its key
            key = (entity, predicate)
            is_latest = (i == latest_by_key.get(key, i))

            status = "active" if is_latest else "superseded"

            mem = MemoryEntry(
                subject=subject,
                predicate=predicate,
                object_val=object_val,
                status=status,
                confidence=0.9,
                provenance="memoryagentbench",
                timestamp=base_timestamp + (i * time_step),
                agent_id=agent_id
            )
            gold_reconciled_memory_state.append(mem)
            if status == "active":
                gold_visible_shared_state_after_commit.append(mem)
    else:
        # All facts active
        for i, fact in enumerate(facts):
            agent_id = agents[i % num_agents]
            entity, predicate, object_val = _extract_entity_and_predicate(fact['raw_text'])
            subject = entity[:100] if entity else f"fact_{fact['fact_id']}"

            mem = MemoryEntry(
                subject=subject,
                predicate=predicate,
                object_val=object_val,
                status="active",
                confidence=0.9,
                provenance="memoryagentbench",
                timestamp=base_timestamp + (i * time_step),
                agent_id=agent_id
            )
            gold_reconciled_memory_state.append(mem)
            gold_visible_shared_state_after_commit.append(mem)

    scenario = Scenario(
        scenario_id=f"memoryagentbench_{split}_{idx}",
        agents=agents,
        ordered_events=ordered_events,
        gold_conflict_exists=has_conflict,
        gold_conflict_type=conflict_type,
        gold_resolution_action=resolution_action,
        gold_reconciled_memory_state=gold_reconciled_memory_state,
        gold_visible_shared_state_after_commit=gold_visible_shared_state_after_commit,
        scenario_type=f"mab_{split.lower()}",
        description=f"MemoryAgentBench {split} sample {idx}: {len(facts)} facts, {len(queries)} queries",
        queries=queries,
        base_timestamp=base_timestamp
    )

    return scenario


if __name__ == "__main__":
    # Test loading
    print("Testing MemoryAgentBench loader...")
    for subset in ["Conflict_Resolution", "all"]:
        scenarios = load_memoryagentbench(subset=subset, num_samples=2)
        print(f"\nSubset '{subset}': loaded {len(scenarios)} scenarios")
        if scenarios:
            s = scenarios[0]
            print(f"  Sample ID: {s.scenario_id}")
            print(f"  Type: {s.scenario_type}")
            print(f"  Events: {len(s.ordered_events)}")
            print(f"  Facts: {len(s.gold_reconciled_memory_state)}")
            print(f"  Conflict: {s.gold_conflict_exists}, type={s.gold_conflict_type}")
            print(f"  Queries: {len(s.queries)}")
