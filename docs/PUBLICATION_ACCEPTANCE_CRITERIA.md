# Publication Acceptance Criteria

## Purpose
Define pass/fail standards for this project as a research paper.

These are not style opinions such as:
- "the idea sounds good"
- "the metrics look nice"
- "the system feels reasonable"

They are falsifiable criteria.

## Main Claim Pass Criteria
The main claim is only acceptable on the primary `oracle_structured` track if all of the following are true:

1. `conflict_aware > lww` on headline metrics:
- `FC-SH`
- `FC-MH`
- `SubEM`

2. No regression on arbitration correctness:
- `scenario_accuracy >= lww`
- `action_accuracy >= lww`

3. No contamination:
- `fallback_contamination_detected == false`

4. Reproducibility is in place:
- fixed rerun command
- versioned config
- stable artifact path
- invariant tests pass

## Core Code Pass Criteria
The symbolic core is only considered sharp enough if:

1. Query-aware propagation survives the full pipeline:
- graph edges from proposals are preserved
- query support enters arbitration metadata
- QA consumes explicit conflict-aware graph metadata without letting lineage contaminate the live answer graph

2. Conflict taxonomy is alive in runtime:
- `semantic_overlap`
- `compatible_extension`
- `potential_contradiction`
- `keep_multiple_versions`
must exist as real behaviors, not just names

3. Detector behavior is defensible:
- short entity-slot conflicts should not silently drift away from benchmark-critical overwrite semantics
- descriptive open-text predicates should not be forced into overwrite by default

4. Memory lifecycle is real:
- proposal
- committed
- visible
- superseded
- rejected
- tentative

## Contribution Isolation Pass Criteria
The claim that the memory system itself is a contribution is only valid if ablations support it:

1. Removing the conflict-aware core lowers headline metrics
2. Removing query-aware support lowers headline metrics or behavior quality
3. Removing lifecycle / visibility / supersession weakens metrics or action semantics

## Claim Discipline Pass Criteria
Allowed claims:
- paper-aligned symbolic shared-memory alternative
- primary track is `oracle_structured`
- secondary track is `end_to_end_extract`
- LLM remains proposal-only, not the final judge

Disallowed claims:
- paper-faithful 1:1 replication
- direct win over the original paper
- LLM as the final decision-maker

## Current Status Snapshot
Currently achieved:
- primary/secondary track split is locked
- symbolic writer remains the sole arbitration authority
- fallback contamination is explicit in reports
- a trusted full 8-scenario headline artifact exists
- invariant tests exist
- latest full code-state has `conflict_aware > lww`
- latest full code-state has `conflict_aware > no_query_support`

Not yet achieved:
- lineage-specific gains are still not proven beyond `no_lineage_edges`
- lifecycle-specific ablations are still missing
- official-comparable paper evaluation is still missing
- symbolic QA remains a major downstream bottleneck
