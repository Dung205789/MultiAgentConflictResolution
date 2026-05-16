# Research Status

Last updated: 2026-05-16

This file is the research-facing status tracker for `D:\ProjectMem`.
It is intended to be the shortest path for answering:

- what the project is actually trying to prove
- which claims are already supported by code and artifacts
- which gaps still block research-grade conclusions
- what to optimize next without drifting into toy metrics

## Canonical Direction

The current direction already stated in project docs is:

1. keep arbitration rule-based and explainable
2. avoid LLM dependence in the arbitration critical path
3. evaluate on real benchmark behavior, not only synthetic tasks
4. prioritize `MemoryAgentBench / Conflict_Resolution` until benchmark alignment is trustworthy
5. document progress so future agents can continue immediately

The most important operational clarification is in [SESSION_HANDOFF_STATUS.md](SESSION_HANDOFF_STATUS.md):

- local `state_match` is only an internal proxy
- the real research target is QA over benchmark `questions` and `answers`
- current local QA is useful, but still approximate

## What Is Confirmed

Confirmed from repo artifacts and code paths:

- `src/evaluation/qa_reasoner.py` provides a first deterministic QA-style evaluator over final visible memory.
- `src/benchmarks/memoryagentbench_loader.py` preserves `raw_text` and maps benchmark facts into ISF scenarios.
- `src/pipeline/multi_agent_pipeline.py` now evaluates QA on the final visible state instead of every write event, which reduces runtime blowups.
- `reports/mab_conflict_first2_afterfix.json` shows the local overwrite/state harness can reach perfect results on the first 2 CR scenarios.
- `reports/mab_cr_qa_first2_v5.json` shows QA exact match is meaningful but still not better than `lww` on the first 2 CR scenarios.
- `reports/postfix2_mab2_dummy/mab_conflict_report.json` now shows the first 2 CR scenarios back at `scenario_accuracy=1.0`, `conflict_f1=1.0`, `action_accuracy=1.0`, with QA exact match improved to `0.60`.
- `reports/postfix2_mab8_dummy/mab_conflict_report.json` shows the full local 8-scenario CR split stable again at perfect state/action parity for both `conflict_aware` and `lww`, with QA exact match `0.426`.

Interpretation:

- the system is no longer only a memory-state toy
- but it is not yet a publishable reproduction of the official benchmark result

## Current Problems

### 1. Benchmark validity gap

The main research risk is still evaluation validity.

- The local harness reconstructs gold shared-memory state, but the official task is QA.
- `MemoryAgentBench / Conflict_Resolution` in this checkout is only 8 scenarios and is overwrite-heavy.
- All 8 local CR scenarios map to `mutually_exclusive + overwrite`, so this split is good for overwrite validation but weak for showing advantage over `lww`.

Consequence:

- a perfect `state_match` run is not enough evidence
- even a good QA smoke result on 1-2 scenarios is still only a research proxy

### 2. Metric integrity issues

There are metric-level problems that can mislead research conclusions.

- `action_accuracy` is computed over conflict decisions, but historically `action_appropriateness_score` also counted first writes, which depressed the score even when action accuracy was perfect.
- This was corrected on 2026-05-15 in `src/evaluation/run_evaluation.py` so first writes no longer distort the appropriateness metric.
- Several metrics are still scenario-level while the underlying benchmark behavior is event-heavy and question-heavy, so long scenarios can dominate aggregate impressions.

Consequence:

- do not compare old and new `action_appropriateness_score` values as if they were identical metrics
- prefer QA exact match and final-memory correctness over derived heuristic scores

### 3. Runner and reproducibility gaps

The CLI surface is ahead of the actual execution path.

- `main.py` exposes `--use-dummy`, `--agent1-model`, `--agent2-model`, `--agent1-reliability`, and `--agent2-reliability`.
- The primary evaluation path still builds `MultiAgentPipeline` directly and does not cleanly honor those flags.
- `MultiAgentPipeline` builds plain `AgentRuntime` instances without benchmark-runner level model configuration.

Consequence:

- model-vs-rule experiments are not yet reproducible from the advertised CLI
- README examples overstate how complete the runner currently is

### 4. Documentation drift

Some top-level docs still read like the project has already cleared the main research bar.

Examples of drift:

- README headline claims near-SOTA and 10-20x cost advantage.
- README describes the runner as if real-model and benchmark comparisons are fully wired.
- `PROJECT_DOCUMENTATION.md` presents broad benchmark and contribution claims that are wider than the currently verified artifact set.

Consequence:

- docs should be read together with `SESSION_HANDOFF_STATUS.md`, not in isolation
- claims should be downgraded unless they are backed by current report artifacts

### 5. Research gap in the actual method

The strongest remaining gap is not raw overwrite logic. It is answer preservation under multi-hop reasoning.

Observed frontier:

- scenario `0` QA is materially easier than scenario `1`
- `conflict_aware` is now back at parity with `lww` on the first 2 and full 8 local CR scenarios, but still not better on QA
- the current system still resolves mostly at flat `(subject, predicate)` level

Implication:

- the next research gain likely comes from better graph-aware reasoning and answer-critical conflict handling, not more generic threshold tuning

## Research Priorities

### Priority 1: Improve benchmark alignment

Focus here first:

1. keep the benchmark `questions` and `answers` as the main target
2. analyze missed QA cases on scenario `1`
3. expand question templates and graph parsing only where misses are concrete
4. separate single-hop and multi-hop performance if possible

Success criterion:

- better QA exact match on the first 2 CR scenarios without harming final-memory correctness

### Priority 2: Make experiments reproducible

1. wire CLI agent/model flags into the real evaluation path
2. make the chosen agent mode explicit in report metadata
3. keep smoke runs small on this machine: 1 scenario, then 2, then full 8

Success criterion:

- a documented command line produces the same execution mode that the report claims

### Priority 3: Optimize for answer-critical memory

Candidate research directions with the best justification:

1. graph-aware conflict resolution instead of only flat overwrite
2. conflict clustering before arbitration so related facts are resolved together
3. lightweight provenance graph so downstream QA can prefer the right surviving fact chain
4. benchmark-specific symbolic rules for frequent multi-hop question families

Success criterion:

- `conflict_aware` starts beating `lww` on QA exact match, not only matching it on overwrite state

## Working Rules

Use these rules when reporting progress:

1. Do not claim benchmark parity from `state_match`.
2. Do not claim research improvement unless it improves a real report artifact.
3. Treat synthetic and adversarial runs as ablations, not the main headline.
4. Prefer small validated smoke sets over broad but weak claims.

## Files To Read First

1. [SESSION_HANDOFF_STATUS.md](SESSION_HANDOFF_STATUS.md)
2. [reports/mab_cr_qa_first2_v5.json](reports/mab_cr_qa_first2_v5.json)
3. [src/evaluation/qa_reasoner.py](src/evaluation/qa_reasoner.py)
4. [src/evaluation/run_evaluation.py](src/evaluation/run_evaluation.py)
5. [src/benchmarks/memoryagentbench_loader.py](src/benchmarks/memoryagentbench_loader.py)
6. [src/conflict/conflict_aware_writer.py](src/conflict/conflict_aware_writer.py)
