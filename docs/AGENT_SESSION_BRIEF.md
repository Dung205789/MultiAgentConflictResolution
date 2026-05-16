# Agent Session Brief

This is the primary kickoff document for the next session. Read this file first, then start working immediately.

## 1. Project Name And What Already Exists

### Project name
`ProjectMem`

### Core objective
Build a **rule-based multi-agent shared-memory conflict resolution system** that is:

- independent from LLMs in the arbitration critical path
- explainable and cost-efficient
- evaluated on real benchmarks rather than toy tasks
- optimized toward benchmark-level research results rather than internal-only proxy wins

### What is already implemented

- benchmark entrypoint: [app/main.py](../app/main.py)
- unified benchmark loader: [src/benchmarks/unified_loader.py](../src/benchmarks/unified_loader.py)
- 3-mode execution pipeline (`conflict_aware`, `lww`, `naive`): [src/pipeline/multi_agent_pipeline.py](../src/pipeline/multi_agent_pipeline.py)
- arbitration logic: [src/conflict/conflict_aware_writer.py](../src/conflict/conflict_aware_writer.py)
- conflict detection: [src/conflict/conflict_detector.py](../src/conflict/conflict_detector.py)
- local QA evaluator: [src/evaluation/qa_reasoner.py](../src/evaluation/qa_reasoner.py)
- evaluation and report generation: [src/evaluation/run_evaluation.py](../src/evaluation/run_evaluation.py)
- Colab notebook for heavier model experiments: [notebooks/colab_runner_qwen_8b_3b_fixed.ipynb](../notebooks/colab_runner_qwen_8b_3b_fixed.ipynb)
- Kaggle full benchmark notebook: [kaggle/kaggle_runner_main_full.ipynb](../kaggle/kaggle_runner_main_full.ipynb)
- Kaggle push/poll/download helper: [kaggle/scripts/push_kaggle_kernel.ps1](../kaggle/scripts/push_kaggle_kernel.ps1)

### Reality check

- `state_match` is only an internal proxy
- the real benchmark target is QA over benchmark `questions` and `answers`
- research progress should be judged by benchmark-aligned artifacts, not synthetic headline numbers

## 2. Benchmarks That Matter

Priority order:

1. `MemoryAgentBench / Conflict_Resolution`
2. `LongMemEval`
3. `LoCoMo` / `locomo`
4. `SAFEFLOW`

Notes:

- `MemoryAgentBench / Conflict_Resolution` is the highest-priority benchmark
- the most important research target is QA exact match, especially multi-hop behavior
- `memae` is local-only and optional; it is not the main research headline benchmark

## 3. Models Used By The Project

### Default automation setup for Kaggle T4

- Agent 1: `Qwen/Qwen2.5-1.5B-Instruct`
- Agent 2: `Qwen/Qwen2.5-1.5B-Instruct`

Reason:

- more stable on Kaggle T4
- lower OOM / timeout risk
- better suited for repeated benchmark loops

### Heavier experimental setup

Notebook:
[notebooks/colab_runner_qwen_8b_3b_fixed.ipynb](../notebooks/colab_runner_qwen_8b_3b_fixed.ipynb)

Models:

- Agent 1: `Qwen/Qwen3-8B`
- Agent 2: `Qwen/Qwen2.5-3B-Instruct`

Use this only for targeted experiments, not as the default automation profile.

## 4. Automation Workflow

### Workflow objective

Repeat this loop until the research target is met or the GPU session ends:

1. sync the latest code
2. push code to GitHub
3. push the notebook to Kaggle
4. run benchmarks through `app/main.py`
5. pull `reports` back to local
6. analyze metrics and failures
7. optimize code
8. repeat

### Tokens, files, and folders

#### Local Kaggle authentication

- file: [kaggle/kaggle.json](../kaggle/kaggle.json)
- purpose: authenticate the `kaggle` CLI for kernel push, status polling, and output download

#### Hugging Face token

- file: [kaggle/.env](../kaggle/.env)
- key: `HF_TOKEN`
- purpose:
  - download models
  - download benchmark data from Hugging Face when needed

#### Extra Kaggle token

- file: [kaggle/.env](../kaggle/.env)
- key: `KAGGLE_TOKEN`
- purpose:
  - optional fallback for direct API/REST flows if needed later
  - current workflow primarily uses `kaggle/kaggle.json` with the Kaggle CLI

#### GitHub remote

- remote: `origin`
- purpose:
  - push the latest source code so the Kaggle notebook can clone and run it

### Files and folders used by the workflow

#### Benchmark execution

- entrypoint: [app/main.py](../app/main.py)
- benchmark loading: [src/benchmarks/unified_loader.py](../src/benchmarks/unified_loader.py)
- report output: [reports](../reports)

#### Kaggle automation

- notebook: [kaggle/kaggle_runner_main_full.ipynb](../kaggle/kaggle_runner_main_full.ipynb)
- kernel metadata: [kaggle/kernels/projectmem_full/kernel-metadata.json](../kaggle/kernels/projectmem_full/kernel-metadata.json)
- push helper: [kaggle/scripts/push_kaggle_kernel.ps1](../kaggle/scripts/push_kaggle_kernel.ps1)

#### Colab heavy-model experiments

- notebook: [notebooks/colab_runner_qwen_8b_3b_fixed.ipynb](../notebooks/colab_runner_qwen_8b_3b_fixed.ipynb)

### Standard operating loop

#### Default loop

1. modify code in `src/`
2. commit and push to `origin`
3. sync the Kaggle notebook with the latest code
4. push the Kaggle kernel
5. wait for the kernel to finish or exhaust the GPU session
6. pull outputs back to local
7. read `summary.json` and benchmark report JSON files
8. identify the current bottleneck benchmark or failure mode
9. patch code
10. repeat

#### Optimization priority inside each loop

1. `MemoryAgentBench / Conflict_Resolution` QA exact match
2. single-hop vs multi-hop separation where possible
3. honest benchmark framing
4. only claim improvement when a report artifact proves it

### Stop conditions

Stop when one of the following is true:

1. results are close enough to the real benchmark target
2. artifacts show no meaningful additional improvement
3. the Kaggle GPU session ends

## 5. What The Next Agent Should Do Immediately

### Read in this order

1. [AGENT_SESSION_BRIEF.md](AGENT_SESSION_BRIEF.md)
2. [RESEARCH_STATUS.md](RESEARCH_STATUS.md)
3. [SESSION_HANDOFF_STATUS.md](SESSION_HANDOFF_STATUS.md)

### Then act directly

1. check `git status`
2. check Kaggle notebook and kernel metadata
3. push code to GitHub
4. run the Kaggle full-benchmark workflow through `app/main.py`
5. pull outputs back to local
6. analyze reports
7. optimize code and repeat

### Mandatory rules

- do not confuse `state_match` with benchmark parity
- do not drift into toy benchmarks while the main benchmark is unresolved
- do not claim SOTA without directly comparable benchmark artifacts
- prioritize `MemoryAgentBench / Conflict_Resolution` first

## 6. What Is Still Missing For A Complete Research Product

This is the most important gap summary to keep in mind before the next optimization loop.

### Missing 1: truly benchmark-valid evaluation

The biggest gap is still metric validity.

- the repo now has a QA proxy
- but it has not yet proven that this proxy is equivalent to the official benchmark evaluation
- especially for `MemoryAgentBench / Conflict_Resolution`, the real task is multi-hop QA, not just state reconstruction

Conclusion:

- current results must still be treated as `research proxy` results
- they are not yet strong enough to claim benchmark parity

### Missing 2: clear wins over a strong baseline

The project already shows:

- improvement over `naive`
- stability on the overwrite-heavy CR split
- but not yet a clear QA win for `conflict_aware` over `lww`

Without this, the system is logically competent but not yet a strong research result.

### Missing 3: a sharp enough method contribution

The main frontier is no longer basic overwrite logic.
The project still needs a research contribution with real bite, for example:

- graph-aware conflict resolution
- answer-critical memory preservation
- multi-hop evidence chaining
- conflict clustering across related facts

If the work stays at threshold and weight tuning, it is more engineering refinement than research contribution.

### Missing 4: a reproducible closed-loop experiment workflow

The project still needs a clearly reproducible loop:

1. push code
2. run benchmark
3. pull reports
4. analyze
5. patch
6. repeat

This loop must produce consistent artifacts and preserve experiment history across iterations.

### Missing 5: a research analysis layer

JSON reports alone are not enough.
The project still needs:

- consolidated benchmark tables
- single-hop vs multi-hop breakdown where possible
- failure analysis
- case studies
- ablations
- explanations for why metrics improved or regressed

Without this, the system can run experiments but still lacks a research narrative.

### Missing 6: a full paper package

To count as a complete research product, the repo still needs:

- a clear problem statement
- explicit claims that are actually supported
- a repeatable experiment protocol
- final benchmark tables
- limitations
- threat-to-validity discussion
- a clean release package: code, notebooks, scripts, reports, instructions

### Short conclusion

The single most important missing piece right now is:

`proving improvement on real QA benchmark behavior, especially MemoryAgentBench multi-hop, not just on state reconstruction`

## 7. Done State For This Session

This session should be considered done when:

1. the real research target of the repo is explicitly locked in
2. this briefing exists as the central handoff doc: [AGENT_SESSION_BRIEF.md](AGENT_SESSION_BRIEF.md)
3. benchmark, model, token, file, folder, and automation workflow details are all explicitly written down
4. the remaining gaps to a complete research product are clearly summarized
5. the next session can treat this file as the default starting point and move directly into benchmark optimization
