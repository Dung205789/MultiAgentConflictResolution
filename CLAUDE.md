# CLAUDE.md

## Response style
- Code first. Explain only if asked.
- No preamble: never restate the request, never say "Sure!", "Great question!", "Let me help you with..."
- No summaries at the end ("In summary...", "Hope this helps!")
- If fixing a bug: just fix it. Don't describe what the bug is.
- If multiple files need changes: change all of them in one shot.

## Format
- Use code blocks. Avoid prose when the answer is code.
- Inline comments only when non-obvious.
- No bullet points for things that fit in one line.

## On errors
- Read the traceback, fix immediately.
- Don't ask for more context if the information is already there.
- If something is truly missing: ask exactly one short question.

## On large tasks
- Work in order, one step at a time.
- After each step: short output only (file changed / result).
- No long plan before starting. Just start.

## Dynamic Contextual Arbitration
- `conflict_aware_writer.py` supports scenario-specific weights via `context_weights` in `configs/arbitration.yaml`.
- Pass `scenario_id` to `write()` to activate dynamic weights (e.g., `scenario_id="factual_dispute"`).
- `_calculate_uncertainty()` factors into arbitration decisions for `stale_read_conflict` and contradiction cases.
- Set `proposal["_enable_history_tracking"] = True` to log arbitration records via `ArbitrationHistoryTracker`.