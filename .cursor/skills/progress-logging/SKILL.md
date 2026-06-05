---
name: progress-logging
description: >-
  Maintains a daily project progress log in project_logs/ with timestamped,
  concise summaries of work done, changes made, and current focus. Apply
  automatically and passively throughout every session — after completing tasks,
  making file changes, running commands, debugging, or switching work areas.
  Do not wait for the user to ask. Update the log at natural breakpoints without
  announcing it unless the user asks.
---

# Progress Logging

Maintain a running daily log so the user has a timestamped record of project activity. **Do this passively** — treat logging as part of finishing work, not a separate user request.

## Log Location

```
project_logs/YYYY-MM-DD.md
```

- Folder: `project_logs/` at the project root (create if missing).
- File: one markdown file per calendar day, named `YYYY-MM-DD.md` (e.g. `2026-06-05.md`).
- Get the date via `date '+%Y-%m-%d'` if unsure.

## When to Log (automatic triggers)

Append an entry **without being asked** when any of these occur:

1. **Completed a user request** — implementation, fix, investigation, or answer that involved real work
2. **Made file changes** — created, edited, or deleted project files
3. **Ran meaningful commands** — benchmarks, builds, SLURM jobs, tests (note outcome: success/failure/key numbers)
4. **Debugged something** — record what was wrong and what was tried
5. **Started a new work area** — log current focus even if not finished yet (mark "in progress")
6. **End of a multi-step task** — one entry summarizing the whole task, not every micro-step

**Do not log** for: trivial acks, pure Q&A with no project changes, reading files only, or repeated retries of the same unchanged action.

**Do not** ask the user "should I log this?" — just do it quietly at the end of substantive work.

## How to Update

1. Ensure `project_logs/` exists.
2. Open (or create) today's `YYYY-MM-DD.md`.
3. **Append** a new entry at the bottom — never overwrite previous entries.
4. Keep the log update brief (3–6 bullet points max per entry). Do not let logging dominate the response.

### New file template

If today's file does not exist, create it with:

```markdown
# Project Log — YYYY-MM-DD

> Auto-maintained progress log. Newest entries at the bottom.

---
```

Then append the first entry.

### Entry format

```markdown
### HH:MM — Short title (5–10 words)
- What was done or changed (concise bullet)
- Key files/paths affected
- Outcome or result if applicable
- **In progress:** current focus (only if work is ongoing)
```

Get timestamp via `date '+%H:%M'` or from the user_info date. Use 24-hour or local time consistently within a file.

### Writing style

- **Concise** — telegraphic bullets, not paragraphs
- **Specific** — name files, kernels, scripts, errors; avoid vague "made improvements"
- **Action-oriented** — "Added X", "Fixed Y", "Benchmarked Z (2.3ms → 1.1ms)"
- **Honest status** — note failures, blockers, or open questions

### Example entries

```markdown
### 14:32 — GPU cluster skill
- Created `.cursor/skills/gpu-cluster/SKILL.md` with SLURM/gpu-turing docs
- Added `slurm/run_python.sh` wrapper for general Python-on-GPU execution
- Queried live partition info: 5 nodes, 20 GPUs, 30-min job cap

### 15:10 — RMSNorm correctness debug
- `src/kernels/rmsnorm/benchmark.py` fails `allclose` at batch=8, seq=512
- Compared against HF `Qwen2RMSNorm` — suspect stride/layout mismatch
- **In progress:** inspecting kernel shared-memory reduction
```

## Passive Behavior Rules

1. **Log after work, not before** — finish the task first, then append the entry
2. **No announcements** — do not tell the user "I updated the log" unless they ask about logging
3. **One entry per coherent unit of work** — batch related micro-edits into a single entry
4. **Merge if same minute** — if multiple tiny actions in quick succession, combine into one entry
5. **Never skip on multi-file sessions** — if you changed anything substantive, log it before ending your turn
6. **Reading the log** — if resuming work, skim today's file (and yesterday's if needed) to understand recent context

## What Not to Do

- Do not create logs outside `project_logs/`
- Do not create weekly/monthly files — daily only
- Do not write long prose or paste full code blocks
- Do not log sensitive data (passwords, tokens, private paths outside the project)
- Do not delete or rewrite historical entries

## Quick Checklist (internal, every substantive turn)

```
- [ ] Did I do substantive work this turn?
- [ ] Does project_logs/YYYY-MM-DD.md exist? (create if not)
- [ ] Append timestamped entry summarizing what changed and current status
- [ ] Move on — no need to mention the log to the user
```
