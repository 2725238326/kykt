# Dream Task Snapshot

Last updated: 2026-05-04 (pre-cycle-009 prep sub-pass: launch package + audit recorded; cycle 009 itself still gated on user D4'; mandatory-load + sync-rule integration completed; "Working rules to avoid F-001" section added)

Status: **idle, awaiting user decision**

## Why this file exists

This file is the highest-authority entry point for any Dream session — human, Codex, or another agent. It exists so that an interrupted task can be resumed cleanly, in this conversation or in a fresh one, without context loss.

Read order on session start:

1. **This file first** (`TASK_SNAPSHOT.md`)
2. Then `AGENT_MASTER_PROMPT.md` Mandatory Load Protocol (this file is item 1 of that protocol; the rest follows)
3. Then proceed per `AGENT_MASTER_PROMPT.md`

If this file's "Status" is `in_progress` or `blocked`, do NOT start new work. Resume the named task from `if_interrupted_resume_from` first.

If this file's "Last updated" timestamp is older than the latest cycle log under `cycles/`, this file is stale; trust the cycle log and update this file before doing anything else.

## Current task

```text
task_id:    cycle-008.5-closeout-and-cycle-009-prep
phase:      Phase 1.5 (Research Workflow Deployment)
cycle:      008.5 (sub-cycle within cycle 008)
status:     idle, awaiting user authorization for cycle 009 launch
```

One-line description:

```text
Cycle 008.5 closed: 4 finalist specs drafted at L1, cross-spec contract v1,
literature board v1, planning-layer aligned, inventory-sync sub-pass complete.
Cycle 009 (case-card filling) is gated on user "go" + the four open decision
points listed in WORKFLOW_STATUS.md "Recommended Next User Decision".
```

## Subtask board (latest pass: pre-cycle-009 launch package prep, 2026-05-04)

| ID | Subtask | Status | Canonical artifact |
|---|---|---|---|
| S1 | Pre-launch audit: scan `specs/`, `paradigm/`, `planning/` for `TODO` / `TBD` / `FIXME` / `XXX` | done | 0 hits; recorded in `planning/CYCLE_009_LAUNCH_PACKAGE.md` "Pre-launch audit findings" |
| S2 | Cross-spec contract v1 structural sanity check | done | Purpose / Scope / Signal Owner Table / Per-SPEC Published Signals (all 4) / 6 Conflict Resolution Rules / Versioning all present; recorded in same file |
| S3 | Spec -> contract back-reference scan; record asymmetry as cosmetic note CR-A1 | done | Composer spec links back; Critic / Memory / Permanence (drafted before contract) do not; recorded in same file as cosmetic, not blocking |
| S4 | Verify `paradigm/TEACHER_AUDIENCE_PROFILE.md` is a complete fill-in form (not invented content) | done | 7 user-input sections with example option menus already in place; agent must not populate |
| S5 | Draft cycle 009 subtask board skeleton under default D1'/D2' (parallel Composer + Critic, paper-derived) with conditional branches for sequential and paper+job-derived | done | `planning/CYCLE_009_LAUNCH_PACKAGE.md` "Cycle 009 subtask board skeleton" |
| S6 | Document activation procedure for the moment user `go` on D4' arrives | done | same file, "Activation procedure when user `go` arrives" |
| S7 | Update this `TASK_SNAPSHOT.md`: Last updated stamp + Last completed task pass + this subtask board | done | this file |

## Last completed task pass

```text
pass_name:        Pre-cycle-009 launch package prep sub-pass
date:             2026-05-04
trigger:          continuation immediately after the cycle 008.5
                  inventory-sync sub-pass closed; user asked for "next
                  step" but explicitly deferred decisions D1'..D4', so
                  this sub-pass executed only the work that does NOT
                  require those decisions
files_modified:   TASK_SNAPSHOT.md (this file; subtask board + Last
                  completed task pass + Last updated)
new_artifacts:    planning/CYCLE_009_LAUNCH_PACKAGE.md
                    - pre-launch audit (TODO/TBD scan clean across
                      specs and paradigm; cross-spec contract v1
                      structurally complete; spec -> contract back-
                      reference asymmetry recorded as cosmetic note
                      CR-A1, not blocking)
                    - cycle 009 subtask board skeleton under default
                      D1'/D2' (parallel Composer + Critic, paper-
                      derived only); branches for sequential and for
                      paper+job-derived recorded
                    - activation procedure for when user `go` arrives
discipline:       Surgical Edits + Honesty Override; no spec edits;
                  no cycle 009 case-card filling; no
                  TEACHER_AUDIENCE_PROFILE invention; no decision-
                  packet duplication beyond pointers

prior_pass_name:  Cycle 008.5 inventory-sync sub-pass
prior_pass_date:  2026-05-04
prior_pass_files: AGENT_MASTER_PROMPT.md, INDEX.md, README.md,
                  RESEARCH_STATE.md, WORKFLOW_STATUS.md,
                  cycles/CYCLE-20260504-001.md, logs/QUESTION_LOG.md,
                  registry/source_registry.md,
                  sources/FRONTIER_SOURCE_MAP.md,
                  units/REPRODUCTION_READINESS_MATRIX.md,
                  TASK_SNAPSHOT.md (introduced)
```

## If interrupted, resume from

If a new agent or new conversation is picking this up cold:

```text
1. Read this file (you are here).
2. Read cycles/CYCLE-20260504-001.md "Post-Closeout Inventory Sync" section.
3. Read RESEARCH_STATE.md "Inventory-Sync Sub-Pass (Cycle 008.5
   Post-Closeout)" subsection.
4. Read WORKFLOW_STATUS.md "Recommended Next User Decision" block.
5. Stop. Do NOT start cycle 009 case-card filling without the user's
   explicit "go" on the four open decision points below.
```

If this snapshot says `in_progress` instead of `idle` (i.e. a future task got interrupted again), treat the listed subtask board as the source of truth and resume from the first non-`done` row.

## Open user decisions (currently blocking next forward motion)

These are duplicated from `WORKFLOW_STATUS.md` "Recommended Next User Decision" so that this single file is enough to know what the user owes the project. Canonical wording lives in WORKFLOW_STATUS.md.

```text
1. Cycle 009 ordering: Composer case cards in parallel with Critic
   (default; cross-spec contract is the test path) vs sequential after
   Critic's first card lands.
2. Composer capability card source: paper-derived only (default) vs
   paper-derived + KYKT-job-derived (deferred to cycle 010 under default).
3. paradigm/TEACHER_AUDIENCE_PROFILE.md population (user input required;
   agent will not invent fields).
4. Cycle 009 launch authorization: confirm cycle 009 starts case-card
   filling (CASE-20260505-CRITIC-01..03 first per cycle 008 D1).
```

Items still blocked on user approval per `AGENT_MASTER_PROMPT.md` section 6 (unchanged from prior cycles): final thesis selection, reproduction, training, checkpoint download, KYKT navigation change, frontend implementation, reusable Codex skill packaging, retiring any non-finalist track, declaring teacher-demo readiness.

## Update protocol (highest priority — always honor)

This file MUST be updated at every meaningful task transition. The transitions are:

1. **Start of a task pass**: set `Status` to `in_progress`; populate `Current task`, `Subtask board`, and `If interrupted, resume from`.
2. **Each subtask completion**: flip the row's `Status` to `done` immediately. Don't batch.
3. **Mid-task interrupt or failure**: leave `Status` as `in_progress` and the active subtask as the last non-`done` row. Add a brief "Why interrupted" note. The next session resumes from there.
4. **End of a task pass (clean)**: flip `Status` to `idle`, mark `Last completed task pass` with what just finished, clear obsolete subtasks.
5. **End of a task pass (blocked on user)**: set `Status` to `blocked`; surface the blocker in `Open user decisions`.

Updating `Last updated:` is not optional. If you update this file you stamp it.

This rule is part of the Guidance File Sync Rule chain (see `WORKFLOW_STATUS.md`). It runs **first** in that chain — TASK_SNAPSHOT.md updates before any other file in a sync pass, so an interrupted sync still leaves a valid resume pointer.

## Recent failure modes the next agent should NOT repeat

Captured here as a short list because they are easy to repeat and expensive to recover from. Full prose lives in `cycles/CYCLE-20260504-001.md` "Note On The Earlier 32 MB Failure" and `RESEARCH_STATE.md` "Note On The Earlier 32 MB Failure".

```text
F-001  32 MB request-limit failure (2026-05-04, prior session)
       cause:  cumulative context (multiple full-file Reads + edit history)
               in one window, NOT any single oversized file
       avoid:  Read with offset/limit; Grep with -n for precision; Edit
               (old_string/new_string) instead of Write (full rewrite);
               keep <=2 large state files in context simultaneously;
               cite already-drafted content from cycle log + SPINE files
               instead of re-deriving it
```

Add new entries here as new failure modes are discovered. Do not delete entries; supersede via a "superseded by F-NNN" note (Discipline rule 5 Honesty Override).

## Working rules to avoid F-001 (anti-32MB request-limit)

These rules apply to every agent operating in this workspace, including humans driving an agent. Violating them is the most common cause of multi-edit pass failures.

1. Do not re-Read a file already read earlier in the same conversation. The content is still in context. Cite line numbers from the prior Read; if a slice is needed, use Read with `offset`+`limit`.
2. For lookup, prefer `Grep -n` with `-C` / `-A` / `-B` for context. Reserve full Read for files under ~300 lines OR when the whole file is genuinely needed.
3. Prefer `Edit` (old_string / new_string, diff-only payload) over `Write` (full file payload) for any file that already exists. Use `Write` only for new files.
4. Cap "large" files (>300 lines OR >20 KB) in active context at <=2 simultaneously. When starting a new sync sub-pass, treat earlier large files as evicted; rely on `TASK_SNAPSHOT.md` and the most recent cycle log summary instead of re-Reading.
5. If a single Edit's old_string / new_string would exceed ~200 lines, split into multiple smaller Edits anchored at stable section headers, not in the middle of paragraphs.
6. Sync `TASK_SNAPSHOT.md` FIRST in any Guidance File Sync Rule chain pass, so an interrupt mid-pass still leaves a valid resume pointer (this is the rule already restated in `WORKFLOW_STATUS.md` "Guidance File Sync Rule").
7. Do not run multi-file Read fan-outs in parallel just to "have everything"; pull files in only when the next concrete edit demands them.
8. If a single tool call returns "Request too large", do NOT retry with the same payload. Switch to: smaller Read window, narrower Grep, or split Edit. Record the trigger in this section's F-NNN list if it represents a new failure mode.

These rules are mandatory-equivalent to the Hard rules below, since violating them tends to lose a session's worth of work.

## Hard rules (carried from AGENT_MASTER_PROMPT.md, restated for safety)

1. No reproduction. No checkpoint download. No training. No KYKT navigation change. No frontend implementation. No thesis finalization. No retiring of any non-finalist track. None of these without explicit user approval **in the active conversation**.
2. Apply Surgical Edits (Discipline rule 3): every changed line traces to a user request, decision memo, or Sync Rule trigger.
3. Apply Honesty Override (Discipline rule 5): every mechanism claim carries an evidence label; user approval cannot be invented.
4. ID conventions are fixed: `SPEC-YYYYMMDD-NNN`, `DEC-YYYYMMDD-NNN`, `CYCLE-YYYYMMDD-NNN`, `CASE-YYYYMMDD-NNN`. Today's date prefix is `20260504` for new artifacts produced in this session.
5. Sentence case headers. No em-dashes.
