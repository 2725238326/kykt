# Dream Task Snapshot

Last updated: 2026-05-04 (cycle 008.5 inventory-sync sub-pass closed; gating on user cycle 009 launch authorization)

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

## Subtask board (latest pass: inventory-sync sub-pass, 2026-05-04)

| ID | Subtask | Status | Canonical artifact |
|---|---|---|---|
| S1 | Append Cycle 008.5 SPINE Anchor Map to `registry/source_registry.md` | done | section at end of file |
| S2 | Append symmetric SPINE Anchor Map to `sources/FRONTIER_SOURCE_MAP.md` | done | section at end of file |
| S3 | Update `units/REPRODUCTION_READINESS_MATRIX.md`: dormancy + finalist mapping + cycle 008 source-mining P3 additions + wake-up conditions | done | sections appended after Subagent Merge Notes |
| S4 | Update `logs/QUESTION_LOG.md`: Catchup Gap Note + Round 10 (cycle 008.5 user direction block) | done | top + bottom of file |
| S5 | Sync timestamps + sub-pass section in `AGENT_MASTER_PROMPT.md` / `README.md` / `INDEX.md` / `WORKFLOW_STATUS.md` / `RESEARCH_STATE.md` / `cycles/CYCLE-20260504-001.md` | done | last-updated lines + new subsections |
| S6 | Verify SPINE map symmetry; no orphan files; no duplicate Round IDs; no fabricated evidence labels | done | manual + grep verification recorded in cycle log |

## Last completed task pass

```text
pass_name:        Cycle 008.5 inventory-sync sub-pass
date:             2026-05-04
trigger:          continuation after a 32 MB request-limit failure in the
                  prior session interrupted the same sub-pass mid-edit
files_modified:   AGENT_MASTER_PROMPT.md, INDEX.md, README.md,
                  RESEARCH_STATE.md, WORKFLOW_STATUS.md,
                  cycles/CYCLE-20260504-001.md, logs/QUESTION_LOG.md,
                  registry/source_registry.md,
                  sources/FRONTIER_SOURCE_MAP.md,
                  units/REPRODUCTION_READINESS_MATRIX.md
new_artifacts:    none (this pass appends sections only)
discipline:       Surgical Edits + Honesty Override; no retro-renumbering;
                  no evidence-label changes; no reproduction authorization
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

## Hard rules (carried from AGENT_MASTER_PROMPT.md, restated for safety)

1. No reproduction. No checkpoint download. No training. No KYKT navigation change. No frontend implementation. No thesis finalization. No retiring of any non-finalist track. None of these without explicit user approval **in the active conversation**.
2. Apply Surgical Edits (Discipline rule 3): every changed line traces to a user request, decision memo, or Sync Rule trigger.
3. Apply Honesty Override (Discipline rule 5): every mechanism claim carries an evidence label; user approval cannot be invented.
4. ID conventions are fixed: `SPEC-YYYYMMDD-NNN`, `DEC-YYYYMMDD-NNN`, `CYCLE-YYYYMMDD-NNN`, `CASE-YYYYMMDD-NNN`. Today's date prefix is `20260504` for new artifacts produced in this session.
5. Sentence case headers. No em-dashes.
