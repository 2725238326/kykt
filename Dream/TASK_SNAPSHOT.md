# Dream Task Snapshot

Last updated: 2026-05-04 (cycle 009 activation pass: D1'-D4' locked by user delegation, DEC-20260504-003 + CYCLE-20260505-001 written; CASE-CRITIC-01 draft deferred to next pass per F-001 budget guard)

Status: **in_progress** (cycle 009 launched; S5 of activation board outstanding)

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
task_id:    cycle-009-activation
phase:      Phase 1.5 (Research Workflow Deployment)
cycle:      009 (case-card filling: Critic + Composer parallel, paper-derived)
status:     in_progress; S5 outstanding; S6 deferred to next pass; S7 pending
```

One-line description:

```text
User delegated D1'-D4' to agent ("D1-D4 你自己决策吧，有问题我们商讨").
Agent locked: D1'=parallel, D2'=paper-derived, D3'=unpopulated-deferred
(TEACHER_AUDIENCE_PROFILE.md requires real-teacher facts the agent has no
source for; case cards are paper-derived under D2' so they do not depend
on it), D4'=go. DEC-20260504-003 records the four locks with rationales.
CYCLE-20260505-001.md opens cycle 009 with S1 done (launch) and S2 in
progress (CASE-CRITIC-01 draft). The earlier cycle-009 case-card draft
attempt in this same pass tripped the 32 MB request limit; per F-001
working rules the draft is deferred to a fresh pass with a clean context.
```

## Subtask board (latest pass: cycle 009 activation, 2026-05-04)

| ID | Subtask | Status | Canonical artifact |
| --- | --- | --- | --- |
| S1 | Lock D1'-D4' values under user delegation, with rationales | done | `decisions/DEC-20260504-003-cycle-009-launch.md` |
| S2 | Discover where case cards live (`templates/proxy_case_card.md` is the 138-line skeleton; `cases/` directory does not yet exist) | done | confirmed via Grep + Read; recorded in DEC-20260504-003 |
| S3 | Write `decisions/DEC-20260504-003-cycle-009-launch.md` recording D1'-D4' verbatim with rationales | done | same file (76 lines) |
| S4 | Create `cycles/CYCLE-20260505-001.md` with header + subtask board copied from launch package | done | `cycles/CYCLE-20260505-001.md` (50 lines) |
| S5 | Update this `TASK_SNAPSHOT.md`: status -> in_progress, current task, subtask board, If-interrupted-resume-from, Open user decisions resolution | done | this file |
| S6 | Read `specs/SPEC-20260503-001.md` (Critic), draft `cases/CASE-20260505-CRITIC-01.md` from `templates/proxy_case_card.md` | deferred | next pass; checkpoint chosen per F-001 budget guard (32 MB limit hit on the same pass) |
| S7 | Commit + push the cycle-009 activation (DEC-20260504-003 + CYCLE-20260505-001 + this snapshot); brief report | pending | git history |

## Last completed task pass

```text
pass_name:        Cycle 009 activation pass (partial; S5 in progress, S6
                  deferred, S7 pending)
date:             2026-05-04
trigger:          User message "D1-D4 你自己决策吧，有问题我们商讨"
                  delegating the four open decisions to the agent, with
                  an open consultation channel for any genuinely
                  blocking issue
files_modified:   TASK_SNAPSHOT.md (this file; header + subtask board +
                  last completed task pass + open user decisions)
new_artifacts:    decisions/DEC-20260504-003-cycle-009-launch.md
                    - locks D1'=parallel, D2'=paper-derived,
                      D3'=unpopulated-deferred, D4'=go
                    - records why D3' is the only "agent flagged for
                      user" item: TEACHER_AUDIENCE_PROFILE.md needs
                      Research Taste / Hard Constraints / Demo
                      Precedent for a specific real teacher; agent has
                      no factual source so leaves it unpopulated and
                      proceeds with paper-derived cards (consistent
                      with D2')
                  cycles/CYCLE-20260505-001.md
                    - cycle 009 header (Last updated, status: launched)
                    - subtask board: S1 done, S2 in progress (CASE-
                      CRITIC-01 draft)
discipline:       Surgical Edits + Honesty Override; D3' explicitly
                  flagged as "unpopulated, deferred" rather than
                  invented (Discipline rule 5 Honesty Override); F-001
                  working rule honored by switching to checkpoint when
                  the 32 MB ceiling was hit mid-pass instead of
                  retrying the same payload
budget_event:     32 MB request-too-large fired during this pass on a
                  Write attempt that followed multiple full-file Reads;
                  this is exactly F-001. Switched to Edit-only mode and
                  deferred CASE-CRITIC-01 draft. No new failure mode;
                  no new F-NNN entry needed.

prior_pass_name:  Pre-cycle-009 launch package prep sub-pass
prior_pass_date:  2026-05-04
prior_pass_files: planning/CYCLE_009_LAUNCH_PACKAGE.md (introduced),
                  TASK_SNAPSHOT.md
```

## If interrupted, resume from

If a new agent or new conversation is picking this up cold:

```text
1. Read this file (you are here).
2. Read decisions/DEC-20260504-003-cycle-009-launch.md (D1'-D4' locks
   + rationales).
3. Read cycles/CYCLE-20260505-001.md (cycle 009 board).
4. Resume at S5 of the activation board if this snapshot's Status is
   still in_progress; if S5 already shows done, resume at S6 (draft
   CASE-20260505-CRITIC-01.md from templates/proxy_case_card.md against
   specs/SPEC-20260503-001.md, paper-derived per D2').
5. After S6 lands, S7 = commit + push.
6. Honor F-001 working rules: do not Read large files already cited in
   this snapshot; prefer Grep -n + Edit over full-file Read + Write.
```

If this snapshot says `idle` instead of `in_progress`, the cycle 009 activation is fully closed; consult `cycles/CYCLE-20260505-001.md` for the next live cycle-009 subtask.

## Open user decisions (resolution status, 2026-05-04)

D1'-D4' were delegated to the agent by user message "D1-D4 你自己决策吧，有问题我们商讨" and locked in `decisions/DEC-20260504-003-cycle-009-launch.md`. Summary:

```text
1. Cycle 009 ordering (D1')        -> parallel (Composer + Critic; cross-
                                       spec contract v1 is the test path).
2. Composer card source (D2')      -> paper-derived only; KYKT-job-derived
                                       deferred to cycle 010.
3. TEACHER_AUDIENCE_PROFILE (D3')  -> unpopulated, deferred. The form
                                       requires real-teacher facts
                                       (Research Taste / Hard Constraints
                                       / Demo Precedent) the agent has no
                                       source for; agent will not invent.
                                       Cycle 009 case cards are paper-
                                       derived under D2', so they do not
                                       depend on this form. To be filled
                                       by the user before any teacher-
                                       facing demo step.
4. Cycle 009 launch authorization  -> go. CASE-20260505-CRITIC-01 is the
   (D4')                             first card per cycle 008 D1.
```

Items still blocked on user approval per `AGENT_MASTER_PROMPT.md` section 6 (unchanged from prior cycles): final thesis selection, reproduction, training, checkpoint download, KYKT navigation change, frontend implementation, reusable Codex skill packaging, retiring any non-finalist track, declaring teacher-demo readiness, **populating `paradigm/TEACHER_AUDIENCE_PROFILE.md` (D3' deferred item)**.

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
