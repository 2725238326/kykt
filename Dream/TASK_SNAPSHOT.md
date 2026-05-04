# Dream Task Snapshot

Last updated: 2026-05-04 (cycle 010 activation in flight: user authorized v2 contract upgrade + cycle 010 parallel ordering + agent-recommended D3 continued deferral; cycle 009 closed; cycle 010 case-card portfolio for Memory + Permanence to be drafted next)

Status: **in_progress** (cycle 010 activation pass running across 4 sub-passes A..D; case-card drafting S2..S7 follows)

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
task_id:    cycle-010-activation
phase:      Phase 1.5 (Research Workflow Deployment)
cycle:      010 (case-card filling: Memory + Permanence parallel; v2 cost-typed route_regret active)
status:     in_progress; sub-pass A done (snapshot locked); sub-pass B/C/D following
```

One-line description:

```text
Cycle 009 closed with 6 paper-derived case cards (3 Critic + 3 Composer) +
cross-spec contract usage audit + 1 v1 -> v2 cost-typed route_regret
candidate (CASE-COMPOSER-03). User authorized cycle 010 launch:
(1) v2 contract upgrade -> adopt; (2) cycle 010 ordering -> Memory +
Permanence in parallel (cross-pair pattern from cycle 009 reused);
(3) D3 (first demo target) -> agent recommends continued deferral until
cycle 010 closeout (M+P L2 evidence still missing). cycle 010 will
fill 6 case cards: CASE-20260504-MEMORY-01..03 (per SPEC-20260503-002
line 240) + CASE-20260504-PERMANENCE-01..03 (per SPEC-20260503-003
line 231). Activation pass A..D running this turn.
```

## Subtask board (latest pass: cycle 010 activation, 2026-05-04)

| ID | Subtask | Status | Canonical artifact |
| --- | --- | --- | --- |
| A1 | Lock cycle 010 activation in this snapshot (status in_progress; subtask board; current task; If-interrupted-resume-from) | done | this file |
| A2 | Write `decisions/DEC-20260504-004-cross-spec-contract-v2.md` recording v2 promotion + retained v1 prose | pending | `decisions/DEC-20260504-004-cross-spec-contract-v2.md` |
| A3 | Edit `paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md` to v2: bump version markers; append v2 change log; preserve v1 prose under "Superseded versions" per Discipline rule 5 | pending | `paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md` |
| A4 | Edit `cases/CASE-20260505-COMPOSER-03.md` to mark v2 (cost-adjusted) recommendation as canonical; v1 row preserved as superseded | pending | `cases/CASE-20260505-COMPOSER-03.md` |
| A5 | Write `decisions/DEC-20260504-005-cycle-010-launch.md` recording cycle 010 D-decisions (parallel ordering; D3 continued deferral with rationale; v2 active; D3' sub-field correction) | pending | `decisions/DEC-20260504-005-cycle-010-launch.md` |
| A6 | Write `cycles/CYCLE-20260504-002.md` cycle log (cycle 010 header + subtask board G1/S1..S7 for case-card drafting; references DEC-004 + DEC-005) | pending | `cycles/CYCLE-20260504-002.md` |
| A7 | Run Guidance File Sync Rule chain (RESEARCH_STATE / WORKFLOW_STATUS / INDEX / AGENT_MASTER_PROMPT / README) + final subtask board flip on this snapshot | pending | sync targets + this file |

After A1..A7 complete, the next live pass is cycle 010's own S1..S7 case-card drafting (each card = one anti-F-001 sub-pass), tracked in `cycles/CYCLE-20260504-002.md` not here.

## Last completed task pass

```text
pass_name:        Cycle 009 case-card portfolio + S6 audit + S7 closeout
                  pass (S2..S7 done within a single resumed conversation;
                  only S8 user-facing surfacing remains)
date:             2026-05-04
trigger:          User message "阅读task snapshot.md，继续推进任务"
                  resuming the cycle-009 activation pass after the
                  earlier 32 MB checkpoint and the Critic-side ID-drift
                  cleanup commit.
files_modified:   TASK_SNAPSHOT.md (this file; subtask board + last
                  completed task pass + If interrupted resume from)
                  cycles/CYCLE-20260505-001.md (subtask board flips for
                  S6/S7; new "Contract Usage Audit (S6)" section; new
                  "Closeout (S7)" section; header bump)
                  RESEARCH_STATE.md ("Cycle 009 Case-Card Filling
                  Closeout (CYCLE-20260505-001)" section appended at
                  end; header bump)
                  WORKFLOW_STATUS.md (header bump; Cycle logs row
                  pointer; Cross-spec contract row updated; Geometry
                  Critic + Composer rows updated to "L2 case cards
                  drafted, paper-derived"; Recommended Next User
                  Decision rewritten to cycle-010 launch + contract gap
                  + v1 -> v2 candidate)
                  INDEX.md (cases/ section listing the 6 case cards;
                  header bump)
                  AGENT_MASTER_PROMPT.md (header bump only; no protocol
                  change)
                  README.md (header bump only; no surface change)
new_artifacts:    cases/CASE-20260504-CRITIC-02.md
                  cases/CASE-20260504-CRITIC-03.md
                  cases/CASE-20260505-COMPOSER-01.md
                  cases/CASE-20260505-COMPOSER-02.md
                  cases/CASE-20260505-COMPOSER-03.md
                  (CASE-20260504-CRITIC-01.md was authored in the prior
                  pass; the five above complete the L2 portfolio under
                  D1' parallel + D2' paper-derived.)
discipline:       Surgical Edits + Honesty Override; CR-5 evidence-label
                  propagation enforced across all 6 cards; v1 -> v2
                  cost-typed route_regret candidate is recorded as
                  inferred and explicitly NOT smuggled into v1; contract
                  gaps (CR-2 zero coverage; CR-3 forward-reference
                  shape) recorded as gaps, not papered over; F-001
                  working rules honored (Edit over Write, narrow Reads,
                  no re-Read of in-context files).
budget_event:     None this pass; the prior-pass 32 MB event remains
                  the canonical F-001 record.

prior_pass_name:  Cycle 009 activation pass (S1..S5 + S6 partial)
prior_pass_date:  2026-05-04
prior_pass_files: TASK_SNAPSHOT.md, decisions/DEC-20260504-003-cycle-
                  009-launch.md, cycles/CYCLE-20260505-001.md,
                  cases/CASE-20260504-CRITIC-01.md
```

## If interrupted, resume from

If a new agent or new conversation is picking this up cold:

```text
1. Read this file (you are here).
2. Read decisions/DEC-20260504-003-cycle-009-launch.md (cycle 009
   D1'-D4' locks; cycle 009 authorization).
3. Read decisions/DEC-20260504-004-cross-spec-contract-v2.md (v2
   contract upgrade memo; written in sub-pass B of this activation
   pass; superseded v1 prose preserved per Discipline rule 5).
4. Read decisions/DEC-20260504-005-cycle-010-launch.md (cycle 010
   D-decisions: parallel ordering for Memory + Permanence; D3
   continued deferral with rationale; v2 active; D3'
   sub-field-empty-claim correction).
5. Read cycles/CYCLE-20260504-002.md (cycle 010 cycle log; carries
   case-card drafting subtask board G1/S1..S7 for the 6 cards
   CASE-20260504-MEMORY-01..03 + CASE-20260504-PERMANENCE-01..03).
6. Resume at the first non-`done` row in the activation subtask
   board above (A1..A7). Once A7 is `done` the activation pass is
   closed; switch to cycle 010's own S1..S7 case-card drafting,
   tracked in `cycles/CYCLE-20260504-002.md` not in this file.
7. Honor F-001 working rules throughout: do not Read large files
   already cited in this snapshot; prefer Grep -n + Edit over
   full-file Read + Write; cap large files in active context at <=2
   simultaneously.
```

If this snapshot's Status flips back to `idle`, the cycle 010 activation pass is closed and the next live cycle is cycle 010's own case-card drafting (cycle log is the authoritative artifact for that phase).

## Open user decisions (resolution status, 2026-05-04)

D1'-D4' were delegated to the agent by user message "D1-D4 你自己决策吧，有问题我们商讨" and locked in `decisions/DEC-20260504-003-cycle-009-launch.md`. Summary:

```text
1. Cycle 009 ordering (D1')        -> parallel (Composer + Critic; cross-
                                       spec contract v1 is the test path).
2. Composer card source (D2')      -> paper-derived only; KYKT-job-derived
                                       deferred to cycle 010.
3. TEACHER_AUDIENCE_PROFILE (D3')  -> populated 2026-05-04 by user input.
                                       Earlier snapshot text claimed
                                       three sub-fields remained empty;
                                       this was a stale read. The user
                                       2026-05-04 input populated all 6
                                       fields explicitly: Research Taste
                                       = 科研的训练 / 写作技巧 /
                                       创新范式 / 讲好故事; Prior
                                       Expectations = "老师不知道我们
                                       要做 Dream" -> cold start; Demo
                                       Precedent -> first impression
                                       (implied by Prior Expectations);
                                       Previously Praised = "老师对 3R
                                       方向都没啥具体贬褒" -> no specific
                                       3R praise; Previously Criticized
                                       = same statement -> no specific
                                       3R criticism; Hard Constraints =
                                       no constraints stated. Profile
                                       is fully populated; D3' resolved;
                                       D3 (first demo target choice)
                                       still deferred per DEC-20260504-
                                       002 because cycle 010 case-card
                                       data for Memory + Permanence is
                                       still pending (agent
                                       recommendation per DEC-20260504-
                                       005 cycle 010 launch memo).
4. Cycle 009 launch authorization  -> go. CASE-20260504-CRITIC-01 is the
   (D4')                             first card per cycle 008 D1; drafted
                                     2026-05-04, paper-derived per D2'.
```

Cycle 010 launch decisions (locked 2026-05-04 from user message "1 a 吧 / 2 并行是不是好点儿 / 3 我觉得你决定就好 / 4 这个啥意思" plus agent confirmation; recorded in `decisions/DEC-20260504-005-cycle-010-launch.md`):

```text
1. v2 cost-typed route_regret (E1)  -> adopt. Contract bumped to v2;
                                       capability_match adds
                                       cost_normalized axis; alpha = 0.5
                                       initial (inferred). v1 prose
                                       preserved as Superseded per
                                       Discipline rule 5. CASE-COMPOSER-
                                       03 v2 row promoted to canonical
                                       recommendation; v1 row preserved.
                                       Memo: DEC-20260504-004.
2. Cycle 010 ordering (E2)          -> Memory + Permanence in parallel.
                                       Cross-pair pattern from cycle 009
                                       (CRITIC-02 <-> COMPOSER-01) reused;
                                       CR-2 binding closed via in-cycle
                                       cross-pair (Permanence publishes
                                       suppress_static_write, Memory
                                       consumes; both drafted in cycle
                                       010). Forward-reference null
                                       protocol from CRITIC-03 covers
                                       any timing gap.
3. D3 first demo target (E3)        -> agent decision: continued deferral
                                       to cycle 010 closeout. Two
                                       deferral conditions per
                                       DEC-20260504-002 are now both met
                                       (audience profile populated +
                                       cycle 009 case-card data exists),
                                       but Memory + Permanence still have
                                       no L2 evidence; picking now would
                                       violate Discipline rule 5
                                       (Honesty Override) by selecting on
                                       partial portfolio coverage. Re-
                                       surface at cycle 010 closeout when
                                       all 4 finalists have L2 cards.
4. D3' sub-field correction (E4)    -> stale snapshot text saying "three
                                       sub-fields remain empty" was
                                       wrong; profile is fully populated
                                       (see corrected entry in section 3
                                       above). User asked "这个啥意思"
                                       triggered the correction.
```

Items still blocked on user approval per `AGENT_MASTER_PROMPT.md` section 6 (unchanged from prior cycles): final thesis selection, reproduction, training, checkpoint download, KYKT navigation change, frontend implementation, reusable Codex skill packaging, retiring any non-finalist track, declaring teacher-demo readiness. (D3' previously listed here is now resolved 2026-05-04; D3 demo-target choice remains deferred per DEC-20260504-002.)

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
