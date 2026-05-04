# Dream Task Snapshot

Last updated: 2026-05-04 (cycle 010 fully closed = 6 case cards drafted (3 Memory + 3 Permanence) + cycle 009 contract gaps G1 + G3 closed + CR-3 forward-reference null closed + 3 v2 -> v3 candidates surfaced + 3 new gaps recorded for cycle 011 + Guidance File Sync Rule chain run; only S8 user-facing surfacing remains, after which D3 first demo target becomes user-eligible for decision)

Status: **in_progress** (cycle 010 case-card portfolio + S6 audit + S7 closeout complete; only S8 user-facing report remains, after which this snapshot flips to idle and D3 first-demo-target re-surfacing happens in S8 message)

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
task_id:    cycle-010
phase:      Phase 1.5 (Research Workflow Deployment)
cycle:      010 (case-card filling: Memory + Permanence parallel; v2 cost-typed route_regret active)
status:     in_progress; cycle 010 case-card portfolio + S6 audit + S7 closeout done; S8 user-facing surfacing pending; D3 first-demo-target eligible for re-decision after S8
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

## Subtask board (latest pass: cycle 010 case-card filling, 2026-05-04)

| ID | Subtask | Status | Canonical artifact |
| --- | --- | --- | --- |
| (activation A1..A7) | All cycle 010 activation sub-passes (snapshot lock + v2 contract + DEC-004 + DEC-005 + cycle 010 cycle log + sync chain) | done | commits `8a004dc` / `abf9a1d` / `1c5f5b9` |
| S2 | Draft `cases/CASE-20260504-MEMORY-01.md` (primary Memory L2; MonST3R 48-frame; CR-3 producer of latent_drift_proxy that closes cycle 009 CRITIC-03 forward-reference null) | done | `cases/CASE-20260504-MEMORY-01.md` (commit `7a18304`) |
| S3 | Draft `cases/CASE-20260504-PERMANENCE-01.md` (primary Permanence L2; same KYKT job for in-cycle CR-2 cross-pair; producer of suppress_static_write that MEMORY-01 consumes; closes cycle 009 gap G1) | done | `cases/CASE-20260504-PERMANENCE-01.md` (commit `9329704`) |
| S4 | Draft `cases/CASE-20260504-MEMORY-02.md` + `cases/CASE-20260504-MEMORY-03.md` (Spann3R externalization-of-governance + MASt3R non-hallucination boundary) | done | `cases/CASE-20260504-MEMORY-02.md` + `03.md` (commit `4ad1667`) |
| S5 | Draft `cases/CASE-20260504-PERMANENCE-02.md` + `cases/CASE-20260504-PERMANENCE-03.md` (MASt3R static control + synthetic identity-validation; closes PERMANENCE-01 fail_fast b + c) | done | `cases/CASE-20260504-PERMANENCE-02.md` + `03.md` (commit `8172f79`) |
| S6 | Cross-spec contract usage audit under v2: CR-1..CR-6 instance map across cycle-010 cards + closure status of cycle-009 gaps G1 / G2 / G3 + v2 -> v3 candidate enumeration | done | `cycles/CYCLE-20260504-002.md` "Contract Usage Audit (S6) under v2" |
| S7 | Cycle 010 closeout: write closeout section in cycle log; sync `RESEARCH_STATE.md` / `WORKFLOW_STATUS.md` / `INDEX.md` / `AGENT_MASTER_PROMPT.md` / `README.md`; run Guidance File Sync Rule chain starting from this snapshot | done | `cycles/CYCLE-20260504-002.md` "Closeout (S7)" + sync targets |
| S8 | Surface cycle-010 outputs to user; re-surface D3 first-demo-target now that all 4 finalists have L2 coverage; ask for cycle-011 launch decisions OR D3 demo-target pick | in_progress | user-facing message (this turn) |

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
2. Read decisions/DEC-20260504-005-cycle-010-launch.md (cycle 010
   authorization + 4 D-decisions). DEC-20260504-003 / -004 are
   referenced from DEC-005; consult only if you need their full
   rationale.
3. Read cycles/CYCLE-20260504-002.md "Contract Usage Audit (S6) under
   v2" + "Closeout (S7)" sections — those carry the full cycle-010
   result summary (6 cards drafted, gaps G1 + G3 + cycle-009 CRITIC-03
   forward-reference null all closed, 3 v2 -> v3 candidates, 3 new
   gaps G4/G5/G6 for cycle 011).
4. Cycle 010 is closed at content level (S2..S7 done); only S8
   user-facing surfacing remains. After S8: D3 first-demo-target
   becomes user-eligible for decision (deferral conditions per
   DEC-20260504-002 are now FULLY satisfied — audience profile
   populated AND all 4 finalists have L2 case-card coverage).
5. The four cycle-011 launch decisions to surface in S8 are:
   (a) D3 first teacher demo target (now eligible);
   (b) G2 closure path (tau_spread upgrade requires L3 or KYKT-derived
       work, both have user-approval gates);
   (c) v2 -> v3 candidates 1-3 (8x8 partition / forward-reference
       null protocol / identity_consistency threshold);
   (d) G4 / G5 / G6 closure scope.
6. Honor F-001 working rules throughout: do not Read large files
   already cited in this snapshot; prefer Grep -n + Edit over
   full-file Read + Write; cap large files in active context at <=2
   simultaneously.
```

If this snapshot's Status flips back to `idle`, cycle 010 is fully closed and the next live phase is cycle 011 (gated on user decisions surfaced in S8).

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
