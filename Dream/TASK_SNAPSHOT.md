# Dream Task Snapshot

Last updated: 2026-05-05 (cycle 011 launched + closed at content level: Critic storyboard draft + G4 closed-by-documentation under v2.1 + G5 closed by formalization + v2.1 contract revision active + DEC-20260505-001 covers (1) D3 = Critic, (2) cycle 011 scope, (3) v2.1 forward-reference null protocol; only S8 user-facing surfacing remains)

Status: **in_progress** (cycle 011 S1..S7 done; S8 user-facing report remains; after S8 this snapshot flips to idle)

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
task_id:    cycle-011
phase:      Phase 1.5 (Research Workflow Deployment)
cycle:      011 (Critic demo storyboard draft + G4/G5 closure + v2.1 forward-reference null protocol formalization)
status:     in_progress; cycle 011 S1..S7 done; S8 user-facing surfacing pending
```

One-line description:

```text
User delegated cycle-011 launch decisions (1)(2)(3) to the agent on
2026-05-05 ("你给我决定吧，（1）（2）（3）"). Agent locked: (1) D3 first
teacher demo target = Geometry Critic; (2) cycle 011 scope = G4 + G5
closure (primary) + Critic demo storyboard draft (secondary); (3) v2 ->
v2.1 additive contract revision (forward-reference null protocol
formalization). G6 + G2 + KYKT-derived Composer card + L3 prototype +
paper writing all explicitly deferred. Cycle 011 produced 3 new
artifacts (DEC-001 + cycle log + storyboard) + 1 contract revision
(v2.1 additive) + 1 in-place case-card edit (PERMANENCE-03 G4 closure).
Storyboard status: draft only; no showing authorization. S8
user-facing surfacing remains.
```

## Subtask board (latest pass: cycle 011 launch + closeout, 2026-05-05)

| ID | Subtask | Status | Canonical artifact |
| --- | --- | --- | --- |
| A | Activation pass: write `decisions/DEC-20260505-001-cycle-011-launch-and-d3-demo-target.md` + `cycles/CYCLE-20260505-002.md` + checkpoint commit | done | commit `da7b3b1` |
| S1 | Skim `templates/demo_storyboard.md` + cycle-009 CRITIC-01..03 to confirm storyboard panel candidates (CRITIC-02 Fast3R-vs-Spann3R locked as Panel A primary) | done | (in-context only) |
| S2 | Draft `storyboards/STORY-20260505-001-critic.md` (status: draft; functional vs placeholder labeled; acceptance-for-showing left unchecked) | done | `storyboards/STORY-20260505-001-critic.md` (commit `901f62c`) |
| S3 | G4 closure: in-place edit on `cases/CASE-20260504-PERMANENCE-03.md` documenting CR-2 consumer-side forward-reference null under v2.1 protocol | done | PERMANENCE-03 CR-2 entry (commit `901f62c`) |
| S4 | v2 -> v2.1 additive contract revision: "Forward-reference null protocol" subsection + v2.1 Change Log entry on `paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md` | done | CROSS_SPEC_SIGNAL_CONTRACT.md (commit `901f62c`) |
| S5 | Update DEC-20260505-001 if any cycle-011 deviation needs Honesty Override recording | done (no deviation) | DEC body unchanged; G4 chose form-1 which DEC-001 (2) listed as one of the budget-friendly options |
| S6 | Cycle 011 contract usage audit under v2.1: confirm v2.1 protocol exercise; G4 + G5 closure status; v2 substance unchanged; storyboard claims labeled | done | `cycles/CYCLE-20260505-002.md` "Contract Usage Audit (S6) under v2.1" |
| S7 | Cycle 011 closeout: write closeout section in cycle log; sync chain (this snapshot first; then WORKFLOW_STATUS / RESEARCH_STATE / INDEX / AGENT_MASTER_PROMPT / README) | in_progress | `cycles/CYCLE-20260505-002.md` "Closeout (S7)" + this sync pass |
| S8 | Surface cycle-011 outputs to user; ask for cycle-012 launch decisions (storyboard reviewer pass; cycle 012 scope; demo show authorization gate) | pending | user-facing message (next turn) |

## Last completed task pass

```text
pass_name:        Cycle 011 launch + closeout pass (A + S1..S7 done in
                  one resumed conversation; S8 user-facing surfacing
                  remains)
date:             2026-05-05
trigger:          User message "你给我决定吧，（1）（2）（3）" delegating
                  the three cycle-011 launch decisions surfaced in
                  cycle 010 S8 to the agent.
files_modified:   TASK_SNAPSHOT.md (this file; subtask board + last
                  completed task pass + If interrupted resume from +
                  Open user decisions)
                  cycles/CYCLE-20260505-002.md (S6 audit + S7 closeout
                  + subtask board flips + header bump)
                  cases/CASE-20260504-PERMANENCE-03.md (G4 closure CR-2
                  consumer-side under v2.1 protocol; header bump)
                  paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md (v2 -> v2.1
                  additive: "Forward-reference null protocol" subsection
                  + "v2.1 Change Log" entry; header bump)
                  RESEARCH_STATE.md (cycle 011 closeout section
                  appended; header bump) [pending in this sync pass]
                  WORKFLOW_STATUS.md (header bump; Cycle logs row
                  pointer; Cross-spec contract row -> v2.1 active;
                  Recommended Next User Decision rewritten to cycle
                  012 launch packet + storyboard reviewer pass + demo
                  show authorization gate) [pending in this sync pass]
                  INDEX.md (storyboards/ section new; header bump)
                  [pending in this sync pass]
                  AGENT_MASTER_PROMPT.md (header bump only; mandatory-
                  load list grows by storyboards/ if needed)
                  [pending in this sync pass]
                  README.md (header bump only; storyboards/ subdir
                  noted in Directory Map) [pending in this sync pass]
new_artifacts:    decisions/DEC-20260505-001-cycle-011-launch-and-d3-
                  demo-target.md
                  cycles/CYCLE-20260505-002.md
                  storyboards/STORY-20260505-001-critic.md
discipline:       Surgical Edits (no SPEC body changes; no retro-
                  renumber; PERMANENCE-03 G4 closure traces directly to
                  DEC-001 (2); v2.1 revision additive only) +
                  Honesty Override (storyboard `draft` only; all panels
                  labeled placeholder; no measured claims; no
                  "approved-for-showing" status; v2 substance audit
                  shows zero changes).
budget_event:     None this pass.

prior_pass_name:  Cycle 010 case-card portfolio + S6 audit + S7
                  closeout + S8 user-facing report pass
prior_pass_date:  2026-05-04
prior_pass_files: TASK_SNAPSHOT.md, cycles/CYCLE-20260504-002.md, six
                  case cards under cases/, RESEARCH_STATE.md,
                  WORKFLOW_STATUS.md, INDEX.md
```

## If interrupted, resume from

If a new agent or new conversation is picking this up cold:

```text
1. Read this file (you are here).
2. Read decisions/DEC-20260505-001-cycle-011-launch-and-d3-demo-
   target.md (cycle 011 launch + D3 = Critic + cycle 011 scope +
   v2.1 forward-reference null protocol formalization).
   DEC-20260504-005 + -004 + -002 are referenced from DEC-001;
   consult only if you need their full rationale.
3. Read cycles/CYCLE-20260505-002.md "Contract Usage Audit (S6)
   under v2.1" + "Closeout (S7)" sections — those carry the full
   cycle-011 result summary (3 new artifacts + v2.1 active + G4
   closed-by-documentation + G5 closed by formalization +
   G2/G6 still deferred + cycle 012 launch packet).
4. Cycle 011 is closed at content level (S1..S7 done); only S8
   user-facing surfacing remains. After S8: cycle 012 launch
   decisions become user-facing (storyboard reviewer pass; cycle 012
   scope; demo show authorization gate; G6/G2 closure paths gated).
5. The four cycle-012 launch decisions to surface in S8 are listed
   in cycles/CYCLE-20260505-002.md "Closeout (S7)". Copy them into
   the S8 user message verbatim with one-line agent reading per
   decision.
6. Storyboard storyboards/STORY-20260505-001-critic.md is `draft`
   ONLY. Do NOT promote to `approved-for-showing` without a
   separate DEC. Do NOT start a Gemini CLI frontend handoff for
   storyboard rendering without user approval per
   AGENT_MASTER_PROMPT.md section 6.
7. Honor F-001 working rules throughout: do not Read large files
   already cited in this snapshot; prefer Grep -n + Edit over
   full-file Read + Write; cap large files in active context at <=2
   simultaneously.
```

If this snapshot's Status flips back to `idle`, cycle 011 is fully closed and the next live phase is cycle 012 (gated on user decisions surfaced in S8).

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

Cycle 011 launch decisions (locked 2026-05-05 from user message "你给我决定吧，（1）（2）（3）" delegating to agent; recorded in `decisions/DEC-20260505-001-cycle-011-launch-and-d3-demo-target.md`):

```text
1. D3 first teacher demo target (1)  -> Critic (Geometry Critic /
                                        System-2 3R, SPEC-20260503-001).
                                        Picked on five-axis comparison
                                        (surprise hook strength,
                                        mechanism legibility for
                                        cold-start audience, connection
                                        to Dream3R thesis, L2 portfolio
                                        depth, demo failure-mode
                                        robustness, "looks like paper X"
                                        collapse risk). Storyboard:
                                        STORY-20260505-001-critic.md
                                        drafted; status = draft only;
                                        no `approved-for-showing`
                                        granted by DEC-001. D3 = "first
                                        demo target", not retiring of
                                        other finalists; DEC-20260504-
                                        002 still in force.
2. Cycle 011 scope (2)               -> G4 (CR-2 partial on synthetic
                                        identity-validation clip) +
                                        G5 (forward-reference null
                                        protocol formalization) closure
                                        primary; Critic demo storyboard
                                        draft secondary. G6 + G2 +
                                        KYKT-derived Composer card +
                                        L3 prototype + paper writing
                                        all explicitly deferred.
3. v2 -> v3 candidates (3)           -> v2 -> v2.1 additive revision:
                                        forward-reference null protocol
                                        formalized as a contract-pinned
                                        subsection. The other two
                                        candidates (8x8 grid partition
                                        for Permanence regions;
                                        identity_consistency threshold
                                        pinning at ~0.7) deferred and
                                        not promoted; rationale per
                                        DEC-001 (3): grid partition is
                                        Permanence implementation
                                        detail, threshold pinning needs
                                        measured anchors that don't yet
                                        exist.
4. New (4) blocked items unchanged from cycle 010 closeout (final
   thesis selection / reproduction / training / checkpoint download /
   KYKT navigation change / Codex frontend implementation / reusable
   Codex skill packaging / discarding any non-finalist track /
   declaring teacher-demo readiness; demo SHOWING also blocked).
```

Items still blocked on user approval per `AGENT_MASTER_PROMPT.md` section 6 (unchanged from prior cycles): final thesis selection, reproduction, training, checkpoint download, KYKT navigation change, frontend implementation, reusable Codex skill packaging, retiring any non-finalist track, declaring teacher-demo readiness, **showing the Critic demo (storyboard remains `draft`; promotion to `approved-for-showing` requires a separate DEC)**.

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
