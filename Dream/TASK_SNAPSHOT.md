# Dream Task Snapshot

Last updated: 2026-05-07 (CYCLE 020 CLOSED: combined v0.2 code structure planning + implementation roadmap done; S1-S4 closed across 2026-05-06 launch session + 2026-05-07 closure session; DEC-20260506-004 written 544 lines + planning/DREAM3R_V02_CODE_STRUCTURE.md NEW 1086 lines + planning/DREAM3R_V02_IMPLEMENTATION_ROADMAP.md NEW 1226 lines + cycles/CYCLE-20260506-005.md NEW; full sync chain complete; user-revised post-019 trajectory documented honestly with old-vs-new table per Discipline rule 5 — DEC-003 row preserved unchanged; renumbered post-020 candidates 021-025 documented in CYCLE-20260506-005 §"Cycle 020 outcome (closed)")

Status: **idle** (cycle 020 closed; no active task; pending user direction on renumbered next-cycle candidates 021-025 in CYCLE-20260506-005 §"Cycle 020 outcome (closed) and renumbered post-020 trajectory" — comparator map v0.2 addendum / paper Phase 2 rewrite for A+D / cycle 015 G_run resumption / capability_match measurement pass / first ablation execution authorization)

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
task_id:    cycle-020
phase:      Architecture-first mainline; combined v0.2 code structure planning + implementation roadmap (markdown only; no training, no GPU, no checkpoint creation, no code touch) — CLOSED
cycle:      020 (Dream3R v0.2 code structure + implementation roadmap; per DEC-20260506-004; user-revised trajectory: skip original cycle-020 SPEC-003 v0.2 comparator map; combine original cycles 021+022 into one cycle 020 under "高强度推进")
status:     done (S1-S4 closed across 2026-05-06 launch session and 2026-05-07 closure session; DEC-20260506-004 written 544 lines + DREAM3R_V02_CODE_STRUCTURE.md NEW 1086 lines + DREAM3R_V02_IMPLEMENTATION_ROADMAP.md NEW 1226 lines + cycles/CYCLE-20260506-005.md NEW; full sync chain complete; user-revised post-019 trajectory documented honestly per Discipline rule 5; renumbered post-020 candidates 021-025 documented in CYCLE-20260506-005)
```

One-line description:

```text
Per DEC-20260506-004 (cycle 020 launch + combined code
structure + implementation roadmap), cycle 020 produces two
sibling planning artifacts that together form the load-bearing
other-agent review/modification handoff surface per user
2026-05-06 instruction "其他agent审阅修改 + 你文档更新清楚哈".

Trajectory revision (documented honestly per Honesty Override):
DEC-20260506-003 §"Post-019 trajectory" enumerated cycles 020-025
with original cycle 020 = SPEC-003 v0.2 comparator map addendum.
User 2026-05-06 message "行，你帮我吧，我们需要高强度推进"
revised the trajectory: skip original cycle 020 (deferred to
renumbered cycle 021); combine original cycles 021 (code
structure) + 022 (implementation roadmap) into a single cycle
020 with two artifacts. Both DEC-003 trajectory row and DEC-004
revision row preserved in decision_registry per Discipline rule 5
(no silent supersede).

S1 done: DEC-20260506-004 written
(decisions/DEC-20260506-004-cycle-020-launch-code-structure-and-
implementation-roadmap.md; 544 lines). Carries: scope lock
(combined planning artifacts) + trajectory revision narrative
(original 6-cycle queue renumbered with explicit
old-vs-new table) + allowed/not-allowed enumerations + S2
section structure preview + S3 section structure preview +
review surface guidance.

S2 done: planning/DREAM3R_V02_CODE_STRUCTURE.md written
2026-05-06 (1086 lines). Maps SPEC-004 v0.2 architecture
Delta 1..6 to per-file changes in code/dream3r/. Sections:
existing v0.1 substrate per-file summary (cited from
code/dream3r/PLAN.md; not re-derived); v0.2 delta -> module
mapping table; per-file change manifest with MODIFIED
(modules.py / model.py / losses.py / smoke_test.py / config.py
/ train.py) + NEW (memory_anchor_bank.py / nsa_attention.py /
composer_experts/ subdirectory with 7 expert adapters +
latency_bench.py + datasets per ABL-v02 needs) + STABLE
(bus.py CR-1..CR-6 carry forward; existing token taxonomy
stable). Each MODIFIED + NEW file has explicit review surface
subsection per user request "其他agent审阅修改". Server-side
deployment surface documented per F-002. Honest evidence
labels: speculative for NSA-related; paper-derived for
DINOv3-S; engineering-judgment for expert pool composition;
inferred for line-of-code estimates.

S3 done: planning/DREAM3R_V02_IMPLEMENTATION_ROADMAP.md
written 2026-05-07 (1226 lines). Breaks S2 code structure
into 22 reviewable tasks. Task taxonomy:
  Tier 1 prerequisites (must run first):
    T-v02-A repo skeleton + config schema scaffold
    T-v02-B test harness scaffold (pytest + smoke targets)
  Tier 2 per-module:
    T-v02-C1 DINOv3-S backbone + heads from scratch
    T-v02-C2-mem-bank bounded anchor bank (NEW file)
    T-v02-C2-nsa NSA-style selective retrieval (NEW file)
    T-v02-C5 Composer routing skeleton (modifies modules.py)
    T-v02-EXPERT-1..7 (per-expert adapter; one task per
      admitted lightweight model: MASt3R / Fast3R / Spann3R
      / CUT3R / MoGe-2 / DepthAnything-V2 / Test3R)
  Tier 3 integration:
    T-v02-D model.py wiring + bus tick re-validation
    T-v02-E losses.py L_total update for v0.2 signals
    T-v02-F smoke_test.py + latency_bench.py
  Tier 4 ABL harnesses:
    T-v02-ABL-1..9 (one per ABL-v02-N from SPEC-005;
    each task includes execution gate = separate DEC +
    per-step micro gates F-002)
Each task carries: scope (which files; what changes); inputs
(existing files + spec sections + v0.1 PLAN.md checkboxes
consumed); outputs (modified + new files + tests + logs);
estimated effort (engineering-hours + lines-of-code; both
inferred); pre-execution review checklist (handed to reviewer
BEFORE implementation); post-execution validation checklist
(handed to reviewer AFTER implementation; verifies scope +
tests + no regression); execution gate. Task dependency graph
is a DAG (visualized as ASCII). Total inferred effort:
~250-380 engineering-hours across all 22 tasks.

S4 IN PROGRESS: TASK_SNAPSHOT mid-pass anchor done (this edit,
FIRST in sync chain per F-001 rule 6); decision_registry
append DEC-004 row + WORKFLOW_STATUS / RESEARCH_STATE / INDEX
light sync (line 3 + 2 new planning rows + latest cycle log
pointer) + cycles/CYCLE-20260506-005.md NEW + final
TASK_SNAPSHOT flip to idle remaining in this session.

No training authorized. No GPU runs. No code touch. No
clones / installs / checkpoints. Markdown only.
```

## Subtask board (active pass: cycle 020 combined code structure planning + implementation roadmap; spans 2026-05-06 launch -> 2026-05-07 closure)

| ID | Subtask | Status | Canonical artifact |
| --- | --- | --- | --- |
| S1 | Write DEC-20260506-004 (cycle 020 launch + combined code structure planning + implementation roadmap scope lock; trajectory revision narrative honestly noting deferral of original cycle 020 SPEC-003 v0.2 comparator map; explicit allowed/not-allowed enumerations; review surface guidance) | done | `decisions/DEC-20260506-004-cycle-020-launch-code-structure-and-implementation-roadmap.md` (544 lines) |
| S2 | Write planning/DREAM3R_V02_CODE_STRUCTURE.md (NEW; maps SPEC-004 v0.2 Delta 1..6 to per-file changes in code/dream3r/; per-file change manifest with MODIFIED + NEW + STABLE; explicit review surface subsection per MODIFIED + NEW file per user request "其他agent审阅修改"; server-side deployment surface per F-002; honest evidence labels per change) | done | `planning/DREAM3R_V02_CODE_STRUCTURE.md` (1086 lines) |
| S3 | Write planning/DREAM3R_V02_IMPLEMENTATION_ROADMAP.md (NEW; breaks S2 code structure into 22 reviewable tasks T-v02-A/B/C1/C2-mem-bank/C2-nsa/C5/D/E/F + EXPERT-1..7 + ABL-1..9; per-task scope/inputs/outputs/effort/pre-execution review checklist/post-execution validation checklist/execution gate; task dependency DAG; recommended execution order; ~250-380 engineering-hours total inferred) | done | `planning/DREAM3R_V02_IMPLEMENTATION_ROADMAP.md` (1226 lines) |
| S4 | Sync chain: TASK_SNAPSHOT first per F-001 rule 6 (mid-pass anchor + final flip); decision_registry append DEC-004 row; WORKFLOW_STATUS / RESEARCH_STATE / INDEX line-3 sync + INDEX 2 new planning rows + INDEX latest cycle log pointer to CYCLE-20260506-005; cycles/CYCLE-20260506-005.md NEW cycle log; final TASK_SNAPSHOT flip to idle | done (all 5 sync items closed across 2026-05-06 + 2026-05-07 sessions) | `cycles/CYCLE-20260506-005.md` + sync chain |

S2 deliverable summary (for resume context):

```text
specs/SPEC-20260506-001-dream3r-architecture.md sections:
  - Identity / Approval (per DEC-20260506-001; candidate per DEC-501-004)
  - Scope of v0.1 (markdown only; no training)
  - The architectural claim (control-graph-as-architecture; what is novel
    vs what is carried from existing work)
  - Substrate hypothesis (hybrid transformer + SSM/Mamba + slot + bus;
    per-substrate justification + lineage + evidence labels)
  - Top-level architecture (ASCII block diagram)
  - Tokens (T1 frame / T2 pointmap / T3 evidence / T4 regime / T5 anchor +
    object slot / T6 bus)
  - State ownership (which module owns which state; cross-module reads
    are read-only)
  - Computational cores (C1 Perceiver transformer / C2 Memory SSM / C3
    Permanence slot / C4 Critic small head / C5 Composer table join /
    C6 Memory Bus)
  - The Memory Bus as runtime API (publish + read + handoff + gates)
  - A1-A8 mapping to concrete layers (A7 / A8 are RESERVED hooks; not
    designed in v0.1)
  - Read-write protocol per window (8-step bus tick)
  - Conflict resolution as architectural elements (CR-1..CR-6 as gates /
    masks / invariants / ledgers)
  - Training-objective sketch (NOT a training authorization; INPUT to S3
    ablation plan)
  - Module-level evidence labels (aggregate distribution: ~5 paper-proven
    / ~5 paper-derived / ~10 inferred / ~7 architecture-novel / 2
    speculative)
  - Comparator quick map (lightweight; full at S4)
  - Per-spec cross-reference (each finalist spec re-mapped to architecture)
  - Risks (R1 substrate falsification / R2 bus-as-novelty collapse / R3
    state-ownership invariant under training / R4 A7/A8 reserved hook
    conflict / R5 storytelling-vs-measurement asymmetry / R6 KYKT scope
    creep / R7 paper writing risk)
  - Boundaries (13 explicit boundaries; all carried from prior cycles +
    DEC-20260506-001)
  - Linked artifacts (decisions, finalist specs as INPUTS, paradigm
    artifacts, planning artifacts, failure modes, KYKT artifacts as
    future evidence anchors, storyboards remain `draft`, paper Phase 2
    now SUPPORT)
  - Next step (cycle 016 S3 ablation plan)
  - Open questions for next session (Q1 substrate ablation priority / Q2
    bus falsifying experiment design / Q3 A7/A8 concretization timing /
    Q4 v0.1 -> v0.2 trigger / Q5 KYKT integration for evidence anchoring
    / Q6 paper integration path)
  - Discipline notes
  - Version history
```

## Last completed task pass

```text
pass_name:        Cycle 020 mid-pass anchor (S1-S3 done across
                  2026-05-06 launch + 2026-05-07 closure session;
                  v0.2 code structure + implementation roadmap
                  combined planning artifacts written; user-
                  revised post-019 trajectory documented; S4
                  sync chain mid-pass — registry append + light
                  sync + cycle log + final flip remaining)
date:             2026-05-07 (cycle launched 2026-05-06; this
                  S4 sync chain mid-pass anchor edit on
                  2026-05-07)
trigger:          User authorization "行，你帮我吧，我们需要高强度
                  推进" (2026-05-06) approving option B from
                  agent's prioritized post-cycle-019 next-cycle
                  candidate list (skip original cycle 020 SPEC-
                  003 v0.2 comparator map; jump to code structure
                  + implementation roadmap as the cycles where
                  "其他agent审阅修改" lands) + "高强度推进"
                  intensity instruction justifying combined cycle
                  compression of original cycles 021 + 022.
                  Followed by 2026-05-07 user message "阅读
                  E:\kykt\Dream\TASK_SNAPSHOT.md，继续推进工作"
                  resuming the in-flight S4 sync chain.
files_modified:   TASK_SNAPSHOT.md (this file; header + Status +
                  Current task block + Subtask board + Last
                  completed task pass + If interrupted resume
                  from; FIRST in sync chain per F-001 rule 6 —
                  both mid-pass anchor and final flip)
                  registry/decision_registry.md (DEC-20260506-004
                  row appended; pending in this pass)
                  WORKFLOW_STATUS.md (Last updated; line 3
                  re-stamped; pending in this pass)
                  RESEARCH_STATE.md (Last updated; line 3
                  re-stamped; pending in this pass)
                  INDEX.md (Last updated + planning/ table 2 new
                  rows added + latest cycle log pointer updated
                  to CYCLE-20260506-005; pending in this pass)
new_artifacts:    decisions/DEC-20260506-004-cycle-020-launch-
                  code-structure-and-implementation-roadmap.md
                  (NEW; 544 lines; combined cycle scope lock +
                  trajectory revision narrative + per-deliverable
                  section structure preview + review surface
                  guidance)
                  planning/DREAM3R_V02_CODE_STRUCTURE.md (NEW;
                  1086 lines; v0.2 Delta 1..6 -> per-file change
                  manifest with MODIFIED + NEW + STABLE; explicit
                  review surface subsection per MODIFIED + NEW
                  file per user request "其他agent审阅修改";
                  server-side deployment surface per F-002;
                  honest evidence labels per change)
                  planning/DREAM3R_V02_IMPLEMENTATION_ROADMAP.md
                  (NEW; 1226 lines; 22 reviewable tasks in 4
                  tiers — Tier 1 prerequisites T-v02-A/B; Tier 2
                  per-module T-v02-C1/C2-mem-bank/C2-nsa/C5/
                  EXPERT-1..7; Tier 3 integration T-v02-D/E/F;
                  Tier 4 ABL harnesses T-v02-ABL-1..9; per-task
                  scope/inputs/outputs/effort/pre + post execution
                  review checklists/execution gate; task DAG;
                  ~250-380 engineering-hours total inferred)
                  cycles/CYCLE-20260506-005.md (NEW; cycle 020
                  log; pending in this pass)
in_place_edits:   sync chain only; SPEC-004 / SPEC-005 / SPEC-001
                  / SPEC-002 / SPEC-003 bodies NOT modified;
                  code/dream3r/PLAN.md NOT modified (v0.1 user-
                  authored roadmap preserved unchanged); code/
                  dream3r/*.py source files NOT modified (no code
                  touch — markdown planning only)
discipline:       Surgical Edits (cycle 020 produces 2 NEW
                  planning files + 1 NEW DEC + 1 NEW cycle log;
                  no in-place rewrites of any prior SPEC / DEC /
                  cycle log; pre-existing markdown lint warnings
                  on TASK_SNAPSHOT historical "If interrupted,
                  resume from" block lines 460-524 NOT fixed
                  per Surgical Edits rule 3 — these predate
                  cycle 020 and are out of scope; carried
                  pre-existing lint warnings on SPEC-001 line
                  593 + SPEC-002 line 633 falsification table +
                  cycle log tables + WORKFLOW_STATUS / INDEX
                  spec-row tables also NOT fixed) + Honesty
                  Override (every per-file change in CODE_
                  STRUCTURE carries inline evidence label;
                  every per-task entry in IMPLEMENTATION_
                  ROADMAP carries effort + evidence label;
                  trajectory revision documented honestly with
                  old-vs-new table in DEC-004 §"Why this
                  matters" — original DEC-003 trajectory NOT
                  silently superseded; effort estimates labeled
                  inferred and flagged as needing post-first-
                  task calibration).
budget_event:     None this pass. F-001 rule 1 honored: SPEC-001
                  / SPEC-002 / SPEC-003 v0.1 NOT re-Read; SPEC-004
                  / SPEC-005 v0.2 NOT re-Read (already in
                  context from cycle 018 + 019); cited by section
                  + line anchor only. code/dream3r/PLAN.md cited
                  via INDEX summary; bus.py / modules.py /
                  model.py / losses.py / smoke_test.py NOT Read
                  in cycle 020 (per-file summary cited from
                  PLAN.md INDEX block instead). F-002 carried:
                  markdown-only; server-side /hdd3/kykt26/
                  untouched.

prior_pass_name:  Cycle 019 close pass (S1-S3 done in single
                  session; v0.2 ablation plan addendum locked +
                  documented + synced; SPEC-20260506-005 written
                  991 lines; bundled DEC-20260506-002 stale-
                  status backfill)
prior_pass_date:  2026-05-06
prior_pass_files: TASK_SNAPSHOT.md, registry/decision_registry.md
                  (DEC-20260506-003 append + DEC-002 stale-status
                  backfill), specs/SPEC-20260506-002-dream3r-
                  ablation-plan.md (Version history tail v0.2
                  pointer entry), WORKFLOW_STATUS.md,
                  RESEARCH_STATE.md, INDEX.md, + new artifacts
                  decisions/DEC-20260506-003-cycle-019-launch-
                  ablation-plan-v02-addendum.md (~360 lines),
                  specs/SPEC-20260506-005-dream3r-ablation-plan-
                  v02.md (991 lines; nine ABL-v02; per-ABL
                  review checklist), cycles/CYCLE-20260506-004.md
                  (cycle 019 log)
```

## If interrupted, resume from

If a new agent or new conversation is picking this up cold:

```text
1. Read this file (you are here).

2. Read decisions/DEC-20260506-004-cycle-020-launch-code-structure
   -and-implementation-roadmap.md FIRST — this is the most recent
   strategic decision and it locks (a) cycle 020 combined scope
   (code structure + implementation roadmap as 2 artifacts;
   markdown only; no code touch) and (b) the user-revised post-019
   trajectory: original cycle 020 (SPEC-003 v0.2 comparator map
   addendum) deferred to renumbered cycle 021; cycles 021+022
   (code structure + roadmap) compressed into single cycle 020
   under "高强度推进".

3. Read decisions/DEC-20260506-003-cycle-019-launch-ablation-plan
   -v02-addendum.md and decisions/DEC-20260506-002-cycle-018-
   launch-v02-architecture.md and decisions/DEC-20260506-001-
   mainline-architecture-first.md (parent decisions; v0.2
   architecture + ablation deltas + architecture-first mainline).

4. Read these cycle 020 deliverables (already done in this pass):
   - planning/DREAM3R_V02_CODE_STRUCTURE.md (S2; 1086 lines;
     Delta 1..6 -> per-file change manifest; review surface per
     MODIFIED + NEW file)
   - planning/DREAM3R_V02_IMPLEMENTATION_ROADMAP.md (S3; 1226
     lines; 22 reviewable tasks T-v02-A/B/C1/C2-mem-bank/C2-nsa/
     C5/D/E/F + EXPERT-1..7 + ABL-1..9; per-task pre + post
     execution review checklists; task DAG; ~250-380 engineering-
     hours total)
   - cycles/CYCLE-20260506-005.md (S4; cycle 020 log; CLOSED
     once final TASK_SNAPSHOT flip done)

5. Reference SPEC bodies ONLY when needed (do NOT re-Read full
   files; cite by section + line anchor):
   - specs/SPEC-20260506-004-dream3r-architecture-v02.md (v0.2
     architecture; six deltas; INPUT for code structure)
   - specs/SPEC-20260506-005-dream3r-ablation-plan-v02.md (v0.2
     ablation plan; nine ABL-v02; INPUT for Tier 4 ABL harness
     tasks)
   - specs/SPEC-20260506-001-dream3r-architecture.md (v0.1; the
     substrate that v0.2 deltas modify; very large, ~1821 lines,
     do NOT re-Read)
   - code/dream3r/PLAN.md (v0.1 user-authored implementation
     roadmap; preserved unchanged; INDEX summary cited from
     INDEX.md without re-Reading the source files bus.py /
     modules.py / model.py / losses.py / smoke_test.py)

6. Read C:\Users\27252\.claude\projects\e--kykt\memory\
   feedback_dream_mainline_architecture_first.md and
   feedback_kykt_server_topology.md (cross-session memories;
   F-002 anchor — KYKT 3R model code runs server-side at
   /hdd3/kykt26/; local Windows is markdown + orchestration
   only).

Project state at this snapshot:
   Cycle 015 PAUSED at S9 done (NOT closed; NOT abandoned;
                                infrastructure reusable as future
                                Critic A4 evidence anchor).
   Cycle 016 DONE.
   Cycle 017 DONE (paper draft v1; needs v0.2 update later).
   Cycle 018 DONE (v0.2 architecture deltas; SPEC-20260506-004
                   written 741 lines; six numbered deltas; main-
                   claim narrowed to A+D).
   Cycle 019 DONE (v0.2 ablation plan addendum; SPEC-20260506-
                   005 written 991 lines; 9 ABL-v02; per-ABL
                   review checklist for other-agent handoff).
   Cycle 020 IN PROGRESS (S1-S3 done; S4 sync chain mid-pass):
                S1 done (DEC-20260506-004 written 544 lines;
                   cycle 020 launch + combined scope + trajectory
                   revision)
                S2 done (planning/DREAM3R_V02_CODE_STRUCTURE.md
                   written 1086 lines; per-file change manifest
                   + per-file review surface)
                S3 done (planning/DREAM3R_V02_IMPLEMENTATION_
                   ROADMAP.md written 1226 lines; 22 tasks with
                   per-task pre + post execution checklists)
                S4 in progress (TASK_SNAPSHOT mid-pass anchor
                   done; remaining: registry append DEC-004 row
                   + WORKFLOW_STATUS / RESEARCH_STATE / INDEX
                   line-3 sync + INDEX 2 new planning rows +
                   INDEX latest cycle log pointer to CYCLE-
                   20260506-005 + cycles/CYCLE-20260506-005.md
                   NEW + final TASK_SNAPSHOT flip to idle)

User-revised post-019 trajectory (per DEC-20260506-004
§"Why this matters"):
   Cycle 020 (in progress; closing this pass):
     code structure + implementation roadmap (combined; 2
     artifacts; markdown only; reviewable other-agent handoff
     surface)
   Renumbered cycle 021 (deferred from original DEC-003 cycle 020):
     SPEC-20260506-003 v0.2 comparator map addendum (NEW;
     candidate ID SPEC-006 reorganizes comparators per in-pool /
     out-of-pool / out-of-scope tiers per SPEC-004 Delta 5)
   Renumbered cycle 022:
     PAPER_DRAFT_V1.md Section 3 + Section 6 update for v0.2
     main-claim A+D framing (markdown only)
   Renumbered cycle 023:
     Cycle 015 G_run resumption (Critic A4 measured anchor;
     paused at S9 done; server-side execution per F-002)
   Renumbered cycle 024:
     Capability_match measurement pass per ABL-v02-5 (promotes
     inferred -> measured for 7-expert pool; server-side per
     F-002)
   Renumbered cycle 025:
     First ablation execution authorization per ABL-v02-N
     selection (server-side per F-002; per-ABL DEC + per-step
     micro gates required)

Resume action when interrupted mid S4 sync chain:
   - If decision_registry.md does NOT yet have a DEC-20260506-004
     row: that is the next sub-step. Append to the table
     immediately after DEC-20260506-003.
   - If WORKFLOW_STATUS.md / RESEARCH_STATE.md / INDEX.md line 3
     timestamps still say "2026-05-06 (cycle 019...)": that is
     the next sub-step. Update line 3 of each to point to
     cycle 020 closure.
   - If INDEX.md planning/ table does NOT yet have rows for
     DREAM3R_V02_CODE_STRUCTURE.md and DREAM3R_V02_
     IMPLEMENTATION_ROADMAP.md: add them.
   - If INDEX.md latest-cycle-log pointer still says CYCLE-
     20260506-004: update to CYCLE-20260506-005.
   - If cycles/CYCLE-20260506-005.md does not exist: write it
     (cycle 020 log; closed status; spans 2026-05-06 launch ->
     2026-05-07 closure).
   - When all 5 sync items above are done: flip TASK_SNAPSHOT
     Status to idle + update Last updated line with closure
     summary + flip Subtask board S4 row to done.

Resume action when cycle 020 fully closed:
   No active task. Status idle. Pending user direction on
   renumbered cycles 021-025 in CYCLE-20260506-005 §"Cycle 020
   outcome (closed)" / DEC-20260506-004 §"Why this matters".
   None of cycles 021..025 launches without explicit user
   direction. Do NOT propose training, checkpoint download,
   GPU runs, KYKT navigation change, frontend implementation,
   demo storyboard promotion past `draft`, thesis finalization,
   or retiring of any non-finalist track. DEC-20260501-004
   candidate-not-final + DEC-20260504-002 no-all-in still in
   force.

Hard rules carried (unchanged from prior cycles):
   - No training. No checkpoint download. No reproduction. No
     KYKT navigation change. No frontend implementation. No
     thesis finalization. No retiring of any non-finalist track.
   - DEC-20260501-004 (Dream3R candidate-not-final) and DEC-
     20260504-002 (no-all-in) still in force.
   - Cycle 020 is markdown only. Any code touch (whether per
     T-v02-N task in roadmap or otherwise) requires a separate
     DEC + per-step micro gates per F-002.
   - Honesty Override: every per-file change in CODE_STRUCTURE
     carries an evidence label; every per-task entry in
     IMPLEMENTATION_ROADMAP carries effort estimate +
     evidence label. NSA-related work is speculative for 3R
     transfer. DINOv3-S backbone is paper-derived. Effort
     estimates are inferred and need post-first-task
     calibration.
   - Trajectory revision in DEC-004 documented honestly with
     old-vs-new table; original DEC-003 trajectory NOT
     silently superseded.

Honor F-001 working rules throughout: do not Read large files
already cited in this snapshot; prefer Grep -n + Edit over full-
file Read + Write; cap large files in active context at <=2
simultaneously. Honor F-002: KYKT 3R model work runs server-side;
default to ssh + reuse before installing; check ssh_runner.py:22-44
ServerConfig before asking for SSH details. Cycle 020 itself is
markdown only and stays local.
```

Project state at this snapshot:
   Cycle 015 PAUSED at S9 done (NOT closed; NOT abandoned;
                                infrastructure is reusable as
                                future Critic A4 evidence anchor).
   Cycle 016 IN PROGRESS:
                                S1 done (DEC-20260506-001 + memory +
                                  prior snapshot redirect block)
                                S2 done (architecture spec v0.1
                                  written 2026-05-06; 1821 lines;
                                  95 KB; specs/SPEC-20260506-001-
                                  dream3r-architecture.md)
                                S3 done (ablation plan v0.1
                                  written 2026-05-06; 10 ablations
                                  in 3 tiers; specs/SPEC-20260506-
                                  002-dream3r-ablation-plan.md)
                                S4 done (comparator map v0.1
                                  written 2026-05-06; 14+ models;
                                  specs/SPEC-20260506-003-dream3r-
                                  comparator-map.md)
                                S5 in progress (TASK_SNAPSHOT
                                  updated; cycle log + remaining
                                  sync files pending)

Mainline redirect summary:
   - Old implicit framing: framework-first paper output.
   - New explicit framing: architecture-first; Dream3R architecture
     spec is the PRIMARY output; paper is SUPPORT.
   - Cycle 015 L3 measurement work is SUPPORT / prereq for the
     architecture spec, NOT mainline.
   - Train-first remains deferred / blocked.
   - DEC-501-004 (Dream3R candidate-not-final) and DEC-504-002
     (no-all-in) still in force.

Resume action when user returns:
   Primary path: cycle 016 is nearly complete. S5 sync chain
     is in progress (TASK_SNAPSHOT updated; remaining: cycle log,
     decision registry, other sync files). Once S5 finishes,
     cycle 016 closes.

   Candidate next actions after cycle 016 closes:
     - Review S2 architecture spec + S3 ablation plan + S4
       comparator map (user can read all three and give feedback)
     - v0.2 architecture spec revision (if review or ablation
       plan surfaces substrate/bus/module issues)
     - Paper rewrite to feature Dream3R architecture as central
       claim (per architecture spec Q6)
     - Resume cycle 015 G_run for measured Critic A4 evidence
       (per architecture spec Q5)
     - A7 Cross-Modal / A8 Active Perception spec drafting
       (per architecture spec Q3)
     - Training authorization (requires separate DEC)

   S2 spec Q1..Q6 resolution status:
     Q1 substrate ablation priority -> RESOLVED in S3 ablation
        plan: hybrid first; SSM-only second; transformer-only
        third. ABL-2 in SPEC-20260506-002 specifies all three.
     Q2 adversarial vs natural CR-rule triggering -> RESOLVED
        in S3: both (B1-B5 for performance, B6 for CR-rule
        firing verification).
     Q3..Q6 -> still open; surface during user review of cycle
        016 deliverables.
   Do NOT execute anything beyond markdown drafting without
   explicit user `Go`. Do NOT propose training. Do NOT propose
   thesis finalization. Do NOT promote any of the 4 storyboards
   past `draft`.

Hard rules still in force (verbatim from S2 spec Boundaries):
   - All 4 finalist demo storyboards (STORY-20260505-001..004)
     remain markdown `draft`. Do NOT promote any to `approved-for-
     showing` without a separate DEC.
   - Do NOT start any non-Critic L3 pilot (Memory / Permanence /
     Composer L3) / training / KYKT navigation change / frontend
     implementation without explicit user approval per
     AGENT_MASTER_PROMPT.md section 6.
   - Cycle 015 has narrow exceptions ONLY for the Critic L3 pilot
     scope; everything else stays gated.
   - DEC-20260506-001 authorizes architecture-spec DESIGN +
     ablation PLANNING; NOT training, NOT GPU runs, NOT checkpoint
     creation.
   - v0.1 spec carries 13 explicit boundaries in its "Boundaries"
     section; all in force.

Honor F-001 working rules throughout: do not Read large files
already cited in this snapshot; prefer Grep -n + Edit over full-
file Read + Write; cap large files in active context at <=2
simultaneously. Honor F-002: KYKT 3R model work runs server-side;
default to ssh + reuse before installing; check ssh_runner.py:22-44
ServerConfig before asking for SSH details.
```

If this snapshot's Status is `idle`, both cycle 015 (Critic L3 measurement) and cycle 016 (Dream3R architecture spec drafting) are closed; the next live phase requires a separate user direction.

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

Items still blocked on user approval per `AGENT_MASTER_PROMPT.md` section 6 (unchanged from prior cycles): final thesis selection, reproduction, training, checkpoint download, KYKT navigation change, frontend implementation, reusable Codex skill packaging, retiring any non-finalist track, declaring teacher-demo readiness, **showing any of the 4 demo storyboards (Critic / Memory / Permanence / Composer; all remain `draft`; promotion to `approved-for-showing` requires a separate DEC per finalist)**.

Cycle 013 launch decisions (locked 2026-05-05 from user message "好了，请你做实际的研究部署吧，无论是准备工作还是调研和资料搜集等" + clarification "Phase 2 准备 + 资料调研 (推荐)" delegating to agent; recorded in `decisions/DEC-20260505-003-cycle-013-launch.md`):

```text
1. Cycle 013 scope (delegated)        -> Phase 2 preparation + research
                                        mining. Three locked sub-passes:
                                        S2 source-mining cycle-013 pass
                                        (4 finalist x 3 axes coverage
                                        gaps); S3 paper related-work
                                        prose draft (replace skeleton
                                        bullets with prose in Sections
                                        1-7; draft Sections 8-9 as
                                        prose); S4 L3 prerequisites
                                        briefs per finalist (4 markdown
                                        files under experiments/).
                                        Cycle 013 explicitly excluded:
                                        L3 prototype, checkpoint
                                        download, KYKT runner log
                                        access, model touching,
                                        retroactive edits to prior
                                        cycle artifacts, contract
                                        revision.
2. v2.1 -> v2.2 candidates (delegated) -> NO revision in cycle 013.
                                        No new candidate surfaced.
                                        Both cycle-011 deferred
                                        candidates (8x8 grid partition;
                                        identity_consistency threshold
                                        pinning) remain deferred.
                                        VGGT capability-card gap
                                        surfaced by cycle-013 mining
                                        is per-card, not contract.
3. D3 first demo target (carried)     -> unchanged; Critic per cycle
                                        011 DEC-20260505-001. No
                                        cycle-013 reconsideration.
4. Blocked items                       -> unchanged from cycle 012
                                        closeout. "Showing any of
                                        the 4 demo storyboards"
                                        unchanged. L3 / clone /
                                        download / install / run / KYKT
                                        navigation change / frontend /
                                        thesis finalization / training
                                        all still gated.
```

Cycle 013 G2 status update:

```text
Before cycle 013: G2 = inferred-with-real-inventory-anchor (per cycle
                       012 COMPOSER-04 KYKT-metadata-derived card).
After  cycle 013: unchanged. EXP-20260505-004 inventories the closure
                  path (multi-regime workload; 3+ backbones; measured
                  route_regret) but cycle 013 did NOT execute it.
Closure remains gated on L3 prototype OR KYKT runner log access; both
require separate user authorization.
```

New tracking goal G7 introduced cycle 013:

```text
G7 (paper-related-work-prose-readiness):
  Status after cycle 013: inferred-with-prose-draft-anchor.
  Sections 1-7 of literature/PAPER_RELATED_WORK_SKELETON.md are prose;
  Sections 8-9 are drafted as prose. The cycle-009 case-card-gate that
  blocked prose drafting is lifted (cleared by cycle 010-012 case-card
  + storyboard portfolio).
  Closure: full Phase-2 paper writing (intro / methods / results /
  discussion). Gated on user direction on venue / length / scope.
```

Cycle 012 launch decisions (locked 2026-05-05 from user message "你给我规划吧，然后弄完后告诉我我们的工作做了哪些？" delegating to agent; recorded in `decisions/DEC-20260505-002-cycle-012-launch.md`):

```text
1. Storyboard reviewer pass (1)      -> NOT done in cycle 012. Deferred
                                        to demo-show-authorization
                                        moment when functional-vs-
                                        placeholder tradeoffs become
                                        concrete. Critic storyboard +
                                        the 3 new storyboards all
                                        remain `draft` unchanged in
                                        future agent passes (no silent
                                        revisions).
2. Cycle 012 scope (2)               -> (c) KYKT-metadata-derived
                                        COMPOSER-04 + (e) 3 finalist
                                        demo storyboards (Memory +
                                        Permanence + Composer; all
                                        markdown only; all `draft`).
                                        Options (a) close G6 / (b)
                                        close G2 / (d) request demo
                                        show authorization / (f) start
                                        paper related-work writing all
                                        deferred (gated or premature).
3. v2.1 -> v2.2 candidates (3)       -> NO revision. Both deferred
                                        candidates from cycle 011
                                        (8x8 grid partition for
                                        Permanence regions;
                                        identity_consistency threshold
                                        pinning at ~0.7) remain
                                        deferred. COMPOSER-04 fits
                                        cleanly into existing v2
                                        schema; no new sub-axis needed.
4. (4) blocked items unchanged from cycle 011 closeout; "showing the
   Critic demo" extended to "showing any of the 4 demo storyboards"
   (all 4 finalists now have draft storyboards as of cycle 012).
```

Cycle 012 G2 status update:

```text
Before cycle 012: G2 = inferred (tau_spread = 0.05 paper-derived;
                                 capability_match paper-derived).
After cycle 012:  G2 = inferred-with-real-inventory-anchor
                       (capability_match values now KYKT-metadata-
                        derived per COMPOSER-04; tau_spread itself
                        still inferred; closure still requires
                        measured route_regret; gated).
G2 NOT closed. Closure remains gated on L3 prototype OR KYKT runner
log access.
```

Cycle 015 launch decisions (locked 2026-05-05 from user message "授权 Critic L3 窄域 pilot" delegating cycle 015 entry to the Critic L3 pilot scope; recorded in `decisions/DEC-20260505-005-cycle-015-launch-critic-l3-pilot.md`):

```text
1. Cycle 015 entry (D5)              -> authorize Critic L3 pilot scope
                                        per planning/L3_PILOT_SELECTION.md
                                        "Recommended first-pilot scope" +
                                        EXP-20260505-001 prerequisite
                                        inventory. Allowed: clone Test3R
                                        + CTRL + DUSt3R + MASt3R; download
                                        required checkpoints; install
                                        minimum env on a single box; run
                                        one smoke loop on one hard-case
                                        input; emit one JSONL log;
                                        write thin wrapper
                                        dream_critic_loop.py + hand-
                                        derived capability_match YAML.
2. Per-step micro gates (D5')        -> 5 gates, each a separate
                                        user go in active conversation:
                                        G_clone, G_install, G_download,
                                        G_run, G_log_use. DEC-005 alone
                                        does NOT authorize any of these
                                        steps; each surfaces individually
                                        with proposed path / repos /
                                        checkpoints / hard-case input
                                        before execution.
3. v2.1 -> v2.2 candidates (D5'')    -> NO revision in cycle 015. Both
                                        cycle-011 deferred candidates
                                        (8x8 grid partition; identity_
                                        consistency threshold pinning)
                                        remain deferred. VGGT capability-
                                        card gap remains per-card
                                        (CASE-20260505-COMPOSER-05),
                                        not contract-level.
4. Composer second-pilot precondition (D5''') -> unchanged from
                                        L3_PILOT_SELECTION.md: VGGT row
                                        must be included or explicitly
                                        excluded with reason before any
                                        Composer route_regret sweep is
                                        frozen. Cycle 015 does not act on
                                        this; Composer pilot is OUT of
                                        cycle 015 scope.
5. Blocked items unchanged from cycle 014 closeout, with one tightening:
   the cycle-015 authorization is Critic-L3-only. Memory / Permanence /
   Composer L3 prototypes remain blocked. Final thesis selection,
   reproduction (in the sense of paper-result re-runs), training,
   KYKT navigation change, frontend implementation, reusable Codex
   skill packaging, retiring any non-finalist track, declaring
   teacher-demo readiness, and showing any of the 4 demo storyboards
   all remain blocked.
```

Cycle 015 G2 / G6 / G7 status update at launch:

```text
G2 (route_regret closure):  unchanged. inferred-with-real-inventory-
                            anchor (cycle 012 anchor). Critic smoke
                            does NOT close G2; G2 closure remains
                            gated on multi-regime measured
                            route_regret OR KYKT runner log access.
G6 (memory governance):     unchanged.
G7 (paper related-work
     prose readiness):      unchanged. inferred-with-blueprint-anchor
                            (cycle 014 anchor). Closure still gated
                            on user direction on venue / length /
                            scope of Phase 2 paper.
```

Cycle 016 mainline redirect (locked 2026-05-06 from user message "所以现在我们做的和新模型有啥关系？或者说什么时候能开始推进主线了？" + selection of "B. Architecture-first" from a 3-option strategic question; recorded in `decisions/DEC-20260506-001-mainline-architecture-first.md`):

```text
1. Mainline definition (D6)         -> Dream's mainline is
                                       **architecture-first**: design a
                                       new 3R architecture (transformer
                                       / SSM / state-space / hybrid) as
                                       a markdown spec + ablation plan
                                       + comparator map. NOT framework-
                                       first paper writing.

2. Cycle 015 posture (D6')          -> stays paused at S9 done. NOT
                                       closed. NOT abandoned. The L3
                                       infrastructure (test3r conda env
                                       on server; launch.py patch; F-002
                                       memory; 4 local shallow clones)
                                       is reusable as evidence anchor
                                       for the architecture spec's
                                       Critic-module section. G_run can
                                       be resumed later if the
                                       architecture spec needs measured
                                       evidence; until then, no G_run.

3. Paper Phase 2 blueprint (D6'')   -> demoted from primary output to
                                       SUPPORT artifact. Still useful
                                       (control-graph theory becomes
                                       the THEORY behind the
                                       architecture); but the
                                       architecture spec is the
                                       PRIMARY output of the project.

4. Past decisions still in force (D6''') -> DEC-20260501-004 (Dream3R
                                       candidate-not-final) and
                                       DEC-20260504-002 (no-all-in any
                                       single finalist) both still
                                       apply. Architecture spec must
                                       (a) remain a candidate that can
                                       be revised/replaced/merged, and
                                       (b) preserve all 4 finalist
                                       mechanisms as composable
                                       modules, not collapse into one.

5. Train-first (option C) NOT authorized. Architecture-first
   authorizes design + ablation planning, NOT training, NOT GPU
   ablation runs, NOT checkpoint creation. Train-first remains
   deferred / blocked.

6. Blocked items unchanged from prior cycles. Hard rules carried from
   AGENT_MASTER_PROMPT.md section 6 unchanged: no reproduction / no
   checkpoint download / no training / no KYKT navigation change / no
   frontend / no thesis finalization / no retiring of any non-finalist
   track / no demo storyboard promotion past `draft`.

7. Cycle 016 launch deferred. S1 of cycle 016 = this DEC + feedback
   memory + this snapshot block (done 2026-05-06). S2..S5 (architecture
   spec draft + ablation plan + comparator map + sync chain) deferred
   to next session per user "今日进度到此为止".
```

Cycle 016 G2 / G6 / G7 status update at redirect:

```text
G2 (route_regret closure):  unchanged. Architecture spec drafting does
                            NOT close G2 (still gated on measured
                            route_regret OR KYKT runner log access).
G6 (memory governance):     unchanged.
G7 (paper related-work
     prose readiness):      unchanged. Paper output is now SUPPORT,
                            not primary; G7 closure is no longer the
                            project's main milestone. The new primary
                            milestone is "Dream3R architecture spec
                            v1 draft" (call it G8 if/when formalized).
                            Whether G7 stays open vs is retired is a
                            separate cycle-016 decision.
```

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

F-002  agent assumed local Windows execution for KYKT 3R model work
       (2026-05-05, cycle 015 S7 G_install pre-flight)
       cause:  agent probed local GPU/Python/conda and proposed creating
               a conda env on the local Windows box and pip-installing
               dust3r/mast3r/Test3R locally. Did NOT consult
               E:\kykt\WORK.md or
               E:\kykt\.omx\plans\kykt-app-backend-model-integration-plan.md,
               both of which document a canonical server-side topology:
               KYKT 3R models live on /hdd3/kykt26 on a remote server,
               reached via system ssh + scp. Existing runners
               (dust3r_runner.py, mast3r_runner.py, monst3r_runner.py,
               spann3r_runner.py, fast3r_runner.py) already provide envs
               + checkpoints for 4 of the 5 inventoried backbones.
               EXP-20260505-001 was paper-derived and did NOT anchor to
               the production runner inventory; the agent inherited the
               paper-derived framing without re-checking topology.
       avoid:  before any L3 / training / heavy-compute G_install /
               G_download / G_run gate proposal, read E:\kykt\WORK.md +
               E:\kykt\.omx\plans\kykt-app-backend-model-integration-plan.md
               first; default to SERVER-side execution; reuse existing
               runners before installing new ones; ask the user for
               SSH host / path when topology details are missing rather
               than improvising. Local Windows box = markdown + shallow
               code-reading clones + orchestration + result inspection
               only; not the model execution target.
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
4. ID conventions are fixed: `SPEC-YYYYMMDD-NNN`, `DEC-YYYYMMDD-NNN`, `CYCLE-YYYYMMDD-NNN`, `CASE-YYYYMMDD-NNN`. Use the current session date prefix for new artifacts.
5. Sentence case headers. No em-dashes.
