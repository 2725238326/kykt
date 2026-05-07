# Dream Task Snapshot

Last updated: 2026-05-08 (cycle 025 DONE: 方向 3.1 research pass completed; CUT3R persistent state + Spann3R spatial memory code read on server; Dream/planning/MEMORY_V03_DESIGN_STUDY.md written; C2 v0.3 direction = state-token recurrence + spatial key/value memory + geometry-aware bus-gated writes; no model run, no training, no checkpoint download)

Status: **idle** (cycle 025 done; next recommended task = write a C2 Memory v0.3 spec addendum that supersedes Delta 3 from vector AnchorBank to token state + spatial bank, markdown only)

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
task_id:    cycle-025
phase:      Architecture-first research correction; C2 Memory mechanism
            study after cycle 024 framework-heavy drift
cycle:      025 (方向 3.1: CUT3R persistent state + Spann3R spatial
            memory code reading -> Dream3R C2 Memory v0.3 design study)
status:     done (S1 snapshot start anchor done; S2 CUT3R/Spann3R/
            Dream3R C2 code reading done; S3 MEMORY_V03_DESIGN_STUDY
            written; S4 sync chain done)
```

One-line description:

```text
Cycle 025 corrected the project direction from framework scaffolding
back to mechanism research. It read CUT3R state-token recurrence,
Spann3R spatial key/value memory, and current Dream3R C2 code, then
wrote planning/MEMORY_V03_DESIGN_STUDY.md. Main conclusion: Dream3R
C2 v0.3 should stop being GRU + vector cache + NSA label, and become
state-token recurrence + explicit spatial key/value memory +
geometry-aware bus-gated writes.

No training authorized. No model run. No checkpoint download. Server
use was static code reading over SSH only. Local change was markdown
research documentation plus sync metadata.
```

## Subtask board (cycle 025 memory mechanism study; single session 2026-05-08)

| ID | Subtask | Status | Canonical artifact |
| --- | --- | --- | --- |
| S1 | Start-of-pass TASK_SNAPSHOT anchor per F-001 rule 6 | done | `TASK_SNAPSHOT.md` |
| S2 | Read CUT3R state-token and LocalMemory code on server | done | `/hdd3/kykt26/code/cut3r/src/dust3r/model.py`, `/hdd3/kykt26/code/cut3r/src/dust3r/blocks.py` |
| S3 | Read Spann3R SpatialMemory and DUSt3R integration code on server | done | `/hdd3/kykt26/code/spann3r/spann3r/model.py` |
| S4 | Read current Dream3R C2 implementation and compare gaps | done | `/hdd3/kykt26/code/dream3r/dream3r/modules.py`, `memory_anchor_bank.py`, `nsa_attention.py` |
| S5 | Write design study and sync metadata | done | `planning/MEMORY_V03_DESIGN_STUDY.md`, `cycles/CYCLE-20260508-002.md`, sync chain |

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
pass_name:        Cycle 025 close pass (方向 3.1 memory mechanism
                  study done in single session 2026-05-08)
date:             2026-05-08
trigger:          User asked to read HANDOFF_PROMPT_MEMORY_RESEARCH
                  and TASK_SNAPSHOT, then "do the project better".
files_modified:   TASK_SNAPSHOT.md (start anchor + final flip)
                  WORKFLOW_STATUS.md (line-3 status update)
                  RESEARCH_STATE.md (line-3 status update)
                  INDEX.md (planning row for MEMORY_V03)
new_artifacts:    planning/MEMORY_V03_DESIGN_STUDY.md (NEW;
                  CUT3R state-token recurrence + Spann3R spatial
                  key/value memory study; Dream3R C2 v0.3 proposal)
                  cycles/CYCLE-20260508-002.md (NEW; concise
                  cycle 025 log)
discipline:       Research-only. Server-side work was static code
                  reading over SSH. No model run, no training, no
                  checkpoint download, no Dream3R server code edit.
budget_event:     F-001 honored by targeted code slices and Grep/
                  Select-String. F-002 honored by using server-side
                  code paths under /hdd3/kykt26 and keeping local
                  Windows as markdown/orchestration only.

prior_pass_name:  Cycle 022 close pass (S1-S5 done in single session
                  2026-05-07; combined Path C reattempt SUCCEEDED
                  + paper Section 3+6 v0.2 rewrite done; API gateway
                  recovered since cycle 021; 7 RA items captured
                  for conditional cycle 023.5)
prior_pass_date:  2026-05-07
prior_pass_files: TASK_SNAPSHOT.md, registry/decision_registry.md,
                  WORKFLOW_STATUS.md, RESEARCH_STATE.md, INDEX.md,
                  literature/PAPER_DRAFT_V1.md, decisions/DEC-
                  20260507-002-cycle-022-path-c-reattempt-and-
                  paper-s3s6-rewrite.md, cycles/CYCLE-20260507-
                  002.md

older_prior_pass: Cycle 021 close pass (S1+S3+S4+S5 done in
                  single session 2026-05-07; S2 Path C BLOCKED
                  by API gateway 500 panic — 4 attempts × ~3 min
                  each; cycle closes as done-with-S2-blocked-by-
                  infrastructure per Honesty Override Option β;
                  v0.2 markdown trio CLOSED via SPEC-20260507-001)
prior_pass_date:  2026-05-07
prior_pass_files: TASK_SNAPSHOT.md, registry/decision_registry.md
                  (DEC-20260507-001 append), WORKFLOW_STATUS.md,
                  RESEARCH_STATE.md, INDEX.md, specs/SPEC-
                  20260506-003 (Version history pointer append),
                  + new artifacts decisions/DEC-20260507-001
                  (~430 lines), specs/SPEC-20260507-001 (~880
                  lines), cycles/CYCLE-20260507-001.md
```

## If interrupted, resume from

If a new agent or new conversation is picking this up cold:

```text
1. Read this file (you are here).

2. Read decisions/DEC-20260507-002-cycle-022-path-c-reattempt-
   and-paper-s3s6-rewrite.md — this is the most recent strategic
   decision and documents the cycle 022 combined scope (Path C
   reattempt + paper §3+§6 v0.2 rewrite; both DONE; see below).

3. Read cycles/CYCLE-20260507-002.md — this is the cycle 022 log
   documenting: Path C SUCCEEDED (API gateway recovered; both
   agents returned 5-section reviews); 7 review-action items
   RA-01..07 captured for v0.3 addenda (conditional cycle 023.5);
   paper PAPER_DRAFT_V1.md updated to v1.2 (§3.8 + §6.0–6.3
   added; §3.1–3.7 + §6.4 preserved). Cycle 022 status = DONE.

4. Read decisions/DEC-20260506-004 (cycle 020 combined planning),
   decisions/DEC-20260506-003 (cycle 019 ablation plan v0.2),
   decisions/DEC-20260506-002 (cycle 018 v0.2 architecture deltas),
   decisions/DEC-20260506-001 (mainline architecture-first) for
   parent-cycle context.

5. Read these cycle 021 deliverables (already done in this pass):
   - specs/SPEC-20260507-001-dream3r-comparator-map-v02.md (S3;
     ~880 lines; closes v0.2 markdown trio; 5-tier reorganization
     + 3 NEW axes + threat re-rank; 5 risks + 5 open questions
     including ABL-v02-10 Test3R-alone candidate)
   - cycles/CYCLE-20260507-001.md (S4; cycle 021 log; status
     done-with-S2-blocked-by-infrastructure)

6. Reference SPEC bodies ONLY when needed (do NOT re-Read full
   files; cite by section + line anchor):
   - specs/SPEC-20260506-004-dream3r-architecture-v02.md (v0.2
     architecture; six deltas; 882 lines)
   - specs/SPEC-20260506-005-dream3r-ablation-plan-v02.md (v0.2
     ablation plan; nine ABL-v02; 991 lines)
   - specs/SPEC-20260506-003-dream3r-comparator-map.md (v0.1
     comparator map substrate; 625 lines; preserved unchanged
     in body; only Version history tail received v0.2 pointer)
   - specs/SPEC-20260506-001-dream3r-architecture.md (v0.1
     architecture; 1821 lines; do NOT re-Read)
   - planning/DREAM3R_V02_CODE_STRUCTURE.md (1086 lines; cycle
     020 deliverable; Path C review TARGET — deferred)
   - planning/DREAM3R_V02_IMPLEMENTATION_ROADMAP.md (1226 lines;
     cycle 020 deliverable; Path C review TARGET — deferred)
   - code/dream3r/PLAN.md (v0.1 user-authored implementation
     roadmap; preserved unchanged)

7. Read C:\Users\27252\.claude\projects\e--kykt\memory\
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
                   written; six deltas; main-claim narrowed to
                   A+D).
   Cycle 019 DONE (v0.2 ablation plan addendum; SPEC-20260506-
                   005 written; 9 ABL-v02; per-ABL review
                   checklist for other-agent handoff).
   Cycle 020 DONE (combined v0.2 code structure + implementation
                   roadmap planning artifacts; DREAM3R_V02_CODE_
                   STRUCTURE.md + DREAM3R_V02_IMPLEMENTATION_
                   ROADMAP.md NEW; trajectory revision per user
                   "高强度推进").
   Cycle 021 DONE-WITH-S2-BLOCKED:
                S1 done (DEC-20260507-001 written ~430 lines;
                   cycle 021 launch + combined scope + Path C
                   protocol + 5 open questions Q1-Q5)
                S2 BLOCKED (Path C activation; 4 sub-agent
                   attempts × ~3 min each all failed with API
                   500 nil-pointer panic on Calcium-Ion/new-api
                   gateway; deferred to cycle 022 per Honesty
                   Override Option β; full incident in
                   CYCLE-20260507-001 §"Path C activation
                   incident")
                S3 done (specs/SPEC-20260507-001-dream3r-
                   comparator-map-v02.md written ~880 lines;
                   closes v0.2 markdown trio: architecture +
                   ablation + comparator)
                S4 done (cycles/CYCLE-20260507-001.md NEW)
                S5 done (full sync chain)
   Cycle 022 DONE:
                S1 done (DEC-20260507-002 written; combined
                   cycle scope lock + protocols + post-022
                   trajectory)
                S2 done (Path C SUCCEEDED — both agents 38.2s
                   + 47.5s; API gateway recovered; 7 RA items
                   captured: RA-01..04 CODE_STRUCTURE gaps +
                   RA-05..07 IMPLEMENTATION_ROADMAP risks)
                S3 done (literature/PAPER_DRAFT_V1.md v1.2;
                   §3.8 six v0.2 deltas NEW + §6.0–6.3 v0.2
                   comparator positioning NEW; §3.1–3.7 +
                   §6.4 preserved)
                S4 done (cycles/CYCLE-20260507-002.md NEW)
                S5 done (full sync chain)

v0.2 markdown trio + paper status (post cycle 022):
   - Architecture: SPEC-20260506-004 v0.2 (cycle 018; 6 deltas;
     A+D pillars)
   - Ablation plan: SPEC-20260506-005 v0.2 (cycle 019; 9 ABL-
     v02 with per-ABL review checklist)
   - Comparator map: SPEC-20260507-001 v0.2 (cycle 021 S3;
     5-tier reorganization + 3 new axes + threat re-rank)
   - Paper: PAPER_DRAFT_V1.md v1.2 (cycle 022 S3; §3.8 + §6.0–
     6.3 added; §3.1–3.7 + §6.4 v0.1 preserved)
   All 4 artifacts are now v0.2-coherent. v0.1 bodies of all 3
   substrate specs preserved unchanged per Discipline rule 5.

Post-022 trajectory (per CYCLE-20260507-002 + DEC-20260507-002
§"Post-022 trajectory"):
   Cycle 023: SPEC-002 v0.2 ablation plan addendum (markdown only;
              adds ABL-v02-1..10 new ablations; includes ABL-v02-10
              Test3R-alone for Q1; ABL-v02-4 VGGT offline-batch
              baseline for Q2; address RA-07 Pillar A task gap from
              Path C Agent B).
   Cycle 023.5 (CONDITIONAL): v0.3 addendum cycle for cycle 020
              planning artifacts — triggered if RA-01..07 items are
              actioned; produces v0.3 addenda to CODE_STRUCTURE +
              IMPLEMENTATION_ROADMAP via fresh DEC + new spec files
              (B-roadmap-F: no in-place modification); priority items:
              RA-01 (latency budget), RA-02 (ExpertAdapter ABC),
              RA-05 (checkpoint inventory).
   Cycle 024: Cycle 015 G_run resumption (Critic A4 measured
              anchor; server-side per F-002).
   Cycle 025: Capability_match measurement architecture (server-side
              measurement design for route_regret + capability_match).
   Cycle 026: First T-v02-N code task execution (server-side per
              F-002; per-task DEC + per-step micro gates + reviewer
              authorization per IMPLEMENTATION_ROADMAP B-roadmap-F).

Resume action when user returns:
   Cycle 022 is DONE. No active task. Status `idle`. Surface cycle
   023 (ablation plan addendum) + conditional cycle 023.5 (v0.3
   planning addenda; only if RA-01..07 are to be actioned now).

   If user asks about Path C findings:
   - Agent A (CODE_STRUCTURE): 3 HIGH/MEDIUM gaps (latency budget
     absent; ExpertAdapter ABC unresolved; NSA output combination
     underspecified); 1 LOW gap (losses.py labeling contradiction).
     Full details in CYCLE-20260507-002 §S2 / RA-01..04.
   - Agent B (IMPLEMENTATION_ROADMAP): Pillar A faithfulness gap;
     3 risks (checkpoint inventory; NSA kernel cu121; T-v02-F
     oversized). Full details in CYCLE-20260507-002 §S2 / RA-05..07.
   - All 7 RA items captured for conditional cycle 023.5 v0.3
     addenda. NOT actioned in cycle 022 (B-roadmap-F rule).

   None of cycles 023..026 launches without explicit user direction.
   Do NOT propose training, checkpoint download, GPU runs, KYKT
   navigation change, frontend implementation, demo storyboard
   promotion past `draft`, thesis finalization, or retiring of any
   non-finalist track. DEC-20260501-004 candidate-not-final +
   DEC-20260504-002 no-all-in still in force.

Hard rules carried (unchanged from prior cycles):
   - No training. No checkpoint download. No reproduction. No
     KYKT navigation change. No frontend implementation. No
     thesis finalization. No retiring of any non-finalist track.
     No demo storyboard promotion past `draft`. No teacher-
     demo readiness claim.
   - DEC-20260501-004 (Dream3R candidate-not-final) and
     DEC-20260504-002 (no-all-in) still in force.
   - Cycles 021-022 were markdown only. Any code touch (whether
     per T-v02-N task or otherwise) requires a separate DEC +
     per-step micro gates per F-002 + reviewer authorization
     per IMPLEMENTATION_ROADMAP B-roadmap-F.
   - Honesty Override: VGGT offline-batch threat to pillar D
     acknowledged in PAPER_DRAFT_V1.md §6.2 and in
     CYCLE-20260507-002; no ablation numbers manufactured; all
     v0.2 paper claims carry evidence labels; Path C Agent
     findings documented verbatim without cherry-picking.
   - Trajectory adherence: paper v1.2 delivered as planned per
     cycle 022 scope; Path C SUCCEEDED; RA items captured;
     v0.3 addendum deferred to conditional cycle 023.5.
   - Per-task review checklist pattern (cycle 020) first real
     exercise SUCCEEDED in cycle 022. Path C = operational and
     confirmed working. API gateway stable at time of cycle 022.

Honor F-001 working rules throughout: do not Read large files
already cited in this snapshot; prefer Grep -n + Edit over full-
file Read + Write; cap large files in active context at <=2
simultaneously. Honor F-002: KYKT 3R model work runs server-side;
default to ssh + reuse before installing; check ssh_runner.py:22-44
ServerConfig before asking for SSH details. Cycles 022 is markdown
only and stayed local.
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
