# Dream Workflow Status

Last updated: 2026-05-14 (Track B 3R-mix wound down to arXiv-only route on 2026-05-14; README rewritten as canonical entry, Typst legacy files marked deprecated, release checklist appended to NEW_CHAT_HANDOFF.md; Track A Dream3R v0.3 server-verified through cycle 034 unchanged)

## Current Phase

```text
Phase 1.5: Research Workflow Deployment
```

## Current Mode

```text
Two parallel tracks at a checkpoint:
  Track A (Dream3R v0.3 code, architecture-first mainline per DEC-20260506-001):
    server-verified on synthetic + first KITTI real-data smoke; W1-W18
    implementation present (W17-W18 tensor-contract level only); MASt3R +
    Spann3R real adapters loaded; Fast3R real path blocked on `omegaconf`
    in dream3r conda env; CUT3R / MoGe-2 / DepthAnything / Test3R remain
    deterministic fallback.
  Track B (3R-mix Chinese survey, separate workspace Dream/3R-mix/):
    16-page LaTeX manuscript; recommended deliverable
    `deliverables/3r_survey_stage_final_2026-05-13_refined.pdf`.
    **Wound down 2026-05-14 to arXiv-only route (route C, no venue
    submission)**; README rewritten as canonical entry, Typst legacy
    files marked deprecated, release checklist appended to
    NEW_CHAT_HANDOFF.md. Internal terms deliberately absent from
    manuscript surface.

No new reproduction or heavy install authorized.
No real-data training authorized.
No 3DGS renderer install authorized.
Paper writing is now a separate workstream (Track B) but still support, not
primary; Track A architecture-first mainline holds.
Frontend implementation remains delegated to Gemini CLI / designated frontend agent.
```

## Active Thesis Candidate

```text
Dream3R: Geometry-Governed State and Test-Time Reasoning for Long-Context 3R
```

Status:

```text
candidate, not final
```

## Active Workflow Decision

Deploy Dream as a markdown-first research pipeline:

```text
Source -> Mechanism -> 3R Translation -> Research Unit -> Score -> Decision -> Plan -> Implementation
```

## Canonical Agent Prompt

```text
E:\kykt\Dream\AGENT_MASTER_PROMPT.md
```

Use this prompt when handing Dream work to Codex, another agent, or a subagent.

## Canonical Frontend Handoff Prompt

```text
E:\kykt\Dream\handoff\FRONTEND_DESIGN_HANDOFF_PROMPT.md
```

Use this prompt when preparing KYKT frontend design work for Gemini CLI.

## Active Workstreams

| Workstream | Status | Next artifact |
|---|---|---|
| Research workflow | active | `paradigm/RESEARCH_WORKFLOW.md` |
| Collaboration roadmap | active | `handoff/COLLABORATION_ROADMAP.md` |
| Data model | active | `paradigm/RESEARCH_DATA_MODEL.md` |
| Source registry | active; cycle 013 mining pass added SRC-2026-009..015 (7 new rows: MapAnything / Julian Ost AAAI-2026 driving permanence / tttLRM / awesome-dust3r curated index / DUSt3R-MASt3R-VGGT MVS evaluation / NTIRE 2026 / VGGT) | `registry/source_registry.md` |
| Research unit registry | seeded | `registry/research_unit_registry.md` |
| Decision registry | seeded | `registry/decision_registry.md` |
| Cycle logs | active | `cycles/CYCLE-20260511-001.md` (cycle 034 done; KITTI real-data smoke + Mamba/Gaussian + Track B 3R-mix kickoff); `cycles/CYCLE-20260510-001.md` (cycle 033 done; W1-W16 v0.3 architecture advancement); `cycles/CYCLE-20260508-008.md` (cycle 031 local Memory v0.3 P0 scaffold) |
| Dream3R v0.3 code (Track A) | active; server-verified at `/hdd3/kykt26/code/dream3r/`; first KITTI real-data smoke on `2011_09_26_drive_0001_sync_02` window pair (pointmap L2 20.47 = integration evidence, not trained quality) | `code/dream3r/REVIEW_PROMPT.md`, `code/dream3r/RECENT_PROGRESS.md`, `code/dream3r/NEXT_PHASE_ROADMAP.md` |
| 3R-mix Chinese survey (Track B) | **wound down 2026-05-14 (route C: arXiv-only, no venue submission)**; 16 A4 pages, 43 references, 6 figures (4 TikZ/composite + 2 paper-Fig.1 composites), 5 booktabs tables, 0 LaTeX errors / 0 warnings; release checklist + Typst legacy deprecation markers added; deliberately decoupled from Dream/KYKT internal vocabulary | `Dream/3R-mix/README.md`, `Dream/3R-mix/NEW_CHAT_HANDOFF.md`, `Dream/3R-mix/main.tex`, `Dream/3R-mix/deliverables/3r_survey_stage_final_2026-05-13_refined.pdf` |
| Experiment planning | active; local v0.3 P0 scaffold now exists and ABL-memory-0 passed, but later ablations still require separate DEC + gate | `experiments/prototypes/memory_v03_p0/outputs/summary_go_no_go.md` |
| Agent master prompt | active | `AGENT_MASTER_PROMPT.md` |
| C2 Memory v0.3 | active architecture addendum + P0 plan + reviewed ablation addendum + local P0 scaffold. ABL-memory-0 passed as a fixture/logging gate only; C2 memory quality remains unvalidated | `specs/SPEC-20260508-001-dream3r-c2-memory-v03-addendum.md` + `planning/MEMORY_V03_DESIGN_STUDY.md` + `planning/MEMORY_V03_P0_PROTOTYPE_PLAN.md` + `specs/SPEC-20260508-002-dream3r-memory-v03-ablation-addendum.md` + `planning/MEMORY_V03_ABLATION_REVIEW.md` + `experiments/prototypes/memory_v03_p0/README.md` |
| Research content roadmap | active | `paradigm/RESEARCH_CONTENT_ROADMAP.md` |
| Multi-track research canvas | active; cycle 008.5 four-finalist + no-all-in section appended | `planning/MULTI_TRACK_RESEARCH_CANVAS.md` |
| Research graph / paper start | active; cycle 008.5 F6 row note + Next Concrete Artifact supersede applied | `planning/RESEARCH_GRAPH_AND_PAPER_START.md` |
| Branch comparison matrix | filled first comparative pass (cycle 004); cycle 008.5 supersede annotations applied | `planning/BRANCH_COMPARISON_MATRIX.md` |
| Branch shortlist decision surface | user approved option B (cycle 008) | `planning/BRANCH_SHORTLIST_DECISION_SURFACE.md` |
| Architecture mechanism intake | first-pass active | `planning/ARCHITECTURE_MECHANISM_INTAKE.md` |
| Action taxonomy / proxy metrics | first compact pass (cycle 006); cycle 008.5 A5 split + supersede annotations applied | `planning/ACTION_TAXONOMY_AND_PROXY_METRICS.md` |
| Proxy case-card template | active form; first portfolio populated in cycle 009 (6 cards: 3 Critic + 3 Composer, paper-derived) | `templates/proxy_case_card.md` + `cases/` |
| Finalist mechanism spec template | populated for three finalists in cycle 008 | `templates/finalist_mechanism_spec.md` |
| Geometry Critic finalist spec | draft (L1); L2 case cards drafted in cycle 009 (paper-derived under v1 contract); D3 first teacher demo target = Critic per cycle 011 DEC-20260505-001; demo storyboard `STORY-20260505-001-critic.md` drafted in cycle 011 (status: draft only; showing not authorized) | `specs/SPEC-20260503-001-geometry-critic.md` + `cases/CASE-20260504-CRITIC-01..03.md` + `storyboards/STORY-20260505-001-critic.md` |
| Executive Memory finalist spec | draft (L1); L2 case cards drafted in cycle 010 under v2 contract (3 cards across MonST3R / Spann3R / MASt3R regimes); CR-3 producer side closes cycle-009 CRITIC-03 forward-reference null | `specs/SPEC-20260503-002-executive-memory.md` + `cases/CASE-20260504-MEMORY-01..03.md` |
| Dynamic Object Permanence finalist spec | draft (L1); L2 case cards drafted in cycle 010 under v2 contract (3 cards: MonST3R primary + MASt3R static control + synthetic identity-validation); CR-2 producer side closes cycle-009 gap G1 | `specs/SPEC-20260503-003-dynamic-object-permanence.md` + `cases/CASE-20260504-PERMANENCE-01..03.md` |
| 3R Composer finalist spec | draft (L1); L2 case cards drafted in cycle 009 (paper-derived); CASE-COMPOSER-03 v2 row promoted to canonical per DEC-20260504-004; CASE-COMPOSER-04 KYKT-metadata-derived added in cycle 012 (advances G2 inferred -> inferred-with-real-inventory-anchor; G2 NOT closed); CASE-COMPOSER-05 added cycle 014 as VGGT capability-card gap addendum (per-card gap, no v2.2 contract revision); demo storyboard `STORY-20260505-004-composer.md` drafted cycle 012 | `specs/SPEC-20260504-001-3r-composer.md` + `cases/CASE-20260505-COMPOSER-01..05.md` + `storyboards/STORY-20260505-004-composer.md` |
| Cross-spec signal contract | **v2.1 active** (per DEC-20260505-001): additive revision over v2 — adds "Forward-reference null protocol" subsection formalizing the pattern exercised by cycle-009 + cycle-010 cards; v2 substance unchanged (alpha = 0.5 inferred; signal owner table; CR-1..CR-6; cost_adjusted_match; route_regret cost-typed). v1 + v2 prose preserved. Cycle 011 G5 closed by this revision; cycle 010 G4 closed-by-documentation under the protocol. v2 -> v3 candidates 8x8 grid partition + identity_consistency threshold pinning deferred. | `paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md` + `decisions/DEC-20260505-001-cycle-011-launch-and-d3-demo-target.md` |
| Literature guidance board | v1 active; post-cycle-013 SPINE refresh fold-in done. PAPER_RELATED_WORK_SKELETON.md upgraded cycle 013 to prose draft. Cycle 014 added PAPER_PHASE2_BLUEPRINT.md as a claim-safe paper-writing plan; G7 advanced to blueprint anchor but not closed | `literature/INDEX.md` + four `literature/SPINE_*.md` + `literature/CRITICAL_NOTES.md` + `literature/PAPER_RELATED_WORK_SKELETON.md` + `literature/PAPER_PHASE2_BLUEPRINT.md` |
| Work risk register | v1 active; consolidates per-spec + cross-spec risks | `planning/WORK_RISK_REGISTER.md` |
| Demo storyboard template | active form; all 4 finalists now have draft storyboards (Critic from cycle 011 = D3 first demo target; Memory + Permanence + Composer from cycle 012); none authorized for showing; promotion to `approved-for-showing` requires a separate per-finalist DEC | `templates/demo_storyboard.md` + `storyboards/STORY-20260505-001..004.md` |
| Teacher audience profile | placeholder; awaits user input to unblock D3 | `paradigm/TEACHER_AUDIENCE_PROFILE.md` |
| Source mining (cycle 005 pass) | complete for visual priors, depth priors, active perception, event VO | `sources/FRONTIER_SOURCE_MAP.md` (Cycle 005 Source Mining Pass section) |
| Workspace reorganization (cycle 006) | complete; topical subdirectories + archive/ + INDEX.md | `cycles/CYCLE-20260502-006.md` |
| Research & code discipline (cycle 007) | active rulebook for synthesis behavior and Dream-driven code | `paradigm/RESEARCH_CODE_DISCIPLINE.md` |
| Finalist shortlist approval (cycle 008) | user-approved option B; three finalist specs drafted | `decisions/DEC-20260503-002-finalist-shortlist-approval.md` |
| Composer finalist upgrade (cycle 008.5) | user-approved; SPEC-20260504-001 drafted; cross-spec contract formalized | `decisions/DEC-20260504-001-composer-finalist-upgrade.md` |
| No-all-in posture (cycle 008.5) | user-locked; D3 deferred until cycle 009 case-card data + audience profile | `decisions/DEC-20260504-002-no-all-in-on-single-finalist.md` |
| Frontend handoff prompt | active | `handoff/FRONTEND_DESIGN_HANDOFF_PROMPT.md` |
| KYKT backend integration | support only | no backend service changes yet |
| KYKT frontend integration | downstream only | no UI work unless research content and support contract exist |

## Blocked Until User Decision

- any new reproduction, server run, model run, or heavy install
- any new checkpoint download
- C2 v0.3 server integration, model import, or any ablation beyond ABL-memory-0 without a separate DEC
- KYKT Dream page or navigation change
- Codex direct frontend implementation
- major Gemini CLI frontend redesign instruction
- final thesis selection
- deepening any single thesis branch as the default path
- reusable Codex skill packaging

## Recommended Next User Decision

Cycle 034 closeout recommendation (no active cycle; cycles 033 + 034 retroactively logged 2026-05-13; Track B 3R-mix wound down 2026-05-14 to arXiv-only route):

```text
Track B 3R-mix Chinese survey wound down on 2026-05-14 (route C:
arXiv-only, no venue submission). Options below are now narrowed to
Track A and meta-decisions; pick at most one to launch next.

  A. Real-data ablation table on KITTI using the 4 existing synthetic
     ablation variants (baseline_cross_attention / mamba_hybrid /
     no_nsa / no_stable_memory) to produce paired pointmap-L2 /
     depth-RMSE numbers.

  B. Critic threshold calibration on real KITTI geometry distributions
     (Sampson / covisibility / depth-inconsistency histograms; pick
     thresholds; no training, just statistics).

  C. (removed — Track B survey wound down on 2026-05-14)

  D. Resolve Fast3R `omegaconf` dependency in dream3r conda env
     (requires a separate install-authorizing DEC).

  E. Pause and return to research design / paper writing direction
     choice.
```

Track A architecture-first remains the mainline per DEC-20260506-001. Track B paper writing remains support; it is not the project's primary output.

Still blocked on user approval:

- final thesis selection
- moving any finalist from L2 proxy evidence to L3 prototype code
- reproducing any candidate model
- training or fine-tuning
- downloading any new checkpoint
- changing KYKT navigation
- Codex directly editing KYKT frontend code
- packaging a reusable Codex skill
- declaring teacher-demo readiness
- discarding any non-finalist track (Cross-Modal, Active Perception)
- **showing any of the 4 demo storyboards** (Critic / Memory / Permanence / Composer; all `draft`; promotion to `approved-for-showing` requires a separate per-finalist DEC)

## Guidance File Sync Rule

When Dream creates or promotes a workflow artifact, update the relevant guidance files in the same pass. **`TASK_SNAPSHOT.md` updates first in this chain** so that a sync interrupted partway through still leaves a valid resume pointer:

- `TASK_SNAPSHOT.md` (highest-authority resume pointer; updated first; see its own "Update protocol" section for transitions)
- `AGENT_MASTER_PROMPT.md`
- `README.md`
- `WORKFLOW_STATUS.md`
- `RESEARCH_STATE.md`
- current cycle log under `cycles/`
