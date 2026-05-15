# Dream Workflow Status

Last updated: 2026-05-16 (cycle 036 closed — DEC-20260516-001 authorized dual-scope packaging + proposal kickoff; Part A 3 files in `3R-mix/deliverables/` (SUBMISSION_PACKAGE_ADVISOR + SUBMISSION_RECORD + RELATION_TO_TRACK_A) package the 2026-05-15 Track B Chinese survey for advisor / school internal review; Part B 4 files in new subdirectory `planning/proposal_dream3r/` (OUTLINE_V1 + STYLE_CONTRACT + DRAFT_INTERNAL_V1 + DRAFT_EXTERNAL_V1) launch the Chinese Dream3R 开题报告 dual-draft scaffold (§ 1 ~1800 内 / ~1500 外 字 complete; § 2-§ 9 placeholders for cycle 037+); WORK_RISK_REGISTER v1.1 → v1.2 additive (+3 rows R-PROP-VOCAB-1 / R-PROP-CLAIM-1 / R-PROP-SYNC-1); G2 + G3a + G3b + G4 vocab firewall + over-claim greps all 0 hits; no spec / code / server action; Track A Dream3R v0.3 server-verified through cycle 034 unchanged; Track B 3R-mix Chinese survey wound down 2026-05-14 + 2026-05-15 prose naturalization pass remains the canonical deliverable for the actual submission)

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
    18-page LaTeX manuscript; recommended deliverable
    `deliverables/3r_survey_stage_final_2026-05-15_natural.pdf`.
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
| Cycle logs | active | `cycles/CYCLE-20260516-001.md` (cycle 036 done; advisor submission packaging + Dream3R 开题报告 dual-draft kickoff + risk register v1.2 + sync chain); `cycles/CYCLE-20260515-001.md` (cycle 035 done; survey-driven markdown deliverables + 4 risk register additions + sync chain); `cycles/CYCLE-20260511-001.md` (cycle 034 done; KITTI real-data smoke + Mamba/Gaussian + Track B 3R-mix kickoff); `cycles/CYCLE-20260510-001.md` (cycle 033 done; W1-W16 v0.3 architecture advancement); `cycles/CYCLE-20260508-008.md` (cycle 031 local Memory v0.3 P0 scaffold) |
| Dream3R v0.3 code (Track A) | active; server-verified at `/hdd3/kykt26/code/dream3r/`; first KITTI real-data smoke on `2011_09_26_drive_0001_sync_02` window pair (pointmap L2 20.47 = integration evidence, not trained quality) | `code/dream3r/REVIEW_PROMPT.md`, `code/dream3r/RECENT_PROGRESS.md`, `code/dream3r/NEXT_PHASE_ROADMAP.md` |
| 3R-mix Chinese survey (Track B) | **wound down 2026-05-14 (route C: arXiv-only)**; 2026-05-14 quality pass added CroCo + MASt3R mechanism + §10 failure modes + `fig:timeline`; 2026-05-15 prose naturalization pass rewrote 10 paragraphs to drop LLM-style enumerated structures, parallel patterns and workflow vocabulary; 18 A4 pages, 44 references, 6 figures (4 TikZ + 2 paper-Fig.1 composites), 5 booktabs tables, 0 LaTeX errors / 0 warnings; deliberately decoupled from Dream/KYKT internal vocabulary | `Dream/3R-mix/README.md`, `Dream/3R-mix/NEW_CHAT_HANDOFF.md`, `Dream/3R-mix/main.tex`, `Dream/3R-mix/deliverables/3r_survey_stage_final_2026-05-15_natural.pdf` |
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
| Work risk register | v1.2 active (cycle 036 +3 proposal-cycle rows R-PROP-VOCAB-1 / R-PROP-CLAIM-1 / R-PROP-SYNC-1 appended after v1.1 cycle 035 +4 cross-spec rows R-OOD-1 / R-EXT-PRIOR-1 / R-4DGS-LIC-1 / R-INPUT-EXT-1); consolidates per-spec + cross-spec + proposal-cycle risks | `planning/WORK_RISK_REGISTER.md` |
| Dream3R 开题报告 dual-draft (Track C) | active; cycle 036 kickoff — § 1 ~1800 内 + ~1500 外 字 complete as style-and-vocabulary alignment proof; § 2-§ 9 placeholders for cycle 037+; G2 + G3a + G3b + G4 vocab firewall + over-claim greps all 0 hits; 外部稿 uses 代号 "候选架构 X" / "本研究架构" (no raw "Dream3R" in main text) | `planning/proposal_dream3r/OUTLINE_V1.md` + `STYLE_CONTRACT.md` + `DRAFT_INTERNAL_V1.md` + `DRAFT_EXTERNAL_V1.md` |
| Track B advisor submission packaging | active; cycle 036 packaging — Chinese cover note (~600 字, G2 vocab-clean) + submission record with recipient / channel / SHA256 / submitted_at slots + Track A relationship internal meta (not delivered to advisor); actual submission action (email / IM / portal / offline) is post-cycle user action | `3R-mix/deliverables/SUBMISSION_PACKAGE_ADVISOR_2026-05-16.md` + `SUBMISSION_RECORD_2026-05-16.md` + `RELATION_TO_TRACK_A_2026-05-16.md` |
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

Cycle 036 closeout recommendation (cycle 036 closed 2026-05-16; Part A 3 advisor submission packaging files + Part B 4 Dream3R 开题报告 dual-draft kickoff files + WORK_RISK_REGISTER v1.2 + cycle log + sync chain landed; Track A architecture-first mainline unchanged; Track B 3R-mix wound down 2026-05-14 + 2026-05-15 prose naturalization deliverable remains the canonical PDF for the actual submission):

```text
Cycle 036 landed 13 file ops (9 NEW + 4 MODIFIED) per
DEC-20260516-001. Next admissible direction (pick at most one to
launch next; A / C / D / E / F each require their own DEC; B is a
manual user action):

  A. Launch cycle 037 to draft § 2 国内外研究现状 in both
     DRAFT_INTERNAL_V1.md and DRAFT_EXTERNAL_V1.md (recommended
     starting point: § 2 is the largest single-section block
     ~3500 外 + ~4000 内 字, most heavily reuses Track B 综述
     §1-§10 + 5 tables + 6 figures + 44 references, and double-
     draft sync stress-tests the STYLE_CONTRACT vocabulary
     substitution table immediately; per OUTLINE_V1.md §4 drafting
     order).

  B. User executes the actual survey submission action (email /
     IM / portal / offline) and fills the SUBMISSION_RECORD_2026-
     05-16.md slots (recipient / channel / pdf_sha256 via
     PowerShell Get-FileHash / submitted_by / contact /
     submitted_at). This is a manual user action outside any
     cycle. Cover note + submission record + Track A relationship
     meta are all packaging-ready.

  C. Revise OUTLINE_V1.md chapter structure before cycle 037
     (merge / split sections, reorder, change 字数 distribution).
     Per STYLE_CONTRACT §6 sync log discipline, OUTLINE_V1 is
     preserved and OUTLINE_V2 created. Useful if you want to
     change the 9-section default to something tighter.

  D. Pause and revise the cycle 036 deliverables based on
     quality review — e.g., adjust the §1 风格样本 if it doesn't
     read right; adjust the 22-row vocabulary substitution table;
     adjust the candidate-not-final 句式 contrast 表.

  E. Return to architecture-first mainline non-proposal work
     (W22 visualization pack / W23 expert adapter loading
     prerequisites / Fast3R `omegaconf` resolution per cycle 035
     §Next Direction option D). Each independent DEC.

  F. Launch one of the cycle 035 §Next Direction options A-C
     (calibration run on KITTI per CRITIC_CALIBRATION_PLAN_V1 /
     long-seq ablation per LONG_SEQ_REAL_TABLE_PLAN / v0.4 spec
     delta drafting B1/B2/B3) instead — cycle 036 close does not
     preempt those candidates. Each independent DEC.
```

Track A architecture-first remains the mainline per DEC-20260506-001. The proposal upstream (SURVEY_DRIVEN_OPTIMIZATION_PROPOSAL.md) remains `draft, awaiting user review`; cycle 036 consumed it as planning input feeding OUTLINE_V1 §2 §5 §8 §9 素材, not as ratified v0.4 spec design. The Dream3R 开题报告 itself is candidate-not-final per DEC-20260501-011 — the 外部稿 uses 代号 "候选架构 X" / "本研究架构" specifically to encode this status into the academic language. The actual Track B submission is the user's post-cycle action; cycle 036 only delivered the packaging.

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
