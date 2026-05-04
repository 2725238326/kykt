# Dream Workflow Status

Last updated: 2026-05-05 (cycle 013 fully closed: 8 newly-mined sources (7 new SRC IDs SRC-2026-009..015) + paper related-work prose draft (skeleton -> prose; Sections 1-7 prose, Sections 8-9 drafted) + 4 L3 prerequisites briefs (EXP-20260505-001..004) per DEC-20260505-003; v2.1 unchanged; G2 unchanged; G7 paper-related-work-prose-readiness new at inferred-with-prose-draft-anchor; D3 = Critic per cycle 011 unchanged)

## Current Phase

```text
Phase 1.5: Research Workflow Deployment
```

## Current Mode

```text
No reproduction yet.
No heavy installs.
No KYKT app redesign.
Research content discovery and thesis validation are the mainline.
Backend/research pipeline work is support infrastructure.
Frontend implementation is delegated to Gemini CLI / designated frontend agent.
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
| Cycle logs | active | `cycles/CYCLE-20260505-004.md` (cycle 013 closed; Phase 2 prep + research mining) |
| Experiment planning | active; cycle 013 added 4 L3 prerequisite briefs (one per finalist; brief-only, NOT L3 authorization) | `experiments/EXP-20260505-001..004-l3-prerequisites-{critic\|memory\|permanence\|composer}.md` |
| Agent master prompt | active | `AGENT_MASTER_PROMPT.md` |
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
| 3R Composer finalist spec | draft (L1); L2 case cards drafted in cycle 009 (paper-derived); CASE-COMPOSER-03 v2 row promoted to canonical per DEC-20260504-004; CASE-COMPOSER-04 KYKT-metadata-derived added in cycle 012 (advances G2 inferred -> inferred-with-real-inventory-anchor; G2 NOT closed); demo storyboard `STORY-20260505-004-composer.md` drafted cycle 012 | `specs/SPEC-20260504-001-3r-composer.md` + `cases/CASE-20260505-COMPOSER-01..04.md` + `storyboards/STORY-20260505-004-composer.md` |
| Cross-spec signal contract | **v2.1 active** (per DEC-20260505-001): additive revision over v2 — adds "Forward-reference null protocol" subsection formalizing the pattern exercised by cycle-009 + cycle-010 cards; v2 substance unchanged (alpha = 0.5 inferred; signal owner table; CR-1..CR-6; cost_adjusted_match; route_regret cost-typed). v1 + v2 prose preserved. Cycle 011 G5 closed by this revision; cycle 010 G4 closed-by-documentation under the protocol. v2 -> v3 candidates 8x8 grid partition + identity_consistency threshold pinning deferred. | `paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md` + `decisions/DEC-20260505-001-cycle-011-launch-and-d3-demo-target.md` |
| Literature guidance board | v1 active; post-cycle-013 SPINE refresh fold-in done (tttLRM -> SPINE_MEMORY + SPINE_CRITIC; VGGT + MapAnything + DUSt3R-MASt3R-VGGT MVS evaluation + awesome-dust3r -> SPINE_COMPOSER; Julian Ost AAAI-2026 -> SPINE_PERMANENCE advanced; CRITICAL_NOTES gained 3 new deconfusion entries for Julian Ost name-collision / tttLRM vs Test3R / VGGT capability-card gap). PAPER_RELATED_WORK_SKELETON.md upgraded cycle 013 to prose draft (Sections 1-7 prose; Sections 8-9 drafted; filename retained per Surgical Edits) | `literature/INDEX.md` + four `literature/SPINE_*.md` + `literature/CRITICAL_NOTES.md` + `literature/PAPER_RELATED_WORK_SKELETON.md` |
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

- first local reproduction target
- large checkpoint downloads
- KYKT Dream page or navigation change
- Codex direct frontend implementation
- major Gemini CLI frontend redesign instruction
- final thesis selection
- deepening any single thesis branch as the default path
- reusable Codex skill packaging

## Recommended Next User Decision

Cycle 013 fully closed (S2..S7 done in `cycles/CYCLE-20260505-004.md`). Cycle 013 was a Phase 2 preparation + research mining cycle per DEC-20260505-003 (user delegated planning: "好了，请你做实际的研究部署吧" + clarification "Phase 2 准备 + 资料调研"). Three locked sub-passes:

```text
(S2) Source mining: 8 newly-mined sources; 7 new SRC IDs SRC-2026-009
     ..015. Targets cycle-013 coverage gaps.
(S3) Paper related-work prose draft: literature/PAPER_RELATED_WORK_
     SKELETON.md upgraded from skeleton to prose draft. Sections 1-7
     are prose anchored to L2 case cards + SRC-* IDs; Sections 8-9
     drafted as prose enumerating non-claims (7) + parallel additions
     (4 finalists + integrating layer + L2 case-card methodology).
(S4) 4 L3 prerequisites briefs under experiments/ (Critic / Memory /
     Permanence / Composer). Each lists repos + checkpoints + GPU /
     disk / time budget (all `inferred`, not measured) + smoke-test
     path + minimum code change. Brief-only, NOT L3 authorization.
```

Cycle 014 launch will need user decisions on:

```text
1. Cycle 014 scope options (pick one or hold):
   (a) close G6 — gated; L3 prototype on Memory-equipped backbone.
       Path now inventoried in EXP-20260505-002. Authorization
       required for clone / download / install / run per
       AGENT_MASTER_PROMPT.md section 6.
   (b) close G2 — gated; measured route_regret. Path now inventoried
       in EXP-20260505-004. Same authorization required.
   (c) close one of the other inventoried L3 paths (Critic
       EXP-20260505-001 or Permanence EXP-20260505-003). Same
       authorization.
   (d) request demo show authorization for one finalist (separate
       per-finalist DEC; reviewer pass on chosen storyboard happens
       in that DEC's drafting; no storyboard advances past `draft`
       without the DEC).
   (e) Phase 2 paper writing continuation: Sections 1-7 are prose;
       Sections 8-9 are drafted. Full paper (intro / methods /
       results / discussion) requires user direction on venue /
       length / scope.
   (f) hold; no cycle 014 action; archive current state. Acceptable
       outcome — current state has 4-finalist L2 portfolio (13
       cards) + 4-finalist storyboard portfolio (4 storyboards,
       draft) + KYKT-metadata anchor on Composer + paper related-
       work prose draft + 4 L3 prerequisites briefs + 7 newly-mined
       sources is a coherent stop-state.

2. SPINE refresh fold-in: cycle-013 mining surfaced new entries that
   should fold into SPINE files at next refresh (Mem3R into
   SPINE_MEMORY required; tttLRM into SPINE_CRITIC + SPINE_MEMORY;
   MapAnything + DUSt3R-MASt3R-VGGT MVS evaluation + VGGT into
   SPINE_COMPOSER; Julian Ost into SPINE_PERMANENCE advanced +
   CRITICAL_NOTES.md deconfusion). Markdown only. User direction on
   whether this is a cycle-014 line item or rolled into a larger
   cycle.

3. v2.2 candidates: still none surfaced. Both cycle-011 deferred
   candidates (8x8 grid partition; identity_consistency threshold)
   remain deferred until measured anchors exist. VGGT capability-
   card gap is per-card, not contract.

4. D3 first demo target: Critic remains locked per
   DEC-20260505-001. No change in cycle 013. Reconsideration option
   (keep / reconsider / hold) remains open.

5. Blocked items: showing any of the 4 demo storyboards is explicitly
   blocked until per-finalist showing-authorization DEC; rest
   unchanged from cycle 012 closeout.
```

Still blocked on user approval (one extension from cycle 011):

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
