# Dream Workflow Status

Last updated: 2026-05-04 (cycle 008.5 closeout + planning-layer sync + inventory-sync sub-pass + TASK_SNAPSHOT.md introduced as mandatory-load #1 + cycle 009 case-card portfolio closeout: 6 case cards drafted under D1' parallel + D2' paper-derived; cross-spec contract usage audit recorded; v1 -> v2 cost-typed route_regret candidate surfaced; 3 contract gaps recorded for cycle 010)

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
| Source registry | seeded | `registry/source_registry.md` |
| Research unit registry | seeded | `registry/research_unit_registry.md` |
| Decision registry | seeded | `registry/decision_registry.md` |
| Cycle logs | active | `cycles/CYCLE-20260505-001.md` |
| Experiment planning | seeded | `experiments/EXP-20260501-001-dust3r-splatt3r-smoke-plan.md` |
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
| Geometry Critic finalist spec | draft (L1); L2 case cards drafted in cycle 009 (paper-derived) | `specs/SPEC-20260503-001-geometry-critic.md` + `cases/CASE-20260504-CRITIC-01..03.md` |
| Executive Memory finalist spec | draft (L1); L2 case cards scheduled for cycle 010 | `specs/SPEC-20260503-002-executive-memory.md` |
| Dynamic Object Permanence finalist spec | draft (L1); L2 case cards scheduled for cycle 010 | `specs/SPEC-20260503-003-dynamic-object-permanence.md` |
| 3R Composer finalist spec | draft (L1); L2 case cards drafted in cycle 009 (paper-derived); v1 -> v2 cost-typed route_regret candidate surfaced from CASE-COMPOSER-03 | `specs/SPEC-20260504-001-3r-composer.md` + `cases/CASE-20260505-COMPOSER-01..03.md` |
| Cross-spec signal contract | v1 active; first exercise complete in cycle 009 case cards (CR-1 closure via CRITIC-02 <-> COMPOSER-01 cross-pair; CR-3 forward-reference shape; CR-2 zero coverage recorded as gap G1) | `paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md` + `cycles/CYCLE-20260505-001.md` "Contract Usage Audit (S6)" |
| Literature guidance board | v1 active (INDEX + four SPINE + CRITICAL_NOTES + PAPER_RELATED_WORK_SKELETON) | `literature/INDEX.md` |
| Work risk register | v1 active; consolidates per-spec + cross-spec risks | `planning/WORK_RISK_REGISTER.md` |
| Demo storyboard template | active form; populated per finalist after demo authorization | `templates/demo_storyboard.md` |
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

Cycle 009 closed the case-card portfolio for Critic + Composer (6 cards under D1' parallel + D2' paper-derived). The cycle 008.5 follow-up gates D1 / D2 / D4 are locked; D3 remains deferred per `decisions/DEC-20260504-002-no-all-in-on-single-finalist.md`.

The cycle 009 closeout (`cycles/CYCLE-20260505-001.md` "Contract Usage Audit (S6)" + "Closeout (S7)") surfaces four cycle-010 launch decisions for the user:

```text
1. v2 cost-typed route_regret (CASE-COMPOSER-03 surfaced this candidate)
   - adopt as v2 (regime-typed AND cost-typed; CR-4 starts firing routinely
     in cost-asymmetric regimes; route_recommendation can flip vs v1)
   - defer (keep v1; cost-axis remains an open research direction)
   - fold into v1.x (treat the cost_normalized axis as a v1 documentation
     extension rather than a versioned change)
2. Cycle 010 ordering: Memory + Permanence cards in parallel, or Memory
   first (Permanence's suppress_static_write originates from Permanence-
   side, so Memory-first means CR-2 binding is read across cycle
   boundaries).
3. D3 (first teacher demo target): still deferrable. Cycle 009 paper-
   derived case cards do not force D3; the user can lift the deferral
   if desired, or keep it deferred until cycle 010 case-card data
   exists.
4. paradigm/TEACHER_AUDIENCE_PROFILE.md remaining sub-fields (Prior
   Expectations / Demo Precedent / Previously Criticized) are still
   empty by user choice. Populate, or keep empty (agent will not
   invent).
```

Contract gaps recorded for cycle-010 closure (full list in `cycles/CYCLE-20260505-001.md` "Contract Usage Audit (S6)"):

```text
G1: CR-2 (Permanence suppress_static_write binding on Memory) has zero
    coverage in cycle 009 by design (no Memory or Permanence card).
    Cycle 010 Memory + Permanence cards are the closure path.
G2: tau_spread = 0.05 in CR-1 closure is inferred, not measured.
    Demo-observed grounding is a future-cycle question.
G3: CR-4 only routinely exercises under v2 adoption; under v1 alone,
    the cycle-009 portfolio shows no live tie to resolve.
```

Still blocked on user approval (unchanged from prior cycles):

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

## Guidance File Sync Rule

When Dream creates or promotes a workflow artifact, update the relevant guidance files in the same pass. **`TASK_SNAPSHOT.md` updates first in this chain** so that a sync interrupted partway through still leaves a valid resume pointer:

- `TASK_SNAPSHOT.md` (highest-authority resume pointer; updated first; see its own "Update protocol" section for transitions)
- `AGENT_MASTER_PROMPT.md`
- `README.md`
- `WORKFLOW_STATUS.md`
- `RESEARCH_STATE.md`
- current cycle log under `cycles/`
