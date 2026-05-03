# Dream Workflow Status

Last updated: 2026-05-03 (cycle 008 drafted three finalist mechanism specs after user-approved option B)

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
| Cycle logs | active | `cycles/CYCLE-20260503-002.md` |
| Experiment planning | seeded | `experiments/EXP-20260501-001-dust3r-splatt3r-smoke-plan.md` |
| Agent master prompt | active | `AGENT_MASTER_PROMPT.md` |
| Research content roadmap | active | `paradigm/RESEARCH_CONTENT_ROADMAP.md` |
| Multi-track research canvas | active | `planning/MULTI_TRACK_RESEARCH_CANVAS.md` |
| Research graph / paper start | active | `planning/RESEARCH_GRAPH_AND_PAPER_START.md` |
| Branch comparison matrix | filled first comparative pass | `planning/BRANCH_COMPARISON_MATRIX.md` |
| Branch shortlist decision surface | user approved option B (cycle 008) | `planning/BRANCH_SHORTLIST_DECISION_SURFACE.md` |
| Architecture mechanism intake | first-pass active | `planning/ARCHITECTURE_MECHANISM_INTAKE.md` |
| Action taxonomy / proxy metrics | first compact pass | `planning/ACTION_TAXONOMY_AND_PROXY_METRICS.md` |
| Proxy case-card template | active form, populated in cycle 009 | `templates/proxy_case_card.md` |
| Finalist mechanism spec template | populated for three finalists in cycle 008 | `templates/finalist_mechanism_spec.md` |
| Geometry Critic finalist spec | draft (L1); L2 case cards next cycle | `specs/SPEC-20260503-001-geometry-critic.md` |
| Executive Memory finalist spec | draft (L1); L2 case cards next cycle | `specs/SPEC-20260503-002-executive-memory.md` |
| Dynamic Object Permanence finalist spec | draft (L1); L2 case cards next cycle | `specs/SPEC-20260503-003-dynamic-object-permanence.md` |
| Source mining (cycle 005 pass) | complete for visual priors, depth priors, active perception, event VO | `sources/FRONTIER_SOURCE_MAP.md` (Cycle 005 Source Mining Pass section) |
| Workspace reorganization (cycle 006) | complete; topical subdirectories + archive/ + INDEX.md | `cycles/CYCLE-20260502-006.md` |
| Research & code discipline (cycle 007) | active rulebook for synthesis behavior and Dream-driven code | `paradigm/RESEARCH_CODE_DISCIPLINE.md` |
| Finalist shortlist approval (cycle 008) | user-approved option B; three finalist specs drafted | `decisions/DEC-20260503-002-finalist-shortlist-approval.md` |
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

Cycle 008 closed the shortlist gate (option B approved) and drafted three finalist mechanism specs under `specs/`. The next user-facing decisions are:

```text
1. Which spec gets the first L2 case-card pass in cycle 009? Default: Critic (cheapest proxy, immediate L2 yes/no). Alternative: Permanence if you want the visual demo earliest.
2. Composer L1 vs L2 framing: draft Composer as a fourth spec once one of the three finalists clears its first case card, or keep Composer as undocumented support until then?
3. First teacher-facing demo target: Critic-on-existing-jobs timeline, Memory simulation panel, or Permanence object-track overlay on MonST3R sample?
4. Annotation cost ceiling for cycle 009 (estimated 1-4 hours per spec; Permanence has a hard 60-minute fail-fast).
```

These four points are also the explicit "next discussion points" carried inside `cycles/CYCLE-20260503-002.md` and at the end of each finalist spec.

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

When Dream creates or promotes a workflow artifact, update the relevant guidance files in the same pass:

- `AGENT_MASTER_PROMPT.md`
- `README.md`
- `WORKFLOW_STATUS.md`
- `RESEARCH_STATE.md`
- current cycle log under `cycles/`
