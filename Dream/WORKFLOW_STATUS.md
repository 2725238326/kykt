# Dream Workflow Status

Last updated: 2026-05-02 (cycle 005)

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
E:\kykt\Dream\FRONTEND_DESIGN_HANDOFF_PROMPT.md
```

Use this prompt when preparing KYKT frontend design work for Gemini CLI.

## Active Workstreams

| Workstream | Status | Next artifact |
|---|---|---|
| Research workflow | active | `RESEARCH_WORKFLOW.md` |
| Collaboration roadmap | active | `COLLABORATION_ROADMAP.md` |
| Data model | active | `RESEARCH_DATA_MODEL.md` |
| Source registry | seeded | `registry/source_registry.md` |
| Research unit registry | seeded | `registry/research_unit_registry.md` |
| Decision registry | seeded | `registry/decision_registry.md` |
| Cycle logs | active | `cycles/CYCLE-20260502-005.md` |
| Experiment planning | seeded | `experiments/EXP-20260501-001-dust3r-splatt3r-smoke-plan.md` |
| Agent master prompt | active | `AGENT_MASTER_PROMPT.md` |
| Research content roadmap | active | `RESEARCH_CONTENT_ROADMAP.md` |
| Multi-track research canvas | active | `MULTI_TRACK_RESEARCH_CANVAS.md` |
| Research graph / paper start | active | `RESEARCH_GRAPH_AND_PAPER_START.md` |
| Branch comparison matrix | filled first comparative pass | `BRANCH_COMPARISON_MATRIX.md` |
| Branch shortlist decision surface | draft ready for user decision | `BRANCH_SHORTLIST_DECISION_SURFACE.md` |
| Architecture mechanism intake | first-pass active | `ARCHITECTURE_MECHANISM_INTAKE.md` |
| Action taxonomy / proxy metrics | first compact pass | `ACTION_TAXONOMY_AND_PROXY_METRICS.md` |
| Proxy case-card template | active form, not yet populated | `templates/proxy_case_card.md` |
| Finalist mechanism spec template | active form, blocked on user shortlist approval | `templates/finalist_mechanism_spec.md` |
| Source mining (cycle 005 pass) | complete for visual priors, depth priors, active perception, event VO | `FRONTIER_SOURCE_MAP.md` (Cycle 005 Source Mining Pass section) |
| Frontend handoff prompt | active | `FRONTEND_DESIGN_HANDOFF_PROMPT.md` |
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

Choose finalist branches from `BRANCH_SHORTLIST_DECISION_SURFACE.md`:

```text
A. Draft mechanism specs for Geometry Critic + Executive Memory, Composer as support.
B. Add Dynamic Object Permanence as third finalist before mechanism specs.
C. Keep all six branches; first fill proxy case-card instances using `templates/proxy_case_card.md`.
D. Mine more sources (cycle 005 completed one pass; a second pass is only needed if a chosen branch is blocked on a specific missing comparator).
```

Current recommendation:

```text
Return to A / B / C. Cycle 005 added primary-source anchors for the previously weakest areas (visual priors, depth priors, active perception, event VO). Templates for proxy case cards and finalist mechanism specs are ready under `templates/`. Another round of mining (D again) should only run if a specific missing comparator blocks a chosen branch.
```

## Guidance File Sync Rule

When Dream creates or promotes a workflow artifact, update the relevant guidance files in the same pass:

- `AGENT_MASTER_PROMPT.md`
- `README.md`
- `WORKFLOW_STATUS.md`
- `RESEARCH_STATE.md`
- current cycle log under `cycles/`
