# Dream Workflow Status

Last updated: 2026-05-01

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
| Cycle logs | seeded | `cycles/CYCLE-20260501-001.md` |
| Experiment planning | seeded | `experiments/EXP-20260501-001-dust3r-splatt3r-smoke-plan.md` |
| Agent master prompt | active | `AGENT_MASTER_PROMPT.md` |
| Research content roadmap | active | `RESEARCH_CONTENT_ROADMAP.md` |
| Multi-track research canvas | active | `MULTI_TRACK_RESEARCH_CANVAS.md` |
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
- reusable Codex skill packaging

## Recommended Next User Decision

Choose the next workflow lane:

```text
A. Start a research content cycle: Dream3R thesis validation plus new mechanism discovery.
B. Mine frontier sources and GitHub projects for mechanisms that can become 3R hypotheses.
C. Build a backend research pipeline contract only as support for selected research content.
D. Prepare Phase 2 smoke-test plan in more detail, still no install.
E. Prepare a Gemini CLI frontend handoff only after the research content and support contract are clear.
```

Current recommendation:

```text
Do a multi-track branch comparison before deepening GEM-3R or any single mechanism.
```
