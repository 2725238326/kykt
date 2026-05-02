# Dream Index

Last updated: 2026-05-02 (cycle 006 workspace reorganization)

Quick navigation for humans and agents. For full operating rules, read `AGENT_MASTER_PROMPT.md` first.

## How To Read This Workspace

1. Start with `AGENT_MASTER_PROMPT.md` (entry prompt with mandatory load protocol).
2. Check `WORKFLOW_STATUS.md` for current phase, blocked decisions, and recommended next user decision.
3. Check `RESEARCH_STATE.md` for living state and cycle history.
4. Use this `INDEX.md` to find any specific file by topic.

## Root-Level Files (Always-On Entry Points)

| File | Role |
|---|---|
| `README.md` | Workspace overview, purpose, non-negotiables |
| `INDEX.md` | This file; topic-based navigation |
| `AGENT_MASTER_PROMPT.md` | Canonical agent operating prompt + mandatory load protocol |
| `WORKFLOW_STATUS.md` | Current phase, active workstreams, blocked decisions, recommended next user decision |
| `RESEARCH_STATE.md` | Living state log; cycle history; current recommendations |

## Subdirectories By Purpose

### `paradigm/` - How Dream Operates

| File | Role |
|---|---|
| `RESEARCH_PARADIGM.md` | Operating paradigm, two-track plan, evidence ladder, user-discussion gates |
| `RESEARCH_WORKFLOW.md` | Source-to-implementation operational workflow |
| `RESEARCH_DATA_MODEL.md` | Schema for sources, mechanisms, RUs, decisions, experiments |
| `RESEARCH_SKILL_RULES_DRAFT.md` | Evolving rules for a future Codex skill |
| `RESEARCH_CONTENT_ROADMAP.md` | Research-content-first roadmap and axes |

### `planning/` - Active Research Planning

| File | Role |
|---|---|
| `MULTI_TRACK_RESEARCH_CANVAS.md` | Multi-branch comparison canvas (do not collapse to single thesis prematurely) |
| `RESEARCH_GRAPH_AND_PAPER_START.md` | Failure-mode / mechanism / composition graph; paper scaffold |
| `BRANCH_COMPARISON_MATRIX.md` | Branch-level comparison matrix and score table |
| `BRANCH_SHORTLIST_DECISION_SURFACE.md` | User decision surface for choosing 2-3 finalist branches |
| `ARCHITECTURE_MECHANISM_INTAKE.md` | Branch-neutral intake map for broad architecture and visual mechanisms |
| `ACTION_TAXONOMY_AND_PROXY_METRICS.md` | Compact A1-A8 actions and P1-P8 proxy validation protocols |
| `DREAM3R_THESIS_STRESS_TEST.md` | Stress test for the Dream3R / GEM-3R candidate branch |
| `MINIMAL_DEMO_CANDIDATES.md` | Teacher-demo candidate analysis |

### `sources/` - Source Mining

| File | Role |
|---|---|
| `FRONTIER_SOURCE_MAP.md` | Verified and pending source map; cycle-tagged source-mining passes |

### `units/` - Research Units, Scoring, Reproduction Readiness

| File | Role |
|---|---|
| `RESEARCH_UNIT_BANK.md` | Structured Dream Research Units (RU-001..) |
| `IDEA_SCOREBOARD.md` | Score table for candidate ideas |
| `REPRODUCTION_READINESS_MATRIX.md` | Repo-level smoke-test and KYKT integration readiness |

### `handoff/` - Collaboration And Frontend

| File | Role |
|---|---|
| `FRONTEND_DESIGN_HANDOFF_PROMPT.md` | Canonical frontend handoff prompt for Gemini CLI |
| `COLLABORATION_ROADMAP.md` | Human-agent collaboration path and near-term deployment sequence |

### `logs/` - Running Logs

| File | Role |
|---|---|
| `QUESTION_LOG.md` | Interview history and next questions |

### `archive/` - Historical / Superseded

| File | Role |
|---|---|
| `PHASE1_RESEARCH_PLAN.md` | Phase 1 plan (historical; Phase 1.5 is current) |
| `PHASE1_EXECUTION_LOG.md` | Phase 1 running log (historical) |
| `PHASE1_DECISION_MEMO.md` | Phase 1 synthesis and gates (historical) |
| `MASTER_RESEARCH_PROMPT_DRAFT.md` | Superseded by `AGENT_MASTER_PROMPT.md` |

### `cycles/` - Per-Cycle Research Logs

Format: `CYCLE-YYYYMMDD-NNN.md`. Newest is the active cycle.

### `decisions/` - Decision Memos

Format: `DEC-YYYYMMDD-NNN-<slug>.md`. Indexed in `registry/decision_registry.md`.

### `experiments/` - Experiment Plans

Format: `EXP-YYYYMMDD-NNN-<slug>.md`. Filing here does not mean the experiment was run.

### `registry/` - Lightweight Indexes

| File | Role |
|---|---|
| `source_registry.md` | Source ID -> title/url/track/evidence map |
| `research_unit_registry.md` | RU ID -> name/track/decision map |
| `decision_registry.md` | DEC ID -> scope/decision/status map |

### `templates/` - Reusable Forms

| File | Role |
|---|---|
| `source_card.md` | Source intake form |
| `research_unit.md` | Dream Research Unit form |
| `decision_memo.md` | Decision memo form |
| `cycle_log.md` | Cycle log form |
| `experiment_plan.md` | Experiment plan form |
| `frontend_design_handoff.md` | Frontend task brief form for Gemini CLI |
| `proxy_case_card.md` | Branch-neutral L2 proxy case-card form (P1-P8) |
| `finalist_mechanism_spec.md` | Branch-neutral mechanism spec form (requires user approval) |

## Find By Question

| Question | Where to look |
|---|---|
| What phase are we in? | `WORKFLOW_STATUS.md` |
| What is the next user decision? | `WORKFLOW_STATUS.md` -> Recommended Next User Decision; backed by `planning/BRANCH_SHORTLIST_DECISION_SURFACE.md` |
| What sources do we know about? | `sources/FRONTIER_SOURCE_MAP.md` and `registry/source_registry.md` |
| What ideas are on the table? | `units/RESEARCH_UNIT_BANK.md`, `units/IDEA_SCOREBOARD.md` |
| Why this branch and not that one? | `planning/BRANCH_COMPARISON_MATRIX.md`, `planning/BRANCH_SHORTLIST_DECISION_SURFACE.md` |
| What can Dream do without asking the user? | `AGENT_MASTER_PROMPT.md` section 6 |
| What requires user approval? | `AGENT_MASTER_PROMPT.md` section 6 + `WORKFLOW_STATUS.md` Blocked Until User Decision |
| What is the latest research result? | newest file under `cycles/` |
| What did we decide? | `registry/decision_registry.md` and files under `decisions/` |
| What experiments are planned? | files under `experiments/` |
| How should the frontend agent work? | `handoff/FRONTEND_DESIGN_HANDOFF_PROMPT.md` |
| How do humans and agents cooperate? | `handoff/COLLABORATION_ROADMAP.md` |

## Convention Reminders

- Evidence labels: `paper-proven`, `code-observed`, `demo-observed`, `inferred`, `speculative`, `unknown`.
- Decision approval gates: see `AGENT_MASTER_PROMPT.md` section 6.
- ID format: `SRC-YYYY-NNN`, `MECH-YYYY-NNN`, `RU-NNN`, `DEC-YYYYMMDD-NNN`, `EXP-YYYYMMDD-NNN`, `CYCLE-YYYYMMDD-NNN`, `CASE-YYYYMMDD-NNN`, `SPEC-YYYYMMDD-NNN`.
- Guidance file sync rule: when promoting a workflow artifact, also update `AGENT_MASTER_PROMPT.md`, `README.md`, `WORKFLOW_STATUS.md`, `RESEARCH_STATE.md`, current cycle log.
