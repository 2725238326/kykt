# Dream Research Workspace

Last updated: 2026-05-01

## Purpose

`Dream` is the research workspace for the next-stage KYKT 3R / visual-geometry agenda.

The goal is to build an **architecture-first 3R research engine** that can continuously absorb:

- new 3R papers and model families
- new neural architectures such as SSM/Mamba, memory models, residual attention, test-time compute, continual learning, and RL
- useful GitHub projects that have not yet been applied to 3R
- demo ideas that can be integrated into the KYKT app

The workspace should eventually produce:

1. a large master research prompt
2. research skill/rules for repeated research-agent use
3. a teacher-facing demo and proposal blueprint
4. candidate model/app integration plans for KYKT

Canonical agent entry prompt:

```text
E:\kykt\Dream\AGENT_MASTER_PROMPT.md
```

## Current Direction

Primary direction:

```text
Architecture-first 3R research, with demo and KYKT app integration as required output surfaces.
```

The current strategy is **not** to prematurely choose one method such as Mamba-3R, Event-DUSt3R, or SplatBridge-4D.

Instead, Dream should first build a systematic research engine that can compare and synthesize:

- Memory / State models
- 3R model composition
- Test-time reasoning and self-correction
- Continual / lifelong spatial learning
- Cross-modal and new sensor extensions
- System demo paths that can surprise a teacher while staying feasible

## File Map

- `RESEARCH_STATE.md`: current decisions, assumptions, and open questions.
- `AGENT_MASTER_PROMPT.md`: canonical operating prompt for future Dream agents.
- `QUESTION_LOG.md`: interview history and next questions.
- `RESEARCH_PARADIGM.md`: the operating paradigm, research loop, evidence ladder, and user-discussion gates.
- `RESEARCH_WORKFLOW.md`: operational workflow from source intake to implementation decision.
- `RESEARCH_DATA_MODEL.md`: schema for sources, mechanisms, research units, decisions, and experiment plans.
- `WORKFLOW_STATUS.md`: current workflow phase, active workstreams, and blocked decisions.
- `PHASE1_RESEARCH_PLAN.md`: concrete plan for the first comprehensive research route survey.
- `PHASE1_EXECUTION_LOG.md`: running log for actual Phase 1 research execution.
- `FRONTIER_SOURCE_MAP.md`: verified and pending source map for papers/projects.
- `RESEARCH_UNIT_BANK.md`: structured Dream Research Units.
- `IDEA_SCOREBOARD.md`: score table for candidate ideas.
- `MINIMAL_DEMO_CANDIDATES.md`: teacher-demo candidate analysis.
- `REPRODUCTION_READINESS_MATRIX.md`: repo-level smoke-test and KYKT integration readiness notes.
- `PHASE1_DECISION_MEMO.md`: preliminary synthesis and Phase 2 decision gates.
- `MASTER_RESEARCH_PROMPT_DRAFT.md`: historical early draft; do not use as the operating prompt.
- `RESEARCH_SKILL_RULES_DRAFT.md`: evolving rules for a project skill and future Codex skill.

## Directory Map

- `registry/`: lightweight indexes for sources, research units, and decisions.
- `cycles/`: per-cycle research logs.
- `decisions/`: decision memos that require commitment or deferral.
- `experiments/`: experiment plans; a file here does not mean the experiment has been run.
- `templates/`: reusable forms for sources, research units, decisions, cycles, and experiments.

## Working Loop

Use this loop after each discussion:

1. Update `RESEARCH_STATE.md` with decisions.
2. Update `QUESTION_LOG.md` with the question/answer trail.
3. Update `RESEARCH_PARADIGM.md` when the operating model or decision gates change.
4. Refine `AGENT_MASTER_PROMPT.md` when the operating prompt, load protocol, phase, or decision gates change.
5. Refine `RESEARCH_SKILL_RULES_DRAFT.md` when we learn a reusable rule.
6. Later, split stable rules into:
   - a project-local version under `E:\kykt\Dream`
   - a reusable Codex skill

## Current Operating Mode

Dream starts with a balanced two-track plan:

```text
Breadth Map + Minimal Demo
```

The breadth track discovers and scores architecture mechanisms. The demo track keeps one small teacher-facing proof path alive so the work stays concrete.

Current operational phase:

```text
Phase 1.5: Research Workflow Deployment
```

This means:

- no model reproduction yet
- no heavy checkpoint downloads yet
- no KYKT app navigation changes yet
- deploy the research process first

Current preliminary thesis candidate:

```text
Dream3R: Geometry-Governed State and Test-Time Reasoning for Long-Context 3R
```

This is not a final commitment. It is the best current umbrella for comparing memory/state, routing, geometry critics, dynamic 4D pointmaps, and reproducible demo paths.

## Non-Negotiables

- Keep the work grounded in 3R / visual geometry, not generic AI trend collection.
- Favor architecture-level novelty over pure application packaging.
- Require some path to a convincing demo.
- Require some path to KYKT app integration.
- Keep engineering cost controlled unless a specific experiment justifies going heavier.
- Separate evidence from speculation.
- Avoid claiming a method works before a minimal experiment or defensible proxy exists.
- Do not move from planned experiment to actual reproduction without a user decision.
