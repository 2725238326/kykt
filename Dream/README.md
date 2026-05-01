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
- `QUESTION_LOG.md`: interview history and next questions.
- `MASTER_RESEARCH_PROMPT_DRAFT.md`: evolving master prompt for research agents.
- `RESEARCH_SKILL_RULES_DRAFT.md`: evolving rules for a project skill and future Codex skill.

## Working Loop

Use this loop after each discussion:

1. Update `RESEARCH_STATE.md` with decisions.
2. Update `QUESTION_LOG.md` with the question/answer trail.
3. Refine `MASTER_RESEARCH_PROMPT_DRAFT.md` when the research scope changes.
4. Refine `RESEARCH_SKILL_RULES_DRAFT.md` when we learn a reusable rule.
5. Later, split stable rules into:
   - a project-local version under `E:\kykt\Dream`
   - a reusable Codex skill

## Non-Negotiables

- Keep the work grounded in 3R / visual geometry, not generic AI trend collection.
- Favor architecture-level novelty over pure application packaging.
- Require some path to a convincing demo.
- Require some path to KYKT app integration.
- Keep engineering cost controlled unless a specific experiment justifies going heavier.
- Separate evidence from speculation.
- Avoid claiming a method works before a minimal experiment or defensible proxy exists.

