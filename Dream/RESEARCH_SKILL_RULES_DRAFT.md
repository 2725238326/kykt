# Research Skill / Rules Draft

Last updated: 2026-05-01

This draft will later become:

1. a project-local Dream research rulebook
2. possibly a reusable Codex skill

## Candidate Skill Name

```text
dream-3r-research
```

## Candidate Skill Description

Use when working on Dream / KYKT 3R research: frontier 3R model architecture ideation, paper and GitHub mining, architecture synthesis, demo planning, KYKT app integration planning, and teacher-facing research proposal construction.

## Core Behavior

When this skill is active, the agent should:

1. prioritize architecture-level mechanisms over surface-level application ideas
2. map every external idea back to a concrete 3R bottleneck
3. separate evidence from speculation
4. keep engineering cost visible
5. always propose a minimal experiment or demo path
6. maintain a connection to KYKT app integration
7. run Dream as a two-track process: Breadth Map + Minimal Demo

## User-Discussion Gates

Discuss with the user before:

- choosing the primary research thesis
- committing to one architecture family
- starting heavy training or large data construction
- making major KYKT app information-architecture changes
- packaging the draft rules as a reusable Codex skill
- declaring a teacher demo ready
- discarding a major candidate track

Proceed without interruption for lightweight note updates, small prompt refinements, source triage, and mock artifacts.

## Research Intake Rules

For each paper/project:

- Record title, source, year, code availability, and task.
- Extract the mechanism, not only the result.
- Ask: what does this change about memory, compute, geometry, dynamics, sensing, or adaptation?
- Ask: what would happen if this mechanism were inserted into a 3R pipeline?
- Reject items that cannot be translated into a 3R hypothesis.

## GitHub Mining Rules

Search broadly but filter aggressively.

Allowed sources:

- 3R / 3D reconstruction repos
- visual geometry foundation models
- Mamba / SSM / long-context architectures
- residual attention and efficient attention mechanisms
- continual learning
- RL / active perception
- 4DGS / dynamic scene representations
- event camera / sensor fusion

Reject or defer:

- generic LLM tools with no spatial mechanism
- repos with no working code unless the idea is unusually strong
- projects requiring heavy training beyond current resources unless they can be reduced to a small experiment

## Innovation Rules

A good Dream idea should satisfy at least three:

- changes the architecture, not only the UI or dataset
- attacks a known 3R bottleneck
- can be demonstrated with existing or lightly modified code
- produces a clear teacher-facing visual
- can be integrated into KYKT as a lane, runner, report, or comparison surface
- has a plausible paper narrative

## Evidence Ladder

Use the cheapest adequate evidence.

Level 1:

- architecture diagram
- pseudo-code
- synthetic or mock demo

Level 2:

- existing model outputs compared on several KYKT samples
- qualitative screenshots
- failure-case analysis

Level 3:

- modified code path
- prototype module
- small runner integration

Level 4:

- quantitative metrics
- ablation
- runtime/memory analysis

The agent should recommend an evidence level rather than force all ideas to Level 4.

## KYKT Integration Rules

Every candidate should map to at least one:

- `model_registry.py`
- backend runner
- `scene_meta.json` output contract
- Sample Matrix comparison
- Advisor/report workflow
- Overview research lane
- System/deployment readiness view

If no integration path exists, classify as background research.

## Anti-Slop Rules

Do not produce only buzzword combinations.

Bad:

```text
Mamba + 3R + world model will solve long video.
```

Better:

```text
Replace temporal attention with a geometry-gated selective state update. The gate is driven by reprojection residual, confidence, dynamic mask, and baseline. The GPU recurrent state remains fixed-size, while scene memory is stored in an external sparse map.
```

## Convergence Rules

Every batch of research should end with:

- one near-term demo candidate
- one architecture candidate
- one long-term speculative candidate
- one rejected/deferred category with reasons

Do not keep expanding the idea pool without pruning.

## Future Skill Packaging

When stable, create both:

- project-local rules under `E:\kykt\Dream`
- reusable Codex skill under the Codex skills directory

The reusable skill should stay concise and move long examples into references.
