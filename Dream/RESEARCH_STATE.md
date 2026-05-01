# Dream Research State

Last updated: 2026-05-01

## User Intent

The research should create a large surprise for the teacher:

- ideally a new research direction in 3R / spatial intelligence
- strong enough to become a future paper basis
- concrete enough to produce a visible demo
- practical enough to connect with the existing KYKT app

Top-conference ambition and model-architecture breakthrough are not treated as conflicting goals. The intended direction is:

```text
Architecture novelty + concrete 3R bottleneck + feasible prototype/demo + KYKT integration path.
```

## Current Preferences

### Surprise Form

The user wants all three:

- a new architecture story
- a demo that can be shown to the teacher
- a proposal-quality research narrative

No single form is fixed yet.

### Architecture Taste

The user is open to a combined exploration of:

- Memory / State direction: Mamba, SSM, long-term spatial memory, streaming 3R.
- Composer direction: extract and combine strengths from MASt3R, MonST3R, Fast3R, Spann3R, CUT3R, and related 3R models.
- Reasoning / Test-time Compute direction: geometry self-checking, iterative correction, hard-case reasoning.
- Continual / Lifelong direction: online adaptation, long-sequence memory, no catastrophic forgetting.
- Cross-modal direction remains allowed, but architecture-first remains the priority.

### GitHub / Paper Mining Scope

The scope should be adaptive rather than absolute:

- 3R / 3D vision repos are the core.
- New architecture, attention, RL, continual learning, and vision projects can be mined if they can be converted into a 3R hypothesis.
- Do not collect unrelated AI trends unless a 3R mechanism can be stated.

### KYKT App Integration

The user wants all of these eventually:

- research idea lane
- fast reproduction lane
- new model prototype lane
- paper/proposal workbench lane
- a separate management area for long-running research work

This management area will be discussed later.

Frontend implementation boundary:

- KYKT app frontend design / implementation remains owned by Gemini CLI or another designated frontend implementation agent.
- Dream / Codex should define the task, constraints, handoff prompt, and acceptance criteria.
- Dream / Codex should not edit frontend code by default unless the user explicitly asks for direct frontend implementation.

### Evidence Standard

No single evidence level is fixed.

Use an evidence ladder:

1. architecture diagram + pseudo-code + small mock demo
2. existing model outputs compared on several samples
3. modified code path or prototype module
4. small-scale quantitative metrics

The correct level depends on the idea's cost and expected payoff.

### Skill / Rules Location

Target both:

- project-local rules under `E:\kykt\Dream`
- later reusable Codex skill for repeated research-agent use

## Current Research Posture

Do not rush into choosing one final project.

First construct:

1. a master research prompt
2. research rules / skill behavior
3. a scoring and convergence system
4. a plan for turning the best ideas into KYKT demos

## Phase 1 Operating Decision

The user selected a balanced two-track mode:

```text
Breadth Map + Minimal Demo
```

Implications:

- Breadth Map: broadly discover 3R-relevant mechanisms from papers, GitHub projects, and new architecture work.
- Minimal Demo: keep one small teacher-facing demo path alive from the beginning.
- Do not let the work become only literature collection.
- Do not let the demo path prematurely collapse the research into ordinary engineering.

Important decisions should be discussed with the user before commitment, especially:

- choosing the primary research thesis
- committing to one architecture family
- heavy training or large data construction
- major KYKT app information-architecture changes
- converting the rules draft into a reusable Codex skill

## Phase 1 Planning Decision

The next critical step is a comprehensive research route survey that prepares future implementation.

The plan is documented in:

```text
E:\kykt\Dream\PHASE1_RESEARCH_PLAN.md
```

This phase should produce:

- a frontier source map
- a research unit bank
- an idea scoreboard
- minimal demo candidates
- a decision memo before deeper implementation

## Phase 1 Execution Status

Started on 2026-05-01.

Current artifacts:

- `PHASE1_EXECUTION_LOG.md`
- `FRONTIER_SOURCE_MAP.md`
- `RESEARCH_UNIT_BANK.md`
- `IDEA_SCOREBOARD.md`
- `MINIMAL_DEMO_CANDIDATES.md`
- `PHASE1_DECISION_MEMO.md`

Updated signal after subagent-assisted survey:

```text
The most promising architecture axis is not a bare Mamba-3R replacement.
It is a geometry-governed control graph for routing, writing, compressing, verifying, and adapting spatial state.

The most feasible first teacher demo axis is:
  Dream research lane + 3R Composer + Geometry Critic-Revision + one pose-free Gaussian visual path.
```

This is not a final thesis. It is a stronger first-pass hypothesis pending reproducibility checks and user discussion.

## Phase 1.5 Workflow Deployment

User direction:

```text
先不急着复现吧，我们要先部署一下研究流程
```

Decision:

```text
Pause model reproduction and deploy the research operating system first.
```

New workflow assets:

- `AGENT_MASTER_PROMPT.md`
- `FRONTEND_DESIGN_HANDOFF_PROMPT.md`
- `RESEARCH_WORKFLOW.md`
- `RESEARCH_DATA_MODEL.md`
- `WORKFLOW_STATUS.md`
- `registry/source_registry.md`
- `registry/research_unit_registry.md`
- `registry/decision_registry.md`
- `cycles/CYCLE-20260501-001.md`
- `experiments/EXP-20260501-001-dust3r-splatt3r-smoke-plan.md`
- templates under `templates/`

Current rule:

```text
Planned experiments are allowed.
Actual reproduction, heavy downloads, and app navigation changes require user confirmation.
Frontend design prompts are allowed.
Direct frontend implementation by Codex requires explicit user confirmation.
```

## Frontend Agent Boundary Decision

Decision:

```text
KYKT frontend design implementation is delegated to Gemini CLI / designated frontend agent.
Dream / Codex owns prompt framing, constraints, sequencing, and acceptance criteria.
```

Canonical handoff prompt:

```text
E:\kykt\Dream\FRONTEND_DESIGN_HANDOFF_PROMPT.md
```

## Agent Master Prompt Decision

Canonical prompt:

```text
E:\kykt\Dream\AGENT_MASTER_PROMPT.md
```

Decision:

```text
Use AGENT_MASTER_PROMPT.md as the required entry point for future Dream agents.
MASTER_RESEARCH_PROMPT_DRAFT.md is now historical only.
```

The master prompt must be updated when:

- current phase changes
- thesis candidate changes
- canonical files change
- decision gates change
- reusable skill packaging happens
- major workflow lanes are activated or blocked
- frontend ownership or handoff rules change

## Current Strongest Candidate

Working title:

```text
Dream3R: Geometry-Governed State and Test-Time Reasoning for Long-Context 3R
```

Core computation graph:

```text
route policy -> state write -> external spatial memory -> sparse global context -> geometry critic -> local revision / adaptation
```

Reasoning:

- Direct 3R frontier sources show memory/state/test-time update is now central.
- Architecture-transfer sources show SSM/linear attention is useful only when attached to an explicit route/state policy.
- Dynamic 4D sources show that persistent state must separate static map updates from dynamic or uncertain regions.
- Demo-enabler sources show that KYKT needs a visible asset path in parallel with the deeper architecture narrative.

## Current Working Name

No final project name yet.

Temporary umbrella name:

```text
Dream: Architecture-First 3R Research Engine
```

Current thesis name candidate:

```text
Dream3R
```

## Next Workflow Choice

The next useful decision is not "which model to install." It is which workflow lane to deepen first:

```text
A. KYKT research-lane data model
B. Master prompt and agent rule refinement
C. Phase 2 smoke-test plan refinement without running it
D. Second research cycle focused on the Dream3R thesis branch
```
