# Dream Research State

Last updated: 2026-05-02

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

## Backend-First Correction

User correction:

```text
我们后端为主，前端这个又不是核心，你别跑偏了
```

Decision:

```text
Dream should prioritize backend/research-pipeline contracts.
Frontend is downstream presentation only.
```

Current priority:

- research data contracts
- backend-owned registries
- state transitions for source / RU / decision / experiment
- API/task boundaries for later KYKT service integration
- evidence and artifact reference rules
- future agent automation hooks

Current non-priority:

- frontend UI implementation
- visual polish
- KYKT navigation changes
- Gemini CLI work unless backend contracts are ready

## Research-Mainline Correction

User correction:

```text
app只是一部分呀，我们的主线还是研究新的内容
```

Decision:

```text
Dream's mainline is research-content discovery, mechanism synthesis, and thesis validation.
The KYKT app, backend contracts, and frontend handoffs are supporting layers for preserving, testing, and presenting that research.
```

Interpretation:

- "backend first" only means backend before frontend when system integration becomes necessary.
- It does not mean backend engineering is the core research priority.
- The next default cycle should deepen new 3R mechanisms, not app or backend implementation.
- KYKT integration remains required, but it should follow the shape of the research content.

## Next Workflow Choice

The next useful decision is not "which model to install." It is which workflow lane to deepen first:

```text
A. Research content cycle: Dream3R thesis validation and new mechanism discovery
B. Frontier source/mechanism mining for 3R-translatable ideas
C. Backend research pipeline contract as support infrastructure
D. Phase 2 smoke-test plan refinement without running it
```

## Collaboration Pathway

Current collaboration roadmap:

```text
E:\kykt\Dream\COLLABORATION_ROADMAP.md
```

Recommended near-term sequence:

```text
1. Collaboration protocol
2. Research content / Dream3R thesis validation cycle
3. Backend research pipeline contract as support infrastructure
4. Teacher-facing storyboard
5. Planned experiment selection
```

Current recommendation:

```text
Start next with a research content cycle; keep backend, app, and frontend downstream.
```

## Cycle 003 Research Signal

Started:

```text
CYCLE-20260501-003: Dream3R Thesis Stress Test
```

New artifact:

```text
E:\kykt\Dream\DREAM3R_THESIS_STRESS_TEST.md
```

Main finding:

```text
The original Dream3R framing is directionally right but too broad and too close to 2026 long-context 3R memory work.
The stronger candidate is GEM-3R: Geometry-Governed Executive Memory for 3R.
```

Interpretation:

- Mamba/SSM is only one possible state-update mechanism, not the thesis.
- Geometry-gated update alone overlaps with PAS3R and FILT3R.
- Hybrid memory alone overlaps with LoGeR and Mem3R.
- Constant-budget cache alone overlaps with OVGGT and LongStream-style cache work.
- The potential new direction is an executive policy that chooses among memory, cache, anchor, critic, dynamic, and adaptation actions based on geometry evidence.
- Subagent synthesis reinforces that the safest novelty spaces are explicit memory control, cross-session revisitable scene memory, dynamic object permanence, and unified executive contracts across reconstruction / matching / localization / SLAM.

Status:

```text
proposed research branch, not final thesis
```

## Multi-Track Correction

User correction on 2026-05-02:

```text
我不喜欢一上来就押宝单单一个小方向，建议多思考一些再做
```

Decision:

```text
Do not deepen GEM-3R as the single next thesis yet.
Run a multi-track comparison first.
```

Current branch pool:

- executive memory / state governance
- geometry critic / System-2 3R
- dynamic object permanence / 4D memory
- cross-modal / event-augmented 3R
- 3R composer / unified model ecology
- active spatial perception / RL-3R

New artifact:

```text
E:\kykt\Dream\MULTI_TRACK_RESEARCH_CANVAS.md
```

Current recommendation:

```text
Create a branch comparison matrix before drafting any one mechanism spec.
```

## Graph-Based Research Method

User synthesis:

```text
Many mentioned, unmentioned, or undiscovered innovation points, modules, techniques, papers, and compositions may become useful at different stages.
A nonlinear, complex graph-like or higher-dimensional structure is more suitable for the research.
```

Decision:

```text
Start from a failure-mode / mechanism / evidence research graph, not from a single branch.
```

New artifacts:

- `RESEARCH_GRAPH_AND_PAPER_START.md`
- `BRANCH_COMPARISON_MATRIX.md`

Paper-start principle:

```text
Do not start with "we use module X."
Start with the field-level fragmentation problem and a unifying 3R spatial-intelligence control formulation.
```

## Cycle 004 Graph And Matrix Pass

Started:

```text
CYCLE-20260502-001: Failure-Mode Graph And Branch Matrix Fill
```

User direction:

```text
Continue Dream in research-content-first mode.
Build the failure-mode / mechanism / composition graph and fill BRANCH_COMPARISON_MATRIX.md before deepening any single thesis branch.
```

Artifacts updated:

- `RESEARCH_GRAPH_AND_PAPER_START.md`
- `BRANCH_COMPARISON_MATRIX.md`

Current interpretation:

- Executive Memory / State Governance remains a strong architecture umbrella, but is not selected.
- Geometry Critic / System-2 3R has the cheapest credible evidence path and strong demo clarity, but still needs a real action beyond diagnostics.
- Dynamic Object Permanence / 4D Memory remains a major branch if it targets object identity and memory policy rather than only 4D visualization.
- 3R Composer / Unified Model Ecology is the best KYKT support and evidence infrastructure, but likely needs pairing with a stronger architecture claim.
- Cross-Modal / Event-Augmented 3R and Active Spatial Perception stay alive as robustness / future branches, with higher evidence and engineering burden.

Current rule:

```text
No branch is discarded, no finalist set is approved, and no mechanism spec should be drafted until the user chooses which branches to deepen.
```

Recommended next research action:

```text
Define a branch-neutral action taxonomy and proxy metrics:
conflict detection, action entropy, anchor retention, memory growth, dynamic pollution,
route regret, and teacher-demo clarity.
```

## Cycle 005 Discussion Synthesis

Started:

```text
CYCLE-20260502-002: Research Preparation And Mechanism Intake Discussion
```

Discussion additions that must not be lost:

- The comparator map must be expanded. Spann3R is an important memory / spatial-context comparator, and Point3R, SLAM3R, MV-DUSt3R+, RayMap3R, and G-CUT3R should also be placed into the appropriate branch comparisons.
- Sparse attention, linear attention, SSM/Mamba, attention residuals, hidden-state reuse, KDA-like finite-state memory, RL, active perception, continual learning, adapter updates, new visual backbones, segmentation, tracking, optical flow, VOS, Gaussian/4D representations, event, depth, IMU, LiDAR, and guided-prior methods should enter Dream's mechanism bank.
- These methods should not be added as buzzwords. Each must be mapped through `Failure mode -> Mechanism -> Action -> Proxy metric -> Comparator -> Evidence level`.
- Combining strengths of several new 3R models is important, but has two layers:
  - 3R Composer L1: system composer / model router / capability cards / evidence report.
  - 3R Composer L2: mechanism distillation into unified 3R spatial-intelligence actions.
- L1 is the strongest KYKT and demo support layer. L2 is the stronger paper-novelty path.

Immediate path:

```text
Comparator completion
-> Architecture mechanism intake
-> Action taxonomy
-> Proxy metrics
-> Branch shortlist
-> Mechanism spec
-> Planned experiment
```

Follow-up progress:

- Created `ARCHITECTURE_MECHANISM_INTAKE.md` as the active branch-neutral intake map for broad architecture and visual-method candidates.
- Updated comparator anchors in `RESEARCH_GRAPH_AND_PAPER_START.md` and `BRANCH_COMPARISON_MATRIX.md` to include Spann3R and related missing comparators.
- Updated `FRONTIER_SOURCE_MAP.md` with a comparator completion pass.
- Updated `registry/source_registry.md` with Spann3R and RayMap3R lightweight entries.
- Updated `AGENT_MASTER_PROMPT.md`, `README.md`, and `WORKFLOW_STATUS.md` so future Dream agents inspect the mechanism-intake artifact when relevant.

Current near-term research object:

```text
Shared action taxonomy + proxy metric bank across all branches.
```

Status:

```text
first-pass drafted in ARCHITECTURE_MECHANISM_INTAKE.md, needs refinement before branch shortlist.
```

## Cycle 006 Action Taxonomy And Proxy Metrics

Started:

```text
ACTION_TAXONOMY_AND_PROXY_METRICS.md
```

Progress:

- Collapsed the broad action vocabulary into eight core Dream actions:
  - A1 State Update Control
  - A2 Spatial Memory Governance
  - A3 Context / Anchor Budgeting
  - A4 Geometry Verification
  - A5 Repair / Reroute Decision
  - A6 Dynamic/Object State Separation
  - A7 Prior / Modality Arbitration
  - A8 Evidence Acquisition / Adaptation Budget
- Defined an evidence signal vector for frame/chunk/model-output/sample-regime reasoning.
- Defined L2 proxy metric protocols:
  - P1 conflict detection
  - P2 anchor retention
  - P3 memory growth and usefulness
  - P4 dynamic pollution
  - P5 route regret
  - P6 action entropy
  - P7 uncertainty reduction / view gain
  - P8 adaptation benefit versus forgetting risk

First research inference:

```text
The near-term finalist pool should probably combine:
Geometry Critic / System-2 3R
+ Executive Memory / State Governance
+ 3R Composer as evidence infrastructure.
```

Important caveat:

```text
This is an inference, not a decision.
Dynamic Object Permanence remains a close candidate and should not be discarded.
User approval is still required before drafting any finalist mechanism spec.
```

Next research object:

```text
Branch shortlist decision surface:
one-page summary per branch, owned A1-A8 actions, weakest comparator pressure,
first proxy test, teacher demo form, and fail-fast condition.
```

Completed:

```text
BRANCH_SHORTLIST_DECISION_SURFACE.md
```

Provisional synthesis:

- Geometry Critic / System-2 3R and Executive Memory / State Governance are the strongest immediate finalists.
- 3R Composer / Unified Model Ecology is the strongest evidence infrastructure and likely support layer.
- Dynamic Object Permanence / 4D Memory should be added as a third finalist if the user wants to preserve the strongest F2 / visual paper story before mechanism specs.

Current recommendation:

```text
Ask the user to choose one of:
A. Geometry Critic + Executive Memory, Composer as support
B. Add Dynamic Object Permanence as third finalist
C. Keep all six branches and prepare proxy case-card templates
D. Mine more sources before choosing finalists
```

Guidance-sync rule:

```text
When a workflow artifact is created or promoted, update AGENT_MASTER_PROMPT.md,
README.md, WORKFLOW_STATUS.md, RESEARCH_STATE.md, and the current cycle log.
```

## Agent Handoff Prompt Update

Decision:

```text
The old short invocation that asks agents to build the graph and fill BRANCH_COMPARISON_MATRIX.md is stale.
Future agents should continue from the shortlist stage.
```

Updated:

- `AGENT_MASTER_PROMPT.md` mandatory load protocol now includes:
  - `ARCHITECTURE_MECHANISM_INTAKE.md`
  - `ACTION_TAXONOMY_AND_PROXY_METRICS.md`
  - `BRANCH_SHORTLIST_DECISION_SURFACE.md`
- `AGENT_MASTER_PROMPT.md` short invocation now tells agents to prepare proxy case-card templates and user-approved finalist mechanism specs, not to repeat graph/matrix filling.
- `registry/decision_registry.md` records this as DEC-20260502-004.

Current handoff instruction:

```text
Use E:\kykt\Dream\AGENT_MASTER_PROMPT.md as your operating prompt. Read its mandatory load protocol first, then continue Dream in research-content-first mode from the current shortlist stage: use BRANCH_SHORTLIST_DECISION_SURFACE.md, ACTION_TAXONOMY_AND_PROXY_METRICS.md, and ARCHITECTURE_MECHANISM_INTAKE.md to prepare proxy case-card templates and user-approved finalist mechanism specs. Do not reproduce models, download checkpoints, train/fine-tune, change KYKT app navigation, implement frontend, discard major branches, or finalize a thesis unless I explicitly approve it in this conversation. Keep guidance files synchronized when creating or promoting workflow artifacts.
```

## Paper-Writing Value Of Broad Mechanism Intake

User correction:

```text
Choosing many new mechanisms also helps us write the paper.
```

Decision:

```text
Mechanism intake should be judged not only by immediate implementation feasibility,
but also by paper-writing value.
```

Interpretation:

- Sparse attention, RL, continual learning, attention residuals, visual priors, event/guided sensing, dynamic/4D methods, and 3R model combinations help form a stronger related-work taxonomy.
- The point is not to implement everything.
- The point is to show the field's partial solutions and define Dream's control vocabulary over memory, verification, dynamics, priors, action, and evidence.
- A mechanism can remain in the intake map as a writing / taxonomy asset even if it is not selected for the first prototype.
- Branch shortlist decisions should score both evidence feasibility and writing value.

Files updated:

- `ARCHITECTURE_MECHANISM_INTAKE.md`
- `ACTION_TAXONOMY_AND_PROXY_METRICS.md`
- `RESEARCH_GRAPH_AND_PAPER_START.md`
