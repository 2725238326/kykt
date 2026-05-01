# Research Graph And Paper Start

Last updated: 2026-05-02

Status: active planning artifact.

## Core Decision

Do not start the next step by picking one branch.

Start by building a research graph:

```text
failure modes -> mechanisms -> compositions -> evidence paths -> paper claims
```

Reason:

The user's research intuition is that useful innovation may come from unexpected combinations of 3R models, memory, reasoning, dynamics, active perception, cross-modal sensing, and system mechanisms.

A linear branch list is useful for organization, but insufficient for discovery.

## Top-Conference Writing Principle

A strong CVPR / ICCV style paper should not begin as:

```text
We use X module on 3R.
```

It should begin as:

```text
Modern 3R foundation models have fragmented into strong but partial mechanisms.
The unsolved problem is how a spatial intelligence system should decide which mechanism to invoke, trust, update, or reject under changing geometry.
```

This lets the paper absorb many modules without becoming a random system mashup.

## The First Research Object

The next object to build is not a model.

It is:

```text
Dream Research Graph
```

Nodes:

- 3R failure modes
- source papers / repos
- mechanisms
- state variables
- memory types
- reasoning / verification actions
- dynamic / 4D representations
- sensor modalities
- active perception actions
- benchmark signals
- teacher-demo artifacts
- KYKT integration surfaces

Edges:

- solves
- partially solves
- fails on
- depends on
- conflicts with
- composes with
- replaces
- verifies
- initializes
- triggers
- enables demo
- creates paper claim

## Failure Modes First

Start the graph from failure modes, not from fashionable techniques.

### F1: Long-Context Drift / Forgetting

Symptoms:

- coordinate drift
- scale drift
- cache contamination
- memory saturation
- early-frame loss
- loop inconsistency

Relevant mechanisms:

- persistent state
- hybrid memory
- spatial pointer memory
- Kalman-style update
- anchor protection
- keyframe-relative gauge

Candidate branches:

- executive memory
- long-context benchmark
- active revisit policy

### F2: Dynamic-Static Entanglement

Symptoms:

- moving objects corrupt camera / static map
- temporal flicker
- object identity loss
- dynamic regions pollute long-term state

Relevant mechanisms:

- dynamic 4D pointmaps
- temporal motion fields
- dynamic/static split
- 4DGS initialization
- object permanence memory

Candidate branches:

- dynamic object permanence
- 4D memory
- event-assisted dynamics

### F3: Hard-Case Geometric Ambiguity

Symptoms:

- occlusion
- low overlap
- repeated structures
- blur
- wrong pairwise alignment
- confident but wrong pointmaps

Relevant mechanisms:

- geometry critic
- test-time consistency
- hypothesis revision
- adaptive compute
- model rerouting

Candidate branches:

- System-2 3R
- critic-revision
- composer controller

### F4: Passive Observation Limit

Symptoms:

- missing views
- blind spots
- unknown backside geometry
- poor camera trajectory

Relevant mechanisms:

- next-best-view
- information gain
- active perception
- RL / planning
- uncertainty-driven camera action

Candidate branches:

- active spatial perception
- embodied 3R

### F5: Sensor / Modality Fragility

Symptoms:

- motion blur
- low light
- high-speed motion
- rolling shutter
- textureless surfaces

Relevant mechanisms:

- event camera
- depth / LiDAR / IMU priors
- guided 3R
- event/RGB pointmap fusion

Candidate branches:

- Event-DUSt3R-style direction
- guided RGB-plus-prior 3R

### F6: Fragmented Model Ecology

Symptoms:

- no single model works across pair, multiview, video, dynamic, streaming, pose-free, and asset generation regimes
- research outputs are hard to compare
- demo paths and paper claims are disconnected

Relevant mechanisms:

- model capability cards
- unified pointmap / pose / confidence contracts
- composer controller
- benchmark matrix
- artifact and evidence reports

Candidate branches:

- 3R composer
- KYKT research workbench
- benchmark / evidence infrastructure

## Mechanism Composition Layer

The important research may come from non-obvious compositions:

```text
external memory + geometry critic
dynamic 4D pointmap + long-term state governance
event stream + test-time geometry revision
next-best-view + uncertainty map
composer routing + proxy benchmark
Kalman update + anchor-protected cache
VLM semantics + geometric failure classification
4DGS initialization + dynamic/static confidence
```

The graph should record not only individual mechanisms, but also possible compositions.

## Paper Seed Skeleton

The first paper-like draft should start with a problem formulation, not a method.

### Working Abstract Skeleton

```text
Recent 3R foundation models have moved beyond traditional SfM by directly predicting dense pointmaps and geometry from images or videos.
However, the field is rapidly fragmenting into partial solutions for streaming state, hybrid memory, dynamic scenes, test-time adaptation, and visual asset generation.
We argue that the next bottleneck is not a single backbone, but the absence of a principled spatial intelligence control layer:
when should a 3R system remember, forget, verify, revise, adapt, or actively acquire new observations?
We introduce a research graph / benchmark / framework that organizes 3R mechanisms around failure modes and compositional actions.
This enables systematic discovery of new 3R architectures and identifies several high-value candidate directions, including executive memory, System-2 geometry reasoning, dynamic object permanence, and active perception.
```

This is not the final abstract. It is a starting scaffold.

### Introduction Logic

Paragraph 1:

```text
DUSt3R-style 3R foundation models changed 3D reconstruction from brittle multi-stage geometry pipelines into direct learned pointmap prediction.
```

Paragraph 2:

```text
Follow-up work solved many local bottlenecks: matching, many-view forward passes, streaming state, hybrid memory, dynamic pointmaps, test-time adaptation, guided reconstruction, and Gaussian asset generation.
```

Paragraph 3:

```text
But these advances are modular and fragmented. Long-context, dynamic, uncertain, or embodied settings require decisions across mechanisms, not simply a larger backbone.
```

Paragraph 4:

```text
We formulate 3R as a spatial intelligence control problem over memory, verification, dynamics, sensing, and action.
```

Paragraph 5:

```text
We instantiate this formulation as a research graph and use it to derive candidate architectures and minimal evidence paths.
```

## Next Concrete Artifact

Create:

```text
BRANCH_COMPARISON_MATRIX.md
```

Required columns:

- branch
- failure modes addressed
- closest competitors
- mechanism ingredients
- possible compositions
- novelty gap
- smallest evidence path
- teacher-facing demo
- engineering cost
- top-conference risk
- KYKT support path
- recommendation

This is the correct next step before mechanism spec or reproduction.

## Research Rule

For the next phase:

```text
Do not ask "which module is coolest?"
Ask "which failure-mode graph creates the strongest paper claim and the cheapest credible evidence?"
```

The answer may still become GEM-3R, System-2 3R, dynamic object permanence, active perception, or another composition. The graph decides.
