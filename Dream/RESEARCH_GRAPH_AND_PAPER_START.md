# Research Graph And Paper Start

Last updated: 2026-05-02

Status: active planning artifact; expanded first graph pass.

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

## Graph Pass 2026-05-02

This pass keeps all major branches alive. It does not select a final thesis and does not deepen GEM-3R or any other branch into a mechanism spec.

### Failure Mode To Mechanism Graph

| Failure mode | Mechanism nodes | Source / comparator anchors | Conflict or caution | Candidate composition edges | Evidence path | Claim status |
|---|---|---|---|---|---|---|
| F1 Long-context drift / forgetting | persistent latent state; hybrid memory; external spatial pointer memory; keyframe-relative gauge; anchor protection; Kalman-style update | CUT3R; STream3R; LONG3R; LoGeR; Mem3R; PAS3R; FILT3R; LongStream; OVGGT; Point3R | generic "memory 3R" is already crowded; O(1) model state is not O(1) scene memory | C1, C2, C5, C8 | proxy policy simulation over frame/chunk metadata: update frequency, anchor retention, memory growth, predicted drift risk | source mechanisms paper-proven; Dream policy inferred; performance unknown |
| F2 Dynamic-static entanglement | dynamic 4D pointmaps; temporal motion field; static/dynamic confidence split; object permanence memory; dynamic branch routing | MonST3R; POMATO; D^2USt3R; Easi3R; RayMap3R; 4DGS variants | dynamic pointmaps and training-free dynamic correction are already occupied | C3, C6, C9 | dynamic failure-case taxonomy plus simulated static/dynamic memory update policy | source mechanisms paper-proven; object permanence policy inferred |
| F3 Hard-case geometric ambiguity | geometry critic; triplet / multiview consistency; hypothesis revision; adaptive compute; model rerouting; adapter budget trigger | Test3R; TTT3R; CTRL-style critic-revision; MASt3R-SfM; SLAM3R | critic-only reports are diagnostics, not architecture; revision action must be concrete | C2, C4, C7, C10 | non-learned critic report: reprojection residual, confidence conflict, pair/triplet inconsistency, retry or model-switch recommendation | consistency objectives paper-proven; Dream critic action set inferred |
| F4 Passive observation limit | uncertainty map; next-best-view; information gain; route/revisit policy; active camera action | active perception / NBV literature; NextBestSense-style systems | robotics stack can dominate the research and raise sim-to-real burden | C11, C12 | design-only or simulation-only study: choose next view from 3R uncertainty without hardware | active perception principle paper-proven broadly; 3R-specific controller speculative |
| F5 Sensor / modality fragility | event/RGB fusion; guided depth / pose / calibration priors; blur-free temporal signal; sensor-prior conflict check | EAG3R; Event-3DGS; G-CUT3R; event depth / reconstruction | obvious event-augmented pointmap framing is already emerging; hardware/data burden is high | C6, C10, C12 | dataset-only source analysis first: identify RGB failure cases where event or priors alter the evidence vector | event/guidance mechanisms paper-proven; Dream policy role inferred |
| F6 Fragmented model ecology | capability cards; unified pointmap / pose / confidence contracts; composer routing; benchmark matrix; artifact evidence reports | DUSt3R; MASt3R; Fast3R; MonST3R; CUT3R; SLAM3R; Splatt3R; NoPoSplat; MV-DUSt3R+ | model routing alone may be system integration rather than paper-grade architecture | C4, C7, C8, C10 | capability matrix plus unified evidence report using existing KYKT/public metadata | model capabilities paper-proven or code-observed per source map; system value inferred |

### Mechanism Node Bank

| ID | Mechanism node | State stored | Computation avoided | Signal / prior added | Error mode corrected | Train-time vs test-time distinction | Branch role |
|---|---|---|---|---|---|---|---|
| M1 | Persistent latent state | compact recurrent geometry belief | full history attention | temporal continuity | early-frame loss; local drift | trained inside stateful 3R models; reused at inference | Memory / State |
| M2 | Hybrid local-global memory | local window plus compressed/global memory | all-to-all long video attention | chunk-level global constraint | long-context drift; loop inconsistency | architecture-defined, test-time memory updates | Memory / State |
| M3 | External spatial pointer memory | geometry-indexed anchors or map entries | storing everything in hidden state | sparse map retrieval | forgetting; memory saturation | mostly test-time write/read policy | Memory / State; Composer |
| M4 | Kalman-style latent filtering | state mean plus uncertainty | blind overwrite of state | uncertainty and adaptive gain | noisy update; drift accumulation | training-free or light test-time controller | Memory / State |
| M5 | Anchor / cache protection | protected keyframes, gauges, cache tokens | unbounded cache growth | anchor importance | scale drift; cache contamination | test-time cache selection | Memory / State |
| M6 | Geometry critic | residuals, conflicts, confidence report | rerunning every case blindly | consistency checks | confident wrong pointmaps; low-overlap errors | can start non-learned at test time; learned later | System-2 / Composer |
| M7 | Hypothesis revision action | local retry queue or alternative output | full reconstruction restart | revision budget | hard-case ambiguity | test-time action after critic trigger | System-2 |
| M8 | Dynamic/static memory split | static map state plus dynamic region state | letting moving objects update static memory | motion / validity field | dynamic corruption; temporal flicker | inference-time branch routing; training can learn masks | Dynamic / 4D |
| M9 | Object permanence memory | object identity, geometry, uncertainty over time | frame-only dynamic estimation | identity continuity | object identity loss | mostly test-time association plus learned representation later | Dynamic / 4D |
| M10 | Event / sensor-prior fusion | asynchronous temporal evidence or priors | relying on blurred RGB only | events, depth, pose, calibration, IMU | motion blur; low light; textureless surfaces | sensor-specific test-time input; training data needed for fusion | Cross-modal |
| M11 | Composer routing | model capability card and input regime state | trying one universal model everywhere | regime classifier and model evidence | fragmented model ecology | test-time system/controller layer | Composer |
| M12 | Unified evidence contract | pointmap, pose, confidence, artifact, report schema | incomparable outputs | common data model | weak comparison and demo disconnect | system-level test-time artifact contract | Composer / KYKT |
| M13 | Uncertainty-driven active view | uncertainty map and view candidate state | passive missing-view failure | information gain | blind spots; poor trajectory | simulation/test-time planning first | Active Perception |
| M14 | Adapter / self-update budget | restricted adapter or fast-weight delta | full fine-tuning | critic-gated update scope | scene/domain shift | test-time adaptation only if state actually changes | Continual |

### Composition Edges

| ID | Composition edge | Connects | Why it may matter | Smallest evidence path | Claim status |
|---|---|---|---|---|---|
| C1 | Kalman update + anchor-protected cache | M4 + M5 -> F1 | treats streaming 3R as belief update while preserving coordinate anchors | simulate gain and anchor-retention decisions over long-video metadata | inferred |
| C2 | External memory + geometry critic | M3 + M6 -> F1/F3 | only writes or trusts memory when geometry evidence passes consistency checks | critic report decides write / merge / ignore actions | inferred |
| C3 | Dynamic/static split + memory governance | M8 + M3/M5 -> F2/F1 | prevents moving regions from poisoning static map and anchors | dynamic taxonomy plus static-map update policy | inferred |
| C4 | Critic + composer rerouting | M6 + M11 -> F3/F6 | hard samples trigger alternate model or retry instead of silent failure | capability matrix plus critic-triggered routing report | inferred |
| C5 | Route-scan policy + hybrid memory | M2 + route policy -> F1 | geometry-aware ordering may preserve local neighborhood and loop cues better than chronological-only state | route simulator with locality/redundancy metrics | speculative |
| C6 | Event signal + dynamic branch | M10 + M8 -> F5/F2 | events can separate fast motion evidence from RGB blur | dataset-only comparison of blurred RGB vs event-aligned motion cues | speculative until data/code verified |
| C7 | Critic + adapter budget | M6 + M14 -> F3/domain shift | adaptation opens only when a geometric failure is diagnosed | design memo plus proxy update triggers | speculative |
| C8 | Composer + benchmark matrix | M11 + M12 -> F6 | turns model fragmentation into measurable capability cards and evidence reports | fill model-regime matrix from existing source map and KYKT metadata | inferred |
| C9 | Object permanence + 4DGS initialization | M9 + 4D asset path -> F2/demo | preserves dynamic identity while producing teacher-visible assets | planned-only storyboard; no 4DGS run | speculative |
| C10 | Guided priors + critic conflict check | M10 + M6 -> F5/F3 | depth/pose/calibration priors help only if conflicts are detected and reported | source analysis of guided 3R plus synthetic conflict examples | inferred |
| C11 | Active view + uncertainty map | M13 -> F4 | system requests the next observation that reduces reconstruction uncertainty | simulation-only next-view choice over mock uncertainty map | speculative |
| C12 | Active sensing + event/guided mode | M13 + M10 -> F4/F5 | action can choose not only viewpoint but also sensing mode | design-only until hardware/sim route exists | speculative |

### Branch-Neutral Evidence Ladder

| Evidence level | Allowed now | Output artifact | Not allowed without approval |
|---|---|---|---|
| L1 Paper-grade design evidence | yes | graph, branch matrix, mechanism forms, pseudo-policy definitions | claiming measured gains |
| L2 Proxy evidence | yes if no heavy run | metadata simulation, capability matrix, critic-report mock, public-output annotation | downloading checkpoints or running reproduction |
| L3 Small prototype | plan only | experiment plan under `experiments/` | local model run, heavy install, checkpoint download |
| L4 Model change | no | future proposal only | training, fine-tuning, learned controller implementation |

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
