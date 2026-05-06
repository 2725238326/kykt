# Dream3R Paper Draft v1

Last updated: 2026-05-06 (cycle 017; architecture-centric rewrite per DEC-20260506-001)

Status: draft (not submission-ready; no measured results; evidence labels per Discipline rule 5)

Supersedes: literature/PAPER_PHASE2_BLUEPRINT.md (demoted to SUPPORT per DEC-20260506-001)

Source anchors:
- specs/SPEC-20260506-001-dream3r-architecture.md (v0.1; central contribution)
- specs/SPEC-20260506-002-dream3r-ablation-plan.md (v0.1; planned experiments)
- specs/SPEC-20260506-003-dream3r-comparator-map.md (v0.1; related work anchor)
- specs/SPEC-20260503-001..003 + SPEC-20260504-001 (4 finalist specs; inputs)
- paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md (v2.1)
- planning/ACTION_TAXONOMY_AND_PROXY_METRICS.md (A1-A8 + V1 + P1-P6)
- literature/PAPER_RELATED_WORK_SKELETON.md (prose draft; reused)
- cases/ L2 portfolio (13 cards)

---

## Working title

```text
Dream3R: A Control-Graph Architecture for Long-Context,
Dynamic-Aware, Multi-Model 3D Reconstruction
```

Evidence label: inferred.

## Abstract (draft)

```text
Post-DUSt3R 3D reconstruction (3R) models have diversified rapidly:
some handle long sequences via persistent memory, some separate dynamic
from static content, some verify geometric consistency at test time,
and some target specific input regimes. Yet each model addresses one
failure mode in isolation, and no shared control vocabulary governs
when to update memory, when to verify, when to reroute to a
different backbone, or when to suppress a static-map write because
a region is dynamic.

We propose Dream3R, a control-graph architecture that makes this
vocabulary first-class. Dream3R synthesizes four mechanisms — an
executive Memory controller (SSM-based), a Geometry Critic (small
transformer head), an Object Permanence module (slot memory), and a
regime-aware Composer (parameter-free table join) — onto a single
cross-specification memory bus. The bus carries a 17-signal evidence
vector and six conflict-resolution rules (CR-1..CR-6) rendered as
architectural gates. The result is a modular, falsifiable
architecture where each component can be independently ablated and
each cross-module interaction is governed by an explicit typed
contract.

We present the architecture design, a 10-experiment ablation plan
organized in three priority tiers, and a comparator map against 14+
existing 3R models showing that Dream3R's control graph, bus, and
gate mechanisms have no direct precedent in the 3R literature.
```

Evidence label: inferred (no training; no measured results; design contribution only).

## 1. Introduction

### 1.1 The fragmentation problem

The DUSt3R family converted 3D reconstruction from a pipeline problem into a single pose-free pointmap regression. Within two years, the follow-up landscape diversified along three axes: matching and speed (MASt3R, Fast3R, MV-DUSt3R+, NoPoSplat), temporal state for streaming (CUT3R, STream3R, Spann3R, LONG3R, LoGeR, LongStream), and dynamic-scene extensions (MonST3R, POMATO, D^2USt3R, Easi3R, RayMap3R). Universal claims such as VGGT and MapAnything compress those axes into single feed-forward backbones.

The field is past the point where one more backbone resolves the open failure modes. What is missing is not more pointmap accuracy on controlled benchmarks; what is missing is a *control vocabulary*: a compact set of typed actions a 3R system can take (update state, verify geometry, reroute model, suppress write, track object identity), and a typed set of regimes under which each action is appropriate.

### 1.2 Our contribution: control graph as architecture

Dream3R proposes that the architectural novelty is the *graph* — the modules, the bus connecting them, and the gates governing their interactions — not any single module's internals.

Concretely:

1. **Four modules on a shared bus.** Memory (SSM/Mamba; owns A1 state update, A2 spatial memory governance, A3 anchor budgeting), Critic (small transformer head; owns A4 verification, A5 repair), Permanence (slot memory; owns A6 dynamic/object separation), and Composer (parameter-free table join; owns A5 routing) share a cross-specification memory bus carrying a 17-signal evidence vector.

2. **Conflict resolution as gates.** Six rules (CR-1..CR-6) from the cross-spec signal contract are rendered as architectural elements: hard masks (CR-1, CR-2), input-feature weightings (CR-3), tiebreak modules (CR-4), label-propagation invariants (CR-5), and audit ledgers (CR-6).

3. **Hybrid substrate hypothesis.** Transformer perception (paper-proven), SSM executive memory (paper-derived), slot memory for object permanence (paper-proven outside 3R), parameter-free routing (paper-derived from MoE literature). The hypothesis is that forcing all substrates into one type degrades performance.

4. **Falsifiable by design.** Every architectural claim maps to a specific ablation (10 ablations in 3 tiers). The highest-priority ablation (ABL-1, bus removal) directly tests whether the bus adds value over flat module composition.

### 1.3 What this paper is and is not

This paper IS:
- An architecture proposal with explicit evidence labels per component
- An ablation plan specifying which experiments falsify which claims
- A comparator map showing where Dream3R sits relative to 14+ existing 3R models

This paper is NOT:
- A trained model (no training authorized)
- A measured performance claim (all numbers are inferred or paper-derived)
- A final thesis (Dream3R remains a candidate per DEC-20260501-004)

## 2. Related work

### 2.1 Pose-free 3R foundations

DUSt3R introduced pose-free pointmap regression. MASt3R added dense local feature matching. VGGT produced per-frame pointmap, camera, and depth tokens in a single forward pass. Fast3R introduced efficient multi-view attention. These models define the perception substrate that Dream3R's Perceiver core (C1) inherits.

### 2.2 Long-context memory (F1)

CUT3R, STream3R, Spann3R, LONG3R, LoGeR, LongStream, and Mem3R each maintain persistent state across frames. Spann3R uses spatial memory with anchor-based retrieval; CUT3R uses full-update per frame; LONG3R adds gating; LoGeR combines local and global retrieval; LongStream decouples gauge from cache. Each picks one update/cache/store rule and shows it dominates their baseline.

Dream3R's Memory module (C2) asks a different question: given a workload spanning multiple streaming regimes and a state budget, which update rule should be selected per window? The A1 update_kind classifier (5-way: full, pose-adaptive, Kalman, skip, reset) makes this a first-class architectural choice rather than a fixed design decision.

### 2.3 Dynamic-static handling (F2)

MonST3R, POMATO, D^2USt3R, Easi3R, and RayMap3R handle dynamic content at the per-frame level. None targets persistent object identity across the dynamic-static split.

Dream3R's Permanence module (C3) adds object_track_set — a slot-memory-based persistent identity set — and the suppress_static_write handoff to Memory (CR-2). The contribution is identity persistence and static-map immunity, not per-frame motion estimation.

### 2.4 Verification and repair (F3)

Test3R enforces test-time consistency via a verifier head. TTT3R triggers test-time training on confidence drops. MASt3R-SfM adds classical SfM refinement. CTRL shows the critic-revision pattern in the LM domain.

All are scoped to one model family. Dream3R's Critic module (C4) asks whether a cross-model A5 reroute — bound to Composer's capability_match — can cross model families at inference time. The A5 action set (accept, rerun_local, reroute_model, open_anchor, request_prior, conflict_unresolved) is broader than simple re-run.

### 2.5 Model ecology and routing (F6)

No 3R paper publishes a routing metric. Capability cards exist informally in README comparison tables but produce no falsification axis.

Dream3R's Composer module (C5) defines route_regret as the first 3R-specific routing falsification axis: if the router chooses backbone X over backbone Y in regime R, how much was left on the table?

### 2.6 What has no 3R precedent

Five Dream3R elements have no close comparator in the existing 3R literature:

1. Cross-spec memory bus as explicit typed tensor schema
2. CR-rules as architectural gates (hard masks, invariants, ledgers)
3. Composer with route_regret falsification axis
4. A5 repair-facet action set beyond simple re-run
5. Object_track_set with persistent identity across frames

These carry the evidence label "architecture-novel" and represent the highest paper novelty — and highest risk.

## 3. Architecture

### 3.1 Overview

Dream3R is a control-graph-as-architecture: six computational cores (C1 Perceiver, C2 Memory SSM, C3 Permanence slot memory, C4 Critic head, C5 Composer table join, C6 Memory Bus) connected by a typed bus carrying the v2.1 cross-spec signal contract as its runtime API.

[Architecture block diagram: see SPEC-20260506-001 section "Top-level architecture"]

### 3.2 Token classes

Six token classes flow through the architecture:

- T1 frame tokens (Perceiver output; paper-proven)
- T2 pointmap tokens (Perceiver pointmap head; paper-proven)
- T3 evidence tokens (17-signal V1 projection; inferred)
- T4 regime token (Composer regime classifier; inferred)
- T5 anchor + object tokens (Memory + Permanence slot memory; inferred)
- T6 bus tokens (typed dict; all modules publish; architecture-novel)

### 3.3 Computational cores

**C1 Perceiver** (transformer, ~300-700M params inferred): per-frame perception backbone. Inherits DUSt3R/MASt3R/VGGT lineage. Evidence: paper-proven.

**C2 Memory SSM** (Mamba-style, ~50-150M params inferred): executive memory with A1 update_kind classifier (5-way), A2 write head, A3 anchor/cache controller. Evidence: paper-derived for SSM-as-3R-memory; inferred for A1 classifier composition.

**C3 Permanence slot memory** (~30-80M params inferred): slot attention over object_track_set + per-region A6 classifier (suppress/admit/defer) + object identity head. Evidence: inferred (slot-attention for 3R-object-permanence is novel; slot attention itself is paper-proven outside 3R).

**C4 Critic head** (small transformer, ~5-30M params inferred): A4 verifier head (scalar conflict_score) + A5 repair-facet classifier (5-way). Evidence: inferred (small-Critic over V1 tokens is novel; verifier-on-3R is paper-proven).

**C5 Composer** (parameter-free, 0 params): regime_card x capability_card -> capability_match -> route_recommendation. Evidence: paper-derived for table-based routing; architecture-novel for 3R-specific regime cards.

**C6 Bus** (no params): typed-dict tensor namespace with publish/read/handoff surfaces + CR-1..CR-6 gate modules. Evidence: architecture-novel.

### 3.4 The memory bus

The bus has three surfaces:

1. **Published signals** (read-only contract): each V1 signal is a typed slot; producer writes once per window, consumers read.

2. **Handoffs** (binding signals): Permanence's suppress_static_write binds Memory's A2 write head (CR-2). Composer's route_recommendation binds Critic's A5 reroute (CR-1 gated).

3. **Gates** (conflict resolution): CR-1 hard mask on reroute (capability_match spread); CR-2 hard mask on static write (suppress handoff); CR-3 input weighting (drift is context-cue-only for A4); CR-4 tiebreak (Critic-internal preference on ties); CR-5 evidence-label propagation (MIN invariant); CR-6 audit log.

### 3.5 Bus tick protocol

Per window: (1) Perceiver forward, (2) Memory pre-read, (3) Permanence forward, (4) Memory forward with CR-2 gate, (5) Composer forward (per-input), (6) Critic forward with CR-1/CR-3/CR-4 gates, (7) Gate housekeeping (CR-5/CR-6), (8) Output aggregation.

The order is deterministic. Reads of (t-1) state follow the v2.1 forward-reference null protocol.

### 3.6 State ownership

State is owned by exactly one module. Cross-module reads are read-only with evidence-label propagation (CR-5). Cross-module commands use handoff signals with refusal protocol.

| State | Owner | Mutability |
|---|---|---|
| latent_state | Memory | per-window (A1) |
| anchor_set | Memory | per-window (A3) |
| object_track_set | Permanence | per-frame (A6) |
| route_history | Critic | append per A5 |
| capability_card | Composer | static per cycle |

### 3.7 Substrate hypothesis

The hybrid substrate (transformer perception + SSM memory + slot permanence) is v0.1's most falsifiable choice. The claim: per-frame perception is local + parallel (transformers excel); memory is sequential + compressed (SSMs excel); forcing both into one substrate degrades performance. ABL-2 in the ablation plan tests this directly.

## 4. A1-A8 action mapping

Each action from the taxonomy maps to a specific module, concrete layer, trigger condition, and bus signals:

| Action | Module | Layer | Bus writes | Bus reads |
|---|---|---|---|---|
| A1 State Update | Memory C2 | 5-way classifier + gate | latent_state, policy_log | pose_novelty, dynamic_ratio |
| A2 Memory Write | Memory C2 | 4-way write head | static_map writes | suppress_static_write (CR-2) |
| A3 Anchor Budget | Memory C2 | anchor + cache controller | anchor_set | anchor_importance, cache_pressure |
| A4 Verification | Critic C4 | scalar regression | conflict_score | T3 evidence subset |
| A5 Repair | Critic C4 | 5-way classifier | route_history | capability_match (CR-1 gated) |
| A5 Routing | Composer C5 | table join | route_recommendation | regime_card, route_history |
| A6 Dynamic Split | Permanence C3 | 3-way + identity head | object_track_set, suppress handoff | T2 pointmap, T3 subset |
| A7 Prior | Reserved | bus hook only | (none in v0.1) | (would read prior_rgb_conflict) |
| A8 Active | Reserved | bus hook only | (none in v0.1) | (would read uncertainty_area) |

## 5. Planned experiments

### 5.1 Ablation plan overview

10 ablations in 3 priority tiers, designed so each architectural claim has a specific kill condition:

**Tier 1 (must-run):**
- ABL-1: Bus removal (flat baseline). Kill: flat matches full on 4+/6 metrics.
- ABL-2: Substrate hypothesis (hybrid vs transformer-only vs SSM-only). Kill: single-substrate matches hybrid.
- ABL-3: Gradient isolation (stop_gradient on cross-spec reads vs free flow). Kill: isolated is worse on all metrics.

**Tier 2 (should-run):**
- ABL-4..7: Per-module removal (Critic, Memory, Permanence, Composer).

**Tier 3 (refinement):**
- ABL-8: Per-CR-rule ablation (6 sub-experiments).
- ABL-9: Evidence signal leave-one-out (17 signals).
- ABL-10: Training loss weight sensitivity.

### 5.2 Benchmark categories

- B1: Static pair (sanity check)
- B2: Many-view static (anchor retention)
- B3: Long dynamic video (drift + pollution)
- B4: Mixed-regime batch (route regret)
- B5: Hard-case ambiguity (conflict detection)
- B6: Adversarial CR-triggering (bus verification)

### 5.3 Proxy metrics

- P1: Conflict detection rate + false alarm rate
- P2: Anchor retention (protected important / total important)
- P3: Memory growth + reuse rate
- P4: Dynamic pollution (static-map purity)
- P5: Route regret (cost-adjusted, per-regime)
- P6: Action entropy (controller validity guard)

## 6. Comparator positioning

Dream3R is NOT a competitor at any single substrate axis. It is a control graph that REUSES existing 3R substrates as its modules. The novelty is the graph + bus + gates.

Threat ranking by novelty overlap:

| Tier | Models | Threat axis | Dream3R differentiation |
|---|---|---|---|
| HIGH | Spann3R, LONG3R/LongStream/LoGeR, VGGT | Persistent memory; long-sequence; strong single-pass | A1 multi-mode update; bus + CR-rules; multi-module control |
| MEDIUM | CUT3R/STream3R, Test3R, MonST3R, SLAM3R, Mamba-3R | Single-axis overlap | Broader action set; explicit bus; composable modules |
| LOW | DUSt3R, MASt3R, Fast3R, TTT3R, MapAnything, 4DGS | Foundation or orthogonal | Dream3R builds on these, does not compete |

## 7. Discussion

### 7.1 Evidence status

The aggregate evidence distribution of Dream3R v0.1:

- ~5 elements paper-proven (perception substrate, token outputs, per-frame dynamic split, slot attention outside 3R)
- ~5 elements paper-derived (SSM-for-3R-memory, Critic substrate pattern, Composer routing pattern)
- ~10 elements inferred (per-action heads, per-module compositions, substrate hypothesis)
- ~7 elements architecture-novel (bus, CR-1..CR-6 as gates, substrate composition)
- 2 elements speculative (A7/A8 reserved hooks)

The architecture-novel elements carry the highest paper novelty but also the highest risk: they are untested by definition.

### 7.2 Limitations

1. No training has been conducted. All parameter counts and performance expectations are inferred.
2. The substrate hypothesis (hybrid > single-substrate) is untested; ABL-2 is the critical experiment.
3. If ABL-1 (bus removal) shows the bus is inert, the architecture story collapses to "modular composition with conventions."
4. A7 (Cross-Modal) and A8 (Active Perception) are reserved hooks, not designed mechanisms.
5. Object_track_set identity consistency has no 3R-specific training target; the loss function is speculative.
6. The 17-signal evidence vector V1 may contain redundant signals; ABL-9 tests this.
7. CR-rule firing rates on natural benchmarks are unknown; adversarial B6 inputs may be needed to demonstrate bus utility.

### 7.3 Risks

- R1: Substrate hypothesis falsification at training time
- R2: Bus-as-novelty collapse if CR-rules never fire
- R3: State-ownership invariant violation under gradient flow
- R5: Storytelling vs measurement asymmetry
- R7: "Control graph" framing may be read as "modular composition" without the bus gates

## 8. Conclusion

Dream3R proposes that the bottleneck for hard 3R cases is not a better encoder but a missing control vocabulary over memory, verification, dynamics, and routing. The architecture makes this vocabulary first-class: four modules sharing a typed memory bus with conflict-resolution gates. Every architectural claim maps to a falsifiable ablation experiment.

The contribution is the graph — not any single node.

---

## Evidence discipline notes

```text
- Every section carries evidence labels per Discipline rule 5
- No measured performance is claimed
- Comparator claims are restricted to what papers report in
  abstracts/results
- Four finalists are treated as parallel modules, not contestants
  for a single thesis spine (DEC-20260504-002 still in force)
- Dream3R remains a candidate, not final thesis
  (DEC-20260501-004 still in force)
```

## Version history

```text
v1  2026-05-06  cycle 017. Architecture-centric paper rewrite per
                DEC-20260506-001. Central contribution = Dream3R
                architecture (control-graph-as-architecture). Sections
                1-8 drafted. Supersedes PAPER_PHASE2_BLUEPRINT.md as
                primary paper artifact. No training; no measured
                results; all claims carry evidence labels.
```
