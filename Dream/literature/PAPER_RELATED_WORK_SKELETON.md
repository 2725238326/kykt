# Paper Related Work Skeleton

Last updated: 2026-05-04 (cycle 008.5; v1; skeleton only, no prose)

Status: skeleton only. Updates as case cards land in cycle 009 and beyond.

## Purpose

A scaffold for the eventual paper's related-work and positioning sections, organized around the F1-F6 failure-mode taxonomy from `planning/RESEARCH_GRAPH_AND_PAPER_START.md` rather than around individual mechanisms. The argument structure follows DEC-20260504-002's no-all-in posture: the four finalists are parallel mechanisms over a shared failure-mode taxonomy, not contestants for thesis spine.

This file is not a finished related-work section. It is the placement skeleton; prose comes later.

## Top-Level Structure

```text
1. Field Framing
   - DUSt3R-style 3R foundation models changed 3D reconstruction.
   - The follow-up landscape is fragmented across F1-F6.
   - The next bottleneck is a control vocabulary, not a single backbone.

2. Failure Mode Taxonomy (F1-F6)
   - one section per failure mode, listing the partial-solution papers
     drawn from the SPINE files.

3. Dream Contribution
   - regime-typed control over A1-A8 actions.
   - four parallel mechanisms (Critic / Memory / Permanence / Composer).
   - cross-spec signal contract as an integrating layer.

4. What This Paper Does And Does Not Claim About Itself
5. What We Add
```

## Section Skeletons

### Section 1: Field Framing

Anchor papers: DUSt3R (SRC-2024-001) as origin; MASt3R / Fast3R / Spann3R / MonST3R / CUT3R / STream3R as the immediate diversification.

Argument: the field is past "single dominant model" stage; partial solutions cluster around F1-F6; no single backbone integrates them.

### Section 2: F1 Long-Context Drift / Forgetting

Drawn from `literature/SPINE_MEMORY.md` required + advanced reading.

Papers to place:

- CUT3R (SRC-2025-002) — persistent state
- STream3R (SRC-2026-001) — causal streaming
- LONG3R (SRC-2025-012) — long-sequence memory gating
- LoGeR (SRC-2026-002) — hybrid memory
- Mem3R (SRC-2026-003) — hybrid KV + map
- OVGGT (SRC-2026-007) — anchor cache
- PAS3R (SRC-2026-004) — pose-adaptive update
- FILT3R (SRC-2026-005) — Kalman filtering
- Point3R (SRC-2025-003) — external pointer
- LongStream (SRC-2026-006) — gauge-decoupled streaming

Argument: each fixes one update / cache / store rule; Dream Memory's contribution is *policy bank over evidence vector*.

### Section 3: F2 Dynamic-Static Entanglement

Drawn from `literature/SPINE_PERMANENCE.md`.

Papers to place:

- MonST3R (SRC-2024-003) — dynamic-aware loss
- POMATO (SRC-2025-010) — dynamic-aware loss with different target
- D2USt3R (SRC-2025-011) — dynamic-aware token routing
- Easi3R (SRC-2025-013) — training-free per-frame
- RayMap3R (SRC-2026-008) — ray-based dynamic 3R
- 4DGS variants — asset axis (cited as out-of-scope counterpart)

Argument: per-frame motion accuracy is the typical metric; Dream Permanence's contribution is *object identity persistence + static-map immunity*.

### Section 4: F3 Hard-Case Geometric Ambiguity

Drawn from `literature/SPINE_CRITIC.md`.

Papers to place:

- Test3R (SRC-2025-007) — in-family consistency self-check
- TTT3R (SRC-2025-004) — test-time training trigger
- CTRL (SRC-2025-008) — critic-revision pattern (LM domain)
- MASt3R-SfM (SRC-2024-009) — classical SfM-stage refinement
- SLAM3R (SRC-2024-010) — sliding-window SLAM consistency loop
- G-CUT3R (SRC-2025-014) — guided priors with conflict potential

Argument: existing critics report or update inside one model family; Dream Critic's contribution is *cross-model A5 reroute action set bound to Composer's capability_match*.

### Section 5: F4 Passive Observation Limit

Active perception family (deferred; specs not yet drafted).

Papers cited but not as active comparators:

- ActiveNeRF (SRC-2022-001)
- FisherRF (SRC-2024-017)
- ActiveSplat (SRC-2024-018)
- ActiveGS (SRC-2024-019)

Argument: active perception is on the canvas at lower priority; A8 ownership is reserved.

### Section 6: F5 Sensor / Modality Fragility

Cross-modal family (deferred; specs not yet drafted).

Papers cited:

- DEVO (SRC-2023-004) — event-only VO
- Depth Anything V2 (SRC-2024-014), Depth Pro (SRC-2024-015), Metric3D v2 (SRC-2024-016) — depth priors
- DINOv2 (SRC-2023-002), SAM 2 (SRC-2024-012), CoTracker (SRC-2023-003), SpatialTracker (SRC-2024-013) — visual priors
- G-CUT3R (SRC-2025-014) — guided 3R

Argument: prior arbitration is a research axis; A7 ownership is reserved for a future Cross-Modal spec.

### Section 7: F6 Fragmented Model Ecology

Drawn from `literature/SPINE_COMPOSER.md`.

Papers to place:

- DUSt3R, MASt3R, MASt3R-SfM, Fast3R, Spann3R, MonST3R, CUT3R, STream3R, SLAM3R, MV-DUSt3R+, Splatt3R, InstantSplat, NoPoSplat (the comparator pool)
- Mixture-of-experts and routing literature (cross-domain analog; cited as related work for the routing pattern, not as 3R comparators)

Argument: the field is fragmented across regimes; Dream Composer's contribution is *regime-typed route_regret falsification axis*, the first 3R-specific routing metric.

### Section 8: What This Paper Does And Does Not Claim About Itself

Reserved. Draft after cycle 009 case cards land.

Bullet skeleton:

- "This paper claims an integrating control vocabulary over the F1-F6 partial solutions, not a single dominant model."
- "This paper does not claim a learned router, learned critic, or learned memory policy; the case-card evidence is policy-design level, not learned controller."
- "This paper does not claim teacher-demo readiness in any specific KYKT navigation surface; demo paths are described, not implemented."
- "Specific evidence labels and case-card outcomes will populate this section after cycle 009."

### Section 9: What We Add

Reserved. Draft after cycle 009 case cards land.

Bullet skeleton:

- A4 / A5 / A6 / A1+A2+A3 / A5-routing decomposition over F1-F6.
- Cross-spec signal contract as the integrating layer.
- L2 case-card methodology (no reproduction, no checkpoint download) as a contribution to *how* 3R research is positioned.
- Specific contributions for each finalist will be filled after the case cards exercise the proxies.

## Update Rule

- When a SPINE file changes spine ordering or evidence labels, update the corresponding section here.
- When a case card lands and the cycle log records a falsification result, update Section 8 / 9.
- Do not write prose in this file until cycle 010+ (case cards in cycle 009 are the gate).
- The skeleton is permitted to be opinionated about positioning; it is not permitted to invent unverified claims about any cited paper. Discipline rule 5 governs.
