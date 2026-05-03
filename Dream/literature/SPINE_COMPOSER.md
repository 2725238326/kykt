# SPINE_COMPOSER: 3R Composer / Unified Model Ecology

Last updated: 2026-05-04 (cycle 008.5; v1; created with the Composer finalist upgrade DEC-20260504-001)

Linked spec: `specs/SPEC-20260504-001-3r-composer.md`

Linked finalist: 3R Composer / Unified Model Ecology

## One-Line Definition

A 3R routing controller that publishes regime cards (per-input regime classification) and capability cards (per-model regime fit), and exposes a `route_regret` falsification axis for the choice between models. Composer owns the routing facet of A5 only; the repair facet of A5 lives in Critic.

Important: per DEC-20260504-002, Composer is not the thesis spine, even if cycle 009 case-card data lands strongly. The capability-card axis is one of four borrowable components.

## Required Reading

These are the papers any Composer case-card author must read before adding a `capability_match` row.

### SRC-2024-001 DUSt3R (paper-proven)

What this paper actually claims: pose-free pointmap reconstruction; the founding paper of the family.

What people often misread it as: a single dominant model. DUSt3R was the start; the 2024-2026 follow-ups carved up the regime space. Composer's regime card argues that no single descendant covers all regimes.

### SRC-2024-002 MASt3R (paper-proven)

What this paper actually claims: 3D-grounded matching with sparse global alignment; matching head is the contribution.

What people often misread it as: replacing DUSt3R. MASt3R extends DUSt3R for matching-style regimes; the regime card encodes this.

### SRC-2025-001 Fast3R (paper-proven)

What this paper actually claims: many images in one forward pass; the regime is many-view.

What people often misread it as: a faster DUSt3R. Fast3R's contribution is the many-view regime fit, not raw speed; the capability card should reflect this.

### SRC-2024-011 Spann3R (paper-proven)

What this paper actually claims: spatial memory for global pointmap prediction; the regime is streaming-with-memory.

What people often misread it as: a memory paper that competes with Mem3R. Spann3R is a 3R model with memory; Mem3R is a hybrid memory architecture. Composer's regime card distinguishes them by regime fit, not memory size.

### Mixture-of-experts and routing literature (cross-domain analog; cited as inferred for 3R use)

What this literature claims: sample-conditioned routing among experts in language and vision.

What people often misread it as: a Composer comparator. MoE is sample-conditioned but not regime-typed; Composer's regime cards are 3R-specific (static_pair, many_view, dynamic_video, streaming, sparse_view). The borrow is the routing pattern, not the implementation.

## Advanced Reading

These are useful for second-round capability-card work.

### SRC-2024-009 MASt3R-SfM (paper-proven)

Matching + retrieval + global SfM. Useful for case-03 (many-view vs pair regime distinction). Capability card under "static_pair" is high; under "streaming" is low.

### SRC-2024-003 MonST3R (paper-proven)

Dynamic-video pointmap. Capability card under "dynamic_video" is high; under "static_pair" is low. Used in case-02 to demonstrate regime-driven routing flips.

### SRC-2025-002 CUT3R (paper-proven)

Persistent state continuous 3D perception. Capability card under "streaming" is high; static-pair is low.

### SRC-2026-001 STream3R (paper-proven)

Causal streaming geometry. Capability card under "streaming" is high; the difference vs CUT3R is causal vs persistent (see SPINE_MEMORY).

### SRC-2024-010 SLAM3R (paper-proven)

Sliding-window SLAM-shaped consumer of 3R outputs. Useful as a counter-example: SLAM3R is a *consumer* of 3R, not a 3R model family member. The capability card should not list SLAM3R as a peer to MASt3R; it operates downstream.

### SRC-2025-005 MV-DUSt3R+ (paper-proven)

Sparse-view multiview pose-free RGB reconstruction. Capability card under "sparse_view" is high; many-view is medium.

### SRC-2024-004 Splatt3R / SRC-2024-005 InstantSplat / SRC-2025-006 NoPoSplat (paper-proven)

Asset-path comparators. Their capability cards apply to the "asset_output" regime; Composer routes asset-path samples to them. They are not 3R-only models; they include Gaussian asset generation.

## Skip With Reason

- robotics / VLA / active perception papers: out of scope; Composer routes 3R inputs, not embodied actions. Active perception A8 is a separate spec.
- pure benchmark / leaderboard papers without per-regime breakdown: cite as background only; capability cards need regime-typed claims, not aggregate metrics.
- VLM / scene-understanding papers without 3R routing: skip unless adding a CRITICAL_NOTES.md entry.

## Cross-Paper Disagreement

- **What is a "regime"?** No published 3R paper defines a regime taxonomy explicitly. Composer's regime list (static_pair, many_view, dynamic_video, streaming, sparse_view, asset_output) is an `inferred` taxonomy drawn from the union of comparator papers' "scope" sections. Different papers implicitly assume different regime cuts.
- **MASt3R vs MASt3R-SfM as separate entries**: should they be one capability card or two? Composer treats them as two because their input regimes differ (matching vs SfM-aligned multiview). Some readers expect one card per published paper rather than per regime variant; the distinction is `inferred`.
- **MoE-routing as comparator**: cycle 009 must be careful here. MoE is the closest cross-domain analog but is not a 3R comparator. Treat MoE references as related-work positioning, not as a "Composer comparator".

## Interface To SPEC-20260504-001

- Composer publishes `capability_card`, `sample_regime_card`, `capability_match`, `route_recommendation`, and `route_regret` per `paradigm/CROSS_SPEC_SIGNAL_CONTRACT.md`.
- Critic A5 reroute_model consumes `route_recommendation`; per CR-1, if `capability_match` has zero spread, Critic must downgrade to `conflict_unresolved` rather than reroute.
- P5 route_regret is the primary Composer falsification axis.
- Capability cards are paper-derived (paper-proven for per-paper claims) but the regime card weighting and the join is `inferred`.

## Evidence Labels Summary

- DUSt3R, MASt3R, MASt3R-SfM, Fast3R, Spann3R, MonST3R, CUT3R, STream3R, SLAM3R, MV-DUSt3R+, Splatt3R, InstantSplat, NoPoSplat: paper-proven for their published per-regime claims.
- Composer regime taxonomy (static_pair / many_view / dynamic_video / streaming / sparse_view / asset_output): inferred.
- capability_match weights, regime probability weights, epsilon_tie: inferred.
- MoE routing borrowed pattern: inferred for 3R use.
