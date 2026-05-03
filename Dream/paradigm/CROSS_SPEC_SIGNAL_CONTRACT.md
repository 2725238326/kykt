# Cross-Spec Signal Contract

Last updated: 2026-05-04 (cycle 008.5; v1; not yet exercised by case cards)

Status: active contract; revision policy below.

## Purpose

The four finalist specs (Critic / Memory / Permanence / Composer) each own a small action subset and publish read-only signals that other specs consume. Without a written contract, each spec's case-card pass in cycle 009 would silently invent assumptions about what the others provide.

This file is the formal contract. It defines:

1. who publishes which signal
2. who consumes it
3. the contract type (read-only, handoff, or no-cross)
4. how conflicts between specs are resolved
5. how the contract itself is versioned

This contract is `inferred`. It has not been exercised by L2 case cards. Cycle 009 case cards are the first test. Corrections produce a new revision rather than a silent edit, per `paradigm/RESEARCH_CODE_DISCIPLINE.md` rule 5 (Honesty Override).

## Scope

The contract covers only the four finalist specs and the cross-modal / active-perception specs that may be drafted in later cycles. It does not cover:

- KYKT app contracts (Advisor / Sample Matrix / runner schemas; those are separate)
- backend service contracts (out of scope for cycle 008.5)
- frontend handoff contracts (handled by `handoff/FRONTEND_DESIGN_HANDOFF_PROMPT.md`)

The contract is consumed at L2 case-card time. It is not consumed by L1 design content; specs may reference the contract abstractly without requiring case-card data.

## Signal Owner Table

The following table lists every signal that crosses spec boundaries. Signals defined inside a single spec and not consumed elsewhere are not listed here.

| Signal | Producer SPEC | Consumer SPEC(s) | Contract type | First exercised by |
|---|---|---|---|---|
| `conflict_score` | Critic SPEC-20260503-001 | Memory SPEC-20260503-002, Permanence SPEC-20260503-003 | read-only | cycle 009 case cards |
| `route_history` | Critic SPEC-20260503-001 | Composer SPEC-20260504-001 | read-only | cycle 009 case cards |
| `recommended_action` | Critic SPEC-20260503-001 | Memory, Permanence, Composer (informational) | read-only | cycle 009 case cards |
| `route_regret_estimate` | Critic SPEC-20260503-001 | Composer SPEC-20260504-001 | read-only | cycle 009 case cards |
| `latent_drift_proxy` | Memory SPEC-20260503-002 | Critic SPEC-20260503-001 | read-only | cycle 009 case cards |
| `anchor_set` | Memory SPEC-20260503-002 | Permanence (informational) | read-only | cycle 009 case cards |
| `policy_log` | Memory SPEC-20260503-002 | Composer (informational) | read-only | cycle 009 case cards |
| `dynamic_ratio` | Permanence SPEC-20260503-003 | Memory (gate for A1/A2), Critic (informational) | read-only | cycle 009 case cards |
| `object_track_stability` | Permanence SPEC-20260503-003 | Memory (informational) | read-only | cycle 009 case cards |
| `suppress_static_write(r)` | Permanence SPEC-20260503-003 | Memory (handoff: Memory must honor) | handoff | cycle 009 case cards |
| `admit_static_write(r)` | Permanence SPEC-20260503-003 | Memory (handoff: Memory's A2 write pipeline) | handoff | cycle 009 case cards |
| `pollution_log` | Permanence SPEC-20260503-003 | Composer (informational) | read-only | cycle 009 case cards |
| `capability_match` | Composer SPEC-20260504-001 | Critic SPEC-20260503-001 (gate for A5 reroute_model) | read-only | cycle 009 case cards |
| `capability_card` | Composer SPEC-20260504-001 | Critic, Memory, Permanence (informational) | read-only | cycle 009 case cards |
| `sample_regime_card` | Composer SPEC-20260504-001 | Critic, Memory, Permanence (informational) | read-only | cycle 009 case cards |
| `route_recommendation` | Composer SPEC-20260504-001 | Critic A5 (when Critic decides reroute) | handoff | cycle 009 case cards |

Contract type definitions:

- **read-only**: consumer may read the value to inform its own decisions but must not mutate, override, or echo it back as authoritative
- **handoff**: producer emits an action the consumer must execute (or explicitly refuse with a logged reason)
- **no-cross**: signal is internal to the producer; not in this table

## Per-SPEC Published Signals

This section restates the published surface of each finalist spec so a consumer does not have to mine the full SPEC body.

### Critic SPEC-20260503-001 publishes

- `conflict_score(t)`: scalar derived from {pose_novelty, view_overlap, reprojection_residual, pointmap_conflict, confidence_drop, prior_rgb_conflict}; range and threshold `theta_conflict` are inferred per Critic spec
- `recommended_action`: one of {accept, rerun_local_region, reroute_model, open_anchor_budget, request_prior, conflict_unresolved}
- `route_history(t)`: list of (model, action) pairs already tried for this input window
- `route_regret_estimate`: gap between chosen route's `capability_match` and best-known `capability_match` for this regime; computed from Composer's published `capability_card`

What Critic does NOT publish:

- learned weights for `conflict_score` aggregation (none exist; weights are inferred)
- a binary "this output is wrong" verdict (Critic emits a score and recommended action only)

### Memory SPEC-20260503-002 publishes

- `latent_drift_proxy`: combination of pose_novelty + confidence_drop drift over windows
- `anchor_set(t)`: set of protected anchor indices in the memory store
- `cache_window(t)`: bounded sliding cache window contents (informational)
- `policy_log(t)`: append-only log of A1/A2/A3 sub-action choices

What Memory does NOT publish:

- the memory store contents themselves (Memory does not own the store; it owns the policy)
- a cross-job memory state (Memory's contract is per-job)

### Permanence SPEC-20260503-003 publishes

- `dynamic_ratio(r, t)`: per-region dynamic-content ratio
- `object_track_stability(o, t)`: per-object identity-confidence trace
- `suppress_static_write(r)`: handoff to Memory; Memory must honor or log refusal
- `admit_static_write(r)`: handoff to Memory's A2 write pipeline
- `pollution_log(t)`: append-only log of suppress / admit / defer decisions
- `object_track_set(t)`: set of `{object_id, last_seen_t, last_position, identity_confidence}` records

What Permanence does NOT publish:

- per-frame motion fields beyond `dynamic_horizon` (evicted)
- 4DGS asset descriptors (out of scope per SPEC-003 boundaries)

### Composer SPEC-20260504-001 publishes

- `capability_card(model_id)`: per-model capability profile across input regimes (static pair, many-view, streaming, dynamic, sparse-view, etc.)
- `sample_regime_card(input)`: per-input regime classification
- `capability_match(model_id, input)`: scalar match score in [0, 1] from card join
- `route_recommendation(input)`: ordered list of model_ids from best to worst expected capability_match
- `route_regret(chosen, input)`: gap between chosen model's match and best-known model's match

What Composer does NOT publish:

- model accuracy claims (capability cards encode regime fit, not measured accuracy)
- a single "best model" selection unless asked via `route_recommendation`

## Conflict Resolution Rules

When two specs disagree, the contract specifies which spec's decision wins.

### Rule CR-1: Critic A5 reroute requires Composer agreement on capability_match spread

If Critic says `reroute_model` for an input but Composer's `capability_match` has zero spread across the comparator pool for that regime (i.e. no alternative model has higher expected match), Critic must downgrade to `conflict_unresolved` and surface to Advisor. Critic does not invent a model that Composer has not characterized.

Rationale: A5 reroute is meaningful only when the alternative is differentiated. Allowing Critic to override would silently bake un-versioned capability assumptions into reroute decisions.

### Rule CR-2: Permanence suppress_static_write is binding on Memory

If Permanence emits `suppress_static_write(r)`, Memory's A2 must apply it. Memory may NOT override and write the suppressed region into the static map even if its own `write_value_estimate` would otherwise admit the write.

If Memory cannot honor (e.g. structural limitation), Memory logs `cross_spec_refusal{producer=Permanence, signal=suppress_static_write, reason=...}` and surfaces to Advisor. Silent override is a contract violation.

Rationale: the dynamic-pollution claim collapses if Memory ignores Permanence. A refusal must be visible.

### Rule CR-3: Memory drift signal does not gate Critic verification

If Memory's `latent_drift_proxy` is high but Critic's per-window evidence vector shows no `pointmap_conflict` or `reprojection_residual`, Critic does NOT auto-trigger A5 just from drift. Drift is informational for Critic. The A5 trigger condition remains the conjunction of `conflict_score(t) > theta_conflict` and the other Critic-internal preconditions.

Rationale: Memory and Critic must remain falsifiable independently. If drift could trigger A5 directly, Memory's P2/P3 results and Critic's P1/P5 results would couple, defeating cycle 009's parallel-track design.

### Rule CR-4: Composer route_recommendation does not bind Critic when capability_match is tied

If Composer's top-1 and top-2 `capability_match` are within `epsilon_tie` (default 0.05; inferred), Composer publishes both as candidates. Critic's A5 reroute_model picks among them by Critic-internal preference (e.g. preference for already-cached models, or for models whose route_history has not been tried this window). Composer does not force the choice.

Rationale: route_regret is informative as a spread, not as a forced ranking under noise.

### Rule CR-5: All cross-spec signals carry their producer's evidence label

A signal's evidence label propagates with it. If Memory's `latent_drift_proxy` is `inferred`, Critic must treat it as `inferred` when reading it. Critic must not silently upgrade an inferred input to paper-proven status downstream. This is a direct application of `paradigm/RESEARCH_CODE_DISCIPLINE.md` rule 5.

### Rule CR-6: Cycle 009 case cards record contract usage

Each cycle 009 case card must list which cross-spec signals it consumed and what the producer's evidence label was at consumption time. If any consumed signal had label `unknown`, the case card carries a caveat. This is the test path for the contract itself.

## Versioning

This contract is versioned. The current version is **v1**.

Revision rules:

- A new signal added, removed, or repurposed produces a new version (v2, v3, ...).
- A change in conflict resolution rules produces a new version.
- A change in evidence-label propagation rules produces a new version.
- Pure typo fixes do not produce a new version.

Each version records:

- date
- summary of change
- which specs are affected
- which case cards exercised the change

When a new version supersedes an older one, the older version is preserved in this file under a "Superseded versions" section rather than deleted. Discipline rule 5 (Honesty Override): retracted contract clauses must be visible, not silently overwritten.

## v1 Change Log

- 2026-05-04 (cycle 008.5): initial contract drafted alongside `decisions/DEC-20260504-001-composer-finalist-upgrade.md`. v1 covers Critic / Memory / Permanence / Composer published signals and six conflict resolution rules. Not yet exercised. Cycle 009 case cards will be the first exercise.

## Superseded Versions

(none yet)

## Out Of Scope For This Contract

- learned signal aggregation (no learned weights for any signal in v1; all weights are inferred)
- multi-job state propagation (each contract instance is per-job; cross-job memory consolidation is a separate future contract)
- KYKT runner / Advisor / Sample Matrix integration (separate contract; see `handoff/FRONTEND_DESIGN_HANDOFF_PROMPT.md` and the workflow status)
- Cross-Modal A7 ownership (still gated; the Cross-Modal spec is not drafted yet)
- Active Perception A8 ownership (still gated; the Active Perception spec is not drafted yet)

## Companion Files

- `paradigm/RESEARCH_CODE_DISCIPLINE.md` — the discipline rules this contract enforces, especially rules 3 (Surgical Edits) and 5 (Honesty Override)
- `planning/ACTION_TAXONOMY_AND_PROXY_METRICS.md` — A1-A8 actions, P1-P8 proxies, evidence signal vector
- `planning/RESEARCH_GRAPH_AND_PAPER_START.md` — F1-F6 failure modes and C1-C16 composition edges
- `specs/SPEC-20260503-001-geometry-critic.md` — Critic spec; reads Composer `capability_match`, publishes `conflict_score` and friends
- `specs/SPEC-20260503-002-executive-memory.md` — Memory spec; reads `conflict_score`, `dynamic_ratio`, honors `suppress_static_write`
- `specs/SPEC-20260503-003-dynamic-object-permanence.md` — Permanence spec; publishes `dynamic_ratio`, `suppress_static_write`, `object_track_stability`
- `specs/SPEC-20260504-001-3r-composer.md` — Composer spec; publishes `capability_card`, `capability_match`, `route_recommendation`
- `planning/WORK_RISK_REGISTER.md` — consolidated risk view including contract drift risk
